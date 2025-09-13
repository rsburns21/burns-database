# server.py
# Burns Database MCP Server — Streamable HTTP (stateless) + Supabase/Edge tools
# Auth is handled by FastCloud (OAuth in prod or bearer in staging). No in-app bearer middleware.
# Version: 5.1.0

from __future__ import annotations

import os
import sys
import time
import json
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from fastmcp import FastMCP
from fastmcp.server.http import create_streamable_http_app

# ---------------------------
# Identity & Environment
# ---------------------------

SERVER_NAME = os.getenv("APP_NAME", "burns-database").strip()
SERVER_VERSION = os.getenv("APP_VERSION", "5.1.0").strip()
MCP_PATH = "/mcp"

# Supabase & Edge
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    or ""
)
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip() or os.getenv("SUPABASE_KEY","").strip()

# IMPORTANT: Use the exact working functions URL if set; otherwise default to project endpoint
EDGEFN_URL = os.getenv(
    "EDGEFN_URL",
    "https://nqkzqcsqfvpquticvwmk.functions.supabase.co/advanced_semantic_search",
).strip()

# Optional OpenAI (used only for diagnostics/rerank ping)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Allowlist for outbound calls
ALLOWLIST_DEFAULT = [
    "nqkzqcsqfvpquticvwmk.supabase.co",
    "nqkzqcsqfvpquticvwmk.functions.supabase.co",
    "api.openai.com",
    "chatgpt.com",
    "raw.githubusercontent.com",
    "httpbin.org",
    "playground.ai.cloudflare.com",
    "example.com",
]
ALLOWLIST_HOSTS: List[str] = [
    h.strip().lower()
    for h in (os.getenv("ALLOWLIST_HOSTS", ",".join(ALLOWLIST_DEFAULT))).split(",")
    if h.strip()
]

# Tunables
MAX_FETCH_SIZE = int(os.getenv("MAX_FETCH_SIZE", "1000000"))
SNIPPET_LENGTH = int(os.getenv("SNIPPET_LENGTH", "500"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "25"))

# Search defaults (edge function accepts both snake_case and camelCase; we send both)
SEARCH_MODE = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
# Non-zero lexical fallback ensures BM25 is attempted even if vector path is unavailable
LEXICAL_K = int(os.getenv("LEXICAL_K", "100"))

# Diagnostics toggle
ENABLE_DIAG = (os.getenv("ENABLE_DIAG", "0").strip() in ("1","true","True","yes","Y"))

# ---------------------------
# Optional Supabase client (for direct table fetches and keyword fallback)
# ---------------------------

try:
    from supabase import create_client, Client as SupabaseClient  # type: ignore
except Exception as e:
    print("[BDB] Supabase SDK import failed:", e, file=sys.stderr)
    create_client = None  # type: ignore
    SupabaseClient = None  # type: ignore

_SUPABASE: Optional["SupabaseClient"] = None
if create_client and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        _SUPABASE = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("[BDB] Supabase client initialized.", file=sys.stderr)
    except Exception as e:
        print("[BDB] Supabase client init failed:", e, file=sys.stderr)
else:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[BDB] Supabase client not initialized (missing SUPABASE_URL or SERVICE_ROLE_KEY).", file=sys.stderr)

# ---------------------------
# Helpers
# ---------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _truncate(s: str, n: int = SNIPPET_LENGTH) -> str:
    if not s:
        return s
    return s if len(s) <= n else s[: n - 3] + "..."

def _resolve_edge_url() -> str:
    # Support either /functions/v1/<fn> or functions.supabase.co/<fn>; normalize trailing slashes
    u = EDGEFN_URL.rstrip("/")
    return u

def _is_host_allowlisted(url: str) -> Tuple[bool, str]:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        return (host in ALLOWLIST_HOSTS, host)
    except Exception:
        return (False, "")

def make_edge_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY
    if key:
        headers["apikey"] = key
        headers["Authorization"] = f"Bearer {key}"
    # Do not set x-authless unless you intentionally want service-role without Authorization
    return headers

async def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, Any], str]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        resp = await client.post(url, json=body, headers=headers)
        text = resp.text
        try:
            data = resp.json()
        except Exception:
            data = {}
        return (resp.status_code, data, text)

def _normalize_exhibit_id(eid: str) -> str:
    s = (eid or "").strip()
    if s.lower().startswith("exhibit_"):
        num = s[len("exhibit_"):].lstrip("0") or "0"
        return f"Ex{int(num):03d}"
    if s.lower().startswith("ex") and s[2:].isdigit():
        return f"Ex{int(s[2:]):03d}"
    return s

def _categorize_exhibit(description: str, filename: str = "") -> str:
    t = (description or "").lower() + " " + (filename or "").lower()
    if any(k in t for k in ("contract", "agreement")):
        return "KEY_CONTRACT"
    if any(k in t for k in ("email", "correspondence", "letter", "communication")):
        return "BREACH_EVIDENCE"
    if any(k in t for k in ("financial", "damages", "statement", "report", "audit")):
        return "DAMAGES"
    return "OTHER"

# ---------------------------
# Defaults
# ---------------------------

class Defaults(BaseModel):
    mode: str = SEARCH_MODE
    top_k: int = SEARCH_TOP_K
    min_score: float = SEARCH_MIN_SCORE
    lexical_k: int = LEXICAL_K
    max_fetch_size: int = MAX_FETCH_SIZE
    snippet_length: int = SNIPPET_LENGTH

DEFAULTS = Defaults()

# ---------------------------
# FastMCP server (stateless HTTP)
# ---------------------------

mcp = FastMCP(name=SERVER_NAME, instructions=f"{SERVER_NAME} MCP server")

# ---------------------------
# Tool schemas
# ---------------------------

class SearchParams(BaseModel):
    q: str = Field(..., description="User query")
    mode: str = Field(SEARCH_MODE, description="vector|bm25|hybrid|semantic")
    top_k: int = Field(SEARCH_TOP_K, ge=1, le=200)
    min_score: float = Field(SEARCH_MIN_SCORE, ge=0.0)
    rerank: bool = Field(True, description="Enable rerank if Edge supports it")
    label: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None

class LabelParams(BaseModel):
    doc_id: str
    label: str

class FetchParams(BaseModel):
    url: str

class AllowlistTestParams(BaseModel):
    url: str

class SetDefaultsParams(BaseModel):
    mode: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=200)
    min_score: Optional[float] = Field(None, ge=0.0)
    lexical_k: Optional[int] = None
    max_fetch_size: Optional[int] = None
    snippet_length: Optional[int] = None

class EdgeActionParams(BaseModel):
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------
# Utility / Diagnostics tools
# ---------------------------

@mcp.tool(description="Echo text back.")
async def echo(text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"echo": text, "ts": _now_ms()}

@mcp.tool(description="List server capabilities and config.")
async def list_capabilities(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "edge": {"url": _resolve_edge_url()},
        "defaults": DEFAULTS.model_dump(),
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "openai_enabled": bool(OPENAI_API_KEY),
        "supabase_url_set": bool(SUPABASE_URL),
    }

@mcp.tool(description="Return version info.")
async def version_info(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"name": SERVER_NAME, "version": SERVER_VERSION}

@mcp.tool(description="Test if a URL host is allowlisted.")
async def allowlist_test(params: AllowlistTestParams) -> Dict[str, Any]:
    ok, host = _is_host_allowlisted(params.url)
    return {"ok": ok, "host": host, "allowlist": ALLOWLIST_HOSTS}

@mcp.tool(description="HTTP GET (allowlisted only). Returns truncated text body.")
async def http_get_allowed(params: FetchParams) -> Dict[str, Any]:
    ok, host = _is_host_allowlisted(params.url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        r = await client.get(params.url)
        return {"ok": r.status_code == 200, "status": r.status_code, "host": host, "text": _truncate(r.text or "")}

@mcp.tool(description="HTTP HEAD (allowlisted only).")
async def http_head_allowed(params: FetchParams) -> Dict[str, Any]:
    ok, host = _is_host_allowlisted(params.url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        r = await client.head(params.url)
        return {"ok": r.status_code == 200, "status": r.status_code, "headers": dict(r.headers)}

@mcp.tool(description="HTTP POST JSON (allowlisted only). Returns JSON or text snippet.")
async def http_post_allowed(url: str, body_json: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ok, host = _is_host_allowlisted(url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        r = await client.post(url, json=body_json)
        try:
            data = r.json()
            return {"ok": r.status_code == 200, "status": r.status_code, "json": data}
        except Exception:
            return {"ok": r.status_code == 200, "status": r.status_code, "text": _truncate(r.text)}

@mcp.tool(description="Check server + edge + OpenAI health quickly.")
async def health(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "ts": _now_ms()},
        "edge": {},
        "openai": {},
    }
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        out["edge"] = {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}
    else:
        headers = make_edge_headers()
        status, data, text = await _post_json(edge_url, {"ping": True}, headers)
        out["edge"] = {"ok": status == 200, "status": status, "url": edge_url, "body": data if data else _truncate(text)}
    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                r = await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
                out["openai"] = {"ok": r.status_code in (200, 401, 403), "status": r.status_code}
        except Exception as e:
            out["openai"] = {"ok": False, "error": str(e)}
    else:
        out["openai"] = {"ok": False, "error": "OPENAI_API_KEY not set"}
    return out

@mcp.tool(description="Detailed edge diagnostic POST with optional payload.")
async def diag_edge(payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}
    status, data, text = await _post_json(edge_url, payload or {"ping": True}, headers)
    return {"ok": status == 200, "status": status, "url": edge_url, "json": data if data else {}, "text": _truncate(text)}

@mcp.tool(description="Show config and defaults (raw).")
async def diag_config(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "edge": {"url": _resolve_edge_url()},
        "env_presence": {
            "SUPABASE_SERVICE_ROLE_KEY": bool(SUPABASE_SERVICE_ROLE_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "SUPABASE_URL": bool(SUPABASE_URL),
            "EDGEFN_URL": bool(EDGEFN_URL),
        },
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "defaults": DEFAULTS.model_dump(),
        "http_timeout_s": HTTP_TIMEOUT_S,
        "enable_diag": ENABLE_DIAG,
    }

@mcp.tool(description="Quick allowlist/host report.")
async def diag_allowlist(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"allowlist_hosts": ALLOWLIST_HOSTS}

@mcp.tool(description="OpenAI connectivity smoke check.")
async def diag_openai(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"ok": False, "error": "OPENAI_API_KEY not set"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            r = await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
            return {"ok": r.status_code in (200, 401, 403), "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------
# Search & Retrieval via Edge
# ---------------------------

async def _edge_search_core(params: SearchParams) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}

    top_k = params.top_k or DEFAULTS.top_k
    # Send both snake_case and camelCase to match your Deno handler exactly.
    body: Dict[str, Any] = {
        "query": params.q,
        "q": params.q,
        "mode": (params.mode or DEFAULTS.mode).lower(),
        "searchType": (params.mode or DEFAULTS.mode).lower(),
        "k": top_k,
        "topK": top_k,
        "limit": top_k,
        "min_score": params.min_score if params.min_score is not None else DEFAULTS.min_score,
        "lexical_k": DEFAULTS.lexical_k,
        "lexicalK": DEFAULTS.lexical_k,
        "rerank": bool(params.rerank),
    }
    if params.label:
        body["label"] = params.label
    if params.metadata_filter:
        body["filter"] = params.metadata_filter

    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, body, headers)
    out: Dict[str, Any] = {"ok": status == 200, "status": status, "mode": body["mode"], "url": edge_url}
    if not out["ok"]:
        out["error"] = data if data else _truncate(text)
        return out
    out["raw"] = data
    return out

@mcp.tool(description="General search via Edge.")
async def edge_search(params: SearchParams) -> Dict[str, Any]:
    return await _edge_search_core(params)

@mcp.tool(description="BM25 search via Edge.")
async def bm25_search(params: SearchParams) -> Dict[str, Any]:
    params.mode = "bm25"
    return await _edge_search_core(params)

@mcp.tool(description="Vector similarity search via Edge.")
async def vector_search(params: SearchParams) -> Dict[str, Any]:
    params.mode = "vector"
    return await _edge_search_core(params)

@mcp.tool(description="Hybrid search (vector + BM25) via Edge.")
async def hybrid_search(params: SearchParams) -> Dict[str, Any]:
    params.mode = "hybrid"
    return await _edge_search_core(params)

@mcp.tool(description="Semantic search (alias to hybrid unless Edge defines differently).")
async def semantic_search(params: SearchParams) -> Dict[str, Any]:
    params.mode = "semantic"
    return await _edge_search_core(params)

@mcp.tool(description="Request embeddings for a query from Edge (action=embed_query).")
async def embed_query(q: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    status, data, text = await _post_json(edge_url, {"action": "embed_query", "query": q, "q": q}, headers)
    if status != 200:
        return {"ok": False, "status": status, "error": data if data else _truncate(text)}
    emb = data.get("embedding") or data.get("vector") or []
    return {"ok": True, "dim": len(emb), "embedding": emb}

# ---------------------------
# Simplified search wrappers (with safe lexical fallback)
# ---------------------------

@mcp.tool(description="Canonical search: return IDs + snippets for follow‑up fetch().")
async def search(query: str, limit: int = 10, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = await edge_search(SearchParams(q=query, mode=DEFAULTS.mode, top_k=limit, min_score=DEFAULTS.min_score, rerank=True))
    # Primary happy path (edge function returned results)
    items: List[Dict[str, Any]] = []
    if base.get("ok"):
        for r in (base.get("raw", {}) or {}).get("results", []):
            ex_id = r.get("exhibit_id") or r.get("id") or _normalize_exhibit_id(r.get("doc_id","")) or "Unknown"
            snippet = _truncate(r.get("snippet") or r.get("text") or r.get("chunk") or r.get("content") or "", n=DEFAULTS.snippet_length)
            score = r.get("score") or r.get("similarity")
            items.append({"id": ex_id, "snippet": snippet, "score": score})
    # Fallback: if edge returned 0 results but Supabase client is available, do BM25 text_search
    if not items and _SUPABASE:
        try:
            resp = (
                _SUPABASE.table("vector_embeddings")
                .select("exhibit_id,text")
                .text_search("text", query, {"config": "english", "type": "websearch"})
                .limit(max(1, min(int(limit), 50)))
                .execute()
            )
            for r in resp.data or []:
                ex_id = r.get("exhibit_id") or "Unknown"
                snippet = _truncate(r.get("text") or "", n=DEFAULTS.snippet_length)
                items.append({"id": ex_id, "snippet": snippet, "fallback": "bm25"})
        except Exception as e:
            # best effort; keep empty
            items = items
    return {"ok": True, "results": items}

# ---------------------------
# Supabase-backed helpers
# ---------------------------

@mcp.tool(description="Assign lightweight labels to results.")
async def label_results(results: List[Dict[str, Any]], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    labeled: List[Dict[str, Any]] = []
    for r in results:
        ex = r.get("exhibit_id") or r.get("id") or ""
        desc, fname = "", ""
        if _SUPABASE and ex:
            try:
                q = _SUPABASE.table("exhibits").select("description,filename").eq("exhibit_id", _normalize_exhibit_id(ex))
                resp = q.limit(1).execute()
                if resp.data:
                    rec = resp.data[0] or {}
                    desc = rec.get("description") or ""
                    fname = rec.get("filename") or ""
            except Exception:
                pass
        label = _categorize_exhibit(desc or r.get("snippet", ""), fname)
        lr = dict(r)
        lr["category"] = label
        labeled.append(lr)
    return {"ok": True, "results": labeled}

@mcp.tool(description="Retrieve full text pages for an exhibit.")
async def fetch_exhibit(exhibit_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "pages": []}
    eid = _normalize_exhibit_id(exhibit_id)
    try:
        chunks = (
            _SUPABASE.table("vector_embeddings")
            .select("text,page_start,page_end,chunk_index")
            .eq("exhibit_id", eid)
            .order("chunk_index", desc=False)
            .execute()
        ).data or []
    except Exception as e:
        return {"ok": False, "error": f"chunk_fetch_error:{e}", "pages": []}

    pages: List[str] = []
    cur_page = None
    buf = ""
    for ch in chunks:
        t = ch.get("text", "") or ""
        ps = ch.get("page_start")
        if ps is None:
            pages.append(t)
            continue
        if cur_page is None:
            cur_page, buf = ps, t
        elif ps == cur_page:
            buf += " " + t
        else:
            pages.append(buf)
            cur_page, buf = ps, t
    if buf:
        pages.append(buf)
    meta = {"exhibit_id": eid}
    try:
        meta_resp = _SUPABASE.table("exhibits").select("description,filename").eq("exhibit_id", eid).limit(1).execute()
        if meta_resp.data:
            meta.update({k: v for k, v in meta_resp.data[0].items() if k in ("description", "filename")})
    except Exception:
        pass
    return {"ok": True, **meta, "pages": pages}

@mcp.tool(description="List exhibits with optional categories.")
async def list_exhibits(withLabels: bool = False, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "exhibits": []}
    try:
        data = (_SUPABASE.table("exhibits").select("exhibit_id,description,filename").execute()).data or []
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "exhibits": []}
    out: List[Dict[str, Any]] = []
    for rec in data:
        ex = rec.get("exhibit_id")
        item = {"exhibit_id": ex, "description": rec.get("description", "")}
        if withLabels:
            item["category"] = _categorize_exhibit(rec.get("description", ""), rec.get("filename", ""))
        out.append(item)
    return {"ok": True, "exhibits": out}

@mcp.tool(description="Keyword/BM25 search via PostgREST text search on vector_embeddings.text.")
async def keyword_search(query: str, top_k: int = 10, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "items": []}
    try:
        resp = (
            _SUPABASE.table("vector_embeddings")
            .select("exhibit_id,text,similarity")
            .text_search("text", query, {"config": "english", "type": "websearch"})
            .limit(max(1, min(int(top_k), 100)))
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        return {"ok": False, "error": f"search_error:{e}", "items": []}
    items = [{"exhibit_id": r.get("exhibit_id", "Unknown"), "snippet": _truncate(r.get("text", ""))} for r in rows]
    return {"ok": True, "items": items}

@mcp.tool(description="Counts of key tables.")
async def get_case_statistics(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized"}
    stats: Dict[str, Any] = {}
    try:
        c = _SUPABASE.table("claims").select("id", count="exact").execute()
        stats["num_claims"] = c.count if hasattr(c, "count") and c.count is not None else len(c.data or [])
    except Exception as e:
        stats["num_claims"] = f"error:{e}"
    try:
        e = _SUPABASE.table("exhibits").select("id", count="exact").execute()
        stats["num_exhibits"] = e.count if hasattr(e, "count") and e.count is not None else len(e.data or [])
    except Exception as e2:
        stats["num_exhibits"] = f"error:{e2}"
    try:
        f = _SUPABASE.table("facts").select("id", count="exact").execute()
        stats["num_facts"] = f.count if hasattr(f, "count") and f.count is not None else len(f.data or [])
    except Exception as e3:
        stats["num_facts"] = f"error:{e3}"
    return {"ok": True, "statistics": stats}

# ---------------------------
# Defaults mgmt
# ---------------------------

@mcp.tool(description="Set session defaults (mode/top_k/min_score/etc).")
async def set_defaults(params: SetDefaultsParams, _extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if params.mode:
        DEFAULTS.mode = params.mode.lower()
    if params.top_k is not None:
        DEFAULTS.top_k = params.top_k
    if params.min_score is not None:
        DEFAULTS.min_score = params.min_score
    if params.lexical_k is not None:
        DEFAULTS.lexical_k = params.lexical_k
    if params.max_fetch_size is not None:
        globals()["MAX_FETCH_SIZE"] = int(params.max_fetch_size)
    if params.snippet_length is not None:
        globals()["SNIPPET_LENGTH"] = int(params.snippet_length)
    return {"ok": True, "defaults": DEFAULTS.model_dump()}

@mcp.tool(description="Get session defaults.")
async def get_defaults(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "defaults": DEFAULTS.model_dump()}

@mcp.tool(description="Reset session defaults to startup values.")
async def reset_defaults(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    DEFAULTS.mode = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()
    DEFAULTS.top_k = int(os.getenv("SEARCH_TOP_K", "10"))
    DEFAULTS.min_score = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
    DEFAULTS.lexical_k = int(os.getenv("LEXICAL_K", "100"))
    globals()["MAX_FETCH_SIZE"] = int(os.getenv("MAX_FETCH_SIZE", "1000000"))
    globals()["SNIPPET_LENGTH"] = int(os.getenv("SNIPPET_LENGTH", "500"))
    return {"ok": True, "defaults": DEFAULTS.model_dump()}

# ---------------------------
# Edge raw action
# ---------------------------

@mcp.tool(description="Call Edge with a raw action/payload (advanced).")
async def edge_action(params: EdgeActionParams, _extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    body = dict(params.payload or {})
    body["action"] = params.action
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    status, data, text = await _post_json(edge_url, body, headers)
    return {"ok": status == 200, "status": status, "json": data if data else {}, "text": _truncate(text)}

# ---------------------------
# HTTP routes (for CI & quick checks)
# ---------------------------

async def _root_ok(request: Request):
    return JSONResponse({"ok": True, "name": SERVER_NAME, "version": SERVER_VERSION, "ts": _now_ms()})

async def _health_get(request: Request):
    return JSONResponse({"name": SERVER_NAME, "version": SERVER_VERSION, "ok": True})

async def _health_head(request: Request):
    return PlainTextResponse("", status_code=200)

async def _diag_http(request: Request):
    return JSONResponse({
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
        "edge_url": _resolve_edge_url(),
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "supabase_url_set": bool(SUPABASE_URL),
        "openai_key_set": bool(OPENAI_API_KEY),
        "stateless_http": True,
        "enable_diag": ENABLE_DIAG,
    })

async def _supabase_health(request: Request):
    headers = {}
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY
    if key:
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    status_auth = 0
    status_rest = 0
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        try:
            auth_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/health" if SUPABASE_URL else ""
            if auth_url:
                r1 = await client.get(auth_url, headers=headers)
                status_auth = r1.status_code
        except Exception:
            status_auth = 0
        try:
            rest_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/" if SUPABASE_URL else ""
            if rest_url:
                r2 = await client.head(rest_url, headers=headers)
                status_rest = r2.status_code
        except Exception:
            status_rest = 0
    return JSONResponse({
        "auth_ok": (status_auth == 200),
        "rest_ok": (status_rest in (200, 204, 405, 404)),
        "status_auth": status_auth,
        "status_rest": status_rest,
    })

_routes = [
    Route("/", endpoint=_root_ok, methods=["GET"]),
    Route(f"{MCP_PATH}/health", endpoint=_health_get, methods=["GET"]),
    Route(f"{MCP_PATH}/health", endpoint=_health_head, methods=["HEAD"]),
    Route(f"{MCP_PATH}/diag_config", endpoint=_diag_http, methods=["GET"]),
    Route(f"{MCP_PATH}/tools/supabase_health", endpoint=_supabase_health, methods=["POST"]),
]

_middleware = [
    Middleware(CORSMiddleware,
               allow_origins=["*"],
               allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
               allow_headers=["Content-Type", "Authorization", "Accept", "Origin"]),
]

# Build the Streamable HTTP app
app = create_streamable_http_app(
    mcp,
    streamable_http_path=MCP_PATH,
    auth=None,              # Let FastCloud own authentication/OAuth
    json_response=True,
    stateless_http=True,
    routes=_routes,
    middleware=_middleware,
)

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
