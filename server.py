# server.py
# Burns Database MCP Server — FastMCP Streamable HTTP (stateless) + Supabase
# Fixes: remove deprecated BearerAuthProvider, add middleware-based Bearer auth
# Version: 4.4.2

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
SERVER_VERSION = os.getenv("APP_VERSION", "4.4.2").strip()

# Streamable MCP path
MCP_PATH = "/mcp"

# Secured readiness/CI token
BDB_AUTH_TOKEN = os.getenv("BDB_AUTH_TOKEN", "").strip()

# Supabase + Edge Function
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    or ""
)
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()  # optional catch-all
EDGEFN_URL = os.getenv(
    "EDGEFN_URL",
    "https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search",
).strip()

# OpenAI (only for diagnostics/rerank ping)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Outbound allowlist
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
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "20"))

SEARCH_MODE = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
LEXICAL_K = int(os.getenv("LEXICAL_K", "0"))
EMBED_DIM = 384

# ---------------------------
# Optional Supabase client
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
    return EDGEFN_URL

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
# FastMCP server (stateless)
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
    rerank: bool = Field(False, description="Enable rerank if Edge supports it")
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

class CodexAgentParams(BaseModel):
    action: str  # start|stop|status|restart|add_task
    task_data: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = None
    config_override: Optional[Dict[str, Any]] = None

# ---------------------------
# Tools — Utility / Diag
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
        "embed_dim": EMBED_DIM,
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
        },
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "defaults": DEFAULTS.model_dump(),
        "http_timeout_s": HTTP_TIMEOUT_S,
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
    body: Dict[str, Any] = {
        "query": params.q,
        "mode": (params.mode or DEFAULTS.mode).lower(),
        "k": top_k,
        "matchCount": top_k,
        "top_k": top_k,
        "min_score": params.min_score if params.min_score is not None else DEFAULTS.min_score,
        "lexical_k": DEFAULTS.lexical_k,
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
    status, data, text = await _post_json(edge_url, {"action": "embed_query", "query": q}, headers)
    if status != 200:
        return {"ok": False, "status": status, "error": data if data else _truncate(text)}
    emb = data.get("embedding") or data.get("vector") or []
    return {"ok": True, "dim": len(emb), "embedding": emb}

@mcp.tool(description="Label a document or match via Edge (action=label).")
async def label_result(params: LabelParams, _extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    body = {"action": "label", "doc_id": params.doc_id, "label": params.label}
    status, data, text = await _post_json(edge_url, body, headers)
    return {"ok": status == 200, "status": status, "json": data if data else {}, "text": _truncate(text)}

@mcp.tool(description="List collections/sources from Edge if supported.")
async def list_collections(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    payload: Dict[str, Any] = {"action": "list_collections"}
    if isinstance(params, dict):
        payload.update({k: v for k, v in params.items() if v is not None})
    status, data, text = await _post_json(edge_url, payload, headers)
    return {"ok": status == 200, "status": status, "json": data if data else {}, "text": _truncate(text)}

@mcp.tool(description="Fetch a document (by id) via Edge if supported.")
async def get_document(doc_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    status, data, text = await _post_json(edge_url, {"action": "get_document", "doc_id": doc_id}, headers)
    return {"ok": status == 200, "status": status, "json": data if data else {}, "text": _truncate(text)}

# ---------------------------
# Simplified search wrappers
# ---------------------------

@mcp.tool(description="Canonical search: return IDs + snippets for follow‑up fetch().")
async def search(query: str, limit: int = 10, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = await edge_search(SearchParams(q=query, mode=DEFAULTS.mode, top_k=limit, min_score=DEFAULTS.min_score, rerank=False))
    if not base.get("ok"):
        return {"ok": False, "error": base.get("error", "search_failed"), "results": []}
    items: List[Dict[str, Any]] = []
    for r in base.get("raw", {}).get("results", []):
        ex_id = r.get("exhibit_id") or r.get("id") or "Unknown"
        snippet = _truncate(r.get("text") or r.get("chunk") or r.get("content") or "", n=DEFAULTS.snippet_length)
        score = r.get("score") or r.get("similarity")
        items.append({"id": ex_id, "snippet": snippet, "score": score})
    return {"ok": True, "results": items}

@mcp.tool(description="High-level semantic/hybrid search with concise snippets.")
async def search_legal(query: str, top_k: int = 10, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    res = await hybrid_search(SearchParams(q=query, mode="hybrid", top_k=top_k, min_score=DEFAULTS.min_score, rerank=False))
    if not res.get("ok"):
        return res
    items: List[Dict[str, Any]] = []
    for r in res.get("raw", {}).get("results", []):
        ex_id = r.get("exhibit_id") or r.get("id") or "Unknown"
        snippet = _truncate(r.get("text") or r.get("chunk") or r.get("content") or "", n=DEFAULTS.snippet_length)
        score = r.get("score") or r.get("similarity")
        items.append({"exhibit_id": ex_id, "snippet": snippet, "score": score})
    return {"ok": True, "items": items}

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

@mcp.tool(description="List claims (optional status filter).")
async def list_claims(status: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "claims": []}
    try:
        q = _SUPABASE.table("claims").select("*")
        if status:
            q = q.ilike("status", status)
        rows = q.execute().data or []
        return {"ok": True, "claims": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "claims": []}

@mcp.tool(description="Facts for a given claim.")
async def get_facts_by_claim(claim_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "facts": []}
    try:
        rows = _SUPABASE.table("facts").select("*").eq("claim_id", claim_id).execute().data or []
        return {"ok": True, "facts": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "facts": []}

@mcp.tool(description="Facts for a given exhibit.")
async def get_facts_by_exhibit(exhibit_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "facts": []}
    eid = _normalize_exhibit_id(exhibit_id)
    try:
        rows = _SUPABASE.table("facts").select("*").eq("exhibit_id", eid).execute().data or []
        return {"ok": True, "facts": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "facts": []}

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

@mcp.tool(description="Static example timeline (placeholder data).")
async def get_case_timeline(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    timeline = [
        {"date": "2020-11-06", "event": "Operating Agreement for Floorable LLC signed"},
        {"date": "2021-09-15", "event": "Employee termination that led to breach allegation"},
        {"date": "2022-01-05", "event": "Legal complaint filed by R. Burns"},
        {"date": "2023-03-10", "event": "Discovery phase begins, key evidence collected"},
        {"date": "2024-07-22", "event": "Trial scheduled in California Superior Court"},
    ]
    return {"ok": True, "timeline": timeline}

@mcp.tool(description="Entities list.")
async def get_entities(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "entities": []}
    try:
        rows = _SUPABASE.table("entities").select("*").execute().data or []
        return {"ok": True, "entities": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "entities": []}

@mcp.tool(description="Individuals list (optional role filter).")
async def get_individuals(role: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "individuals": []}
    try:
        q = _SUPABASE.table("individuals").select("*")
        if role:
            q = q.ilike("role", role)
        rows = q.execute().data or []
        return {"ok": True, "individuals": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "individuals": []}

# Canonical fetch
@mcp.tool(description="Given an exhibit ID, return full content and metadata.")
async def fetch(id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not id:
        return {"ok": False, "error": "missing_id"}
    res = await fetch_exhibit(id)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "fetch_failed"), "id": id}
    pages = res.get("pages", []) or []
    content = "\n\n".join(pages) if isinstance(pages, list) else str(pages)
    meta = {k: v for k, v in res.items() if k in ("exhibit_id", "description", "filename")}
    return {"ok": True, "id": res.get("exhibit_id", id), "content": content, "metadata": meta}

# ---------------------------
# Codex Agent mgmt (kept as-is)
# ---------------------------

@mcp.tool(description="Manage codex background agent (start/stop/status/restart/add_task).")
async def codex_agent_control(params: CodexAgentParams) -> Dict[str, Any]:
    action = (params.action or "").lower()
    try:
        if action == "start":
            r = subprocess.run([sys.executable, "launch_codex_agent.py", "--daemon"], capture_output=True, text=True, timeout=30)
            return {"ok": r.returncode == 0, "action": "start", "stdout": r.stdout, "stderr": r.stderr, "return_code": r.returncode}
        elif action == "stop":
            r = subprocess.run([sys.executable, "launch_codex_agent.py", "stop"], capture_output=True, text=True, timeout=30)
            return {"ok": r.returncode == 0, "action": "stop", "stdout": r.stdout, "stderr": r.stderr, "return_code": r.returncode}
        elif action == "status":
            r = subprocess.run([sys.executable, "launch_codex_agent.py", "status"], capture_output=True, text=True, timeout=10)
            return {"ok": r.returncode == 0, "action": "status", "stdout": r.stdout, "stderr": r.stderr, "return_code": r.returncode, "running": r.returncode == 0}
        elif action == "restart":
            r = subprocess.run([sys.executable, "launch_codex_agent.py", "restart"], capture_output=True, text=True, timeout=60)
            return {"ok": r.returncode == 0, "action": "restart", "stdout": r.stdout, "stderr": r.stderr, "return_code": r.returncode}
        elif action == "add_task":
            if not params.task_type or not params.task_data:
                return {"ok": False, "error": "task_type and task_data are required for add_task"}
            import uuid
            task_id = str(uuid.uuid4())
            payload = {"task_id": task_id, "task_type": params.task_type, "payload": params.task_data, "created_at": _now_ms()}
            task_file = os.path.join(tempfile.gettempdir(), f"codex_task_{task_id}.json")
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return {"ok": True, "action": "add_task", "task_id": task_id, "task_file": task_file, "message": "Task queued"}
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Operation timed out"}
    except Exception as e:
        return {"ok": False, "error": f"Agent control error: {e}"}

@mcp.tool(description="Codex agent info.")
async def codex_agent_info(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        r = subprocess.run([sys.executable, "launch_codex_agent.py", "status"], capture_output=True, text=True, timeout=10)
        is_running = r.returncode == 0
        cfg = {
            "agent_name": os.getenv("CODEX_AGENT_NAME", "burns-codex-agent"),
            "max_workers": os.getenv("CODEX_MAX_WORKERS", "4"),
            "check_interval": os.getenv("CODEX_CHECK_INTERVAL", "30.0"),
            "enable_health_checks": os.getenv("CODEX_ENABLE_HEALTH_CHECKS", "true"),
            "log_level": os.getenv("CODEX_LOG_LEVEL", "INFO"),
            "log_file": os.getenv("CODEX_LOG_FILE"),
            "pid_file": os.getenv("CODEX_PID_FILE", os.path.join(tempfile.gettempdir(), "codex_agent.pid")),
            "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
            "openai_configured": bool(OPENAI_API_KEY),
            "codex_model": os.getenv("CODEX_MODEL", "gpt-4"),
            "codex_endpoint": os.getenv("CODEX_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
            "max_task_duration": os.getenv("CODEX_MAX_TASK_DURATION", "300"),
            "task_retry_attempts": os.getenv("CODEX_TASK_RETRY_ATTEMPTS", "3"),
            "task_retry_delay": os.getenv("CODEX_TASK_RETRY_DELAY", "5.0"),
        }
        return {"ok": True, "running": is_running, "status_output": r.stdout or "", "config": cfg}
    except Exception as e:
        return {"ok": False, "error": f"Failed to get agent info: {e}"}

# ---------------------------
# HTTP layer — routes & auth
# ---------------------------

# Simple Bearer middleware that protects /mcp* endpoints with BDB_AUTH_TOKEN
class SimpleBearerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if BDB_AUTH_TOKEN and request.url.path.startswith(MCP_PATH):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != BDB_AUTH_TOKEN:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

# Route handlers (defined as Starlette callables so we can mount them into the Streamable app)
async def _root_ok(request: Request):
    return JSONResponse({"ok": True, "name": SERVER_NAME, "version": SERVER_VERSION, "ts": _now_ms()})

async def _health_get(request: Request):
    # Guarded by middleware
    return JSONResponse({"name": SERVER_NAME, "version": SERVER_VERSION, "ok": True})

async def _health_head(request: Request):
    # Guarded by middleware
    return PlainTextResponse("", status_code=200)

async def _diag_http(request: Request):
    # Guarded by middleware
    return JSONResponse({
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
        "edge_url": _resolve_edge_url(),
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "supabase_url_set": bool(SUPABASE_URL),
        "openai_key_set": bool(OPENAI_API_KEY),
        "stateless_http": True,
    })

async def _supabase_health(request: Request):
    # Guarded by middleware
    headers = {}
    key = SUPABASE_SERVICE_ROLE_KEY
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
        "rest_ok": (status_rest in (200, 204, 405)),
        "status_auth": status_auth,
        "status_rest": status_rest,
    })

# Build the Streamable HTTP app with our custom routes & middleware
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
    # NOTE: no SimpleBearerMiddleware; FastCloud handles OAuth at the edge
]

app = create_streamable_http_app(
    mcp,
    streamable_http_path=MCP_PATH,
    auth=None,              # ⬅️ let FastCloud own authentication
    json_response=True,
    stateless_http=True,
    routes=_routes,
    middleware=_middleware,
)

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
