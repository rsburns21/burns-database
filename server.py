# server.py
# Burns Database MCP Server — Streamable HTTP (stateless) + Supabase/Edge tools
# Auth is handled by FastCloud (OAuth in prod or bearer in staging via platform). No in-app bearer middleware.
# Version: 6.0.0 (adds spec-compliant `fetch`, search `ids`, root /health, Git SHA)

from __future__ import annotations

import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import asyncio
import httpx

from pydantic import BaseModel, Field

# FastMCP server (HTTP streamable)
from fastmcp import FastMCP
from fastmcp.server.http import create_streamable_http_app

# Starlette primitives for extra HTTP routes
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Optional Supabase client for PostgREST (used for fetch + lexical fallback)
try:
    from supabase import create_client, Client as SupabaseClient  # type: ignore
except Exception as e:
    create_client = None
    SupabaseClient = None
    print("[BDB] Supabase SDK import failed (ok if Edge-only):", e, file=sys.stderr)

# ------------------------------------------------------------------------------
# Identity & Environment (canonical facts)
# ------------------------------------------------------------------------------

SERVER_NAME = os.getenv("APP_NAME", "BDB").strip()
SERVER_VERSION = os.getenv("APP_VERSION", "6.0").strip()
GIT_SHA = os.getenv("GIT_SHA", "").strip()
MCP_PATH = "/mcp"  # mount path — no shims; FastCloud will import app and run it

# Ports/CORS per canonical facts
PORT = int(os.getenv("PORT", "8080"))
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Supabase & Edge
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nqkzqcsqfvpquticvwmk.supabase.co").strip()
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
)
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()  # not used if service role present

EDGEFN_URL = os.getenv(
    "EDGEFN_URL",
    "https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search",
).strip()

# OpenAI (rerank/diag only; not used for vector store)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Allowlist — enforce outbound hosts
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

# Search defaults (Edge function accepts both snake_case and camelCase)
SEARCH_MODE = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
# Non-zero lexical fallback ensures BM25 is attempted even if vector path is unavailable
LEXICAL_K = int(os.getenv("LEXICAL_K", "100"))

# Diagnostics
ENABLE_DIAG = (os.getenv("ENABLE_DIAG", "1").strip().lower() in ("1", "true", "yes", "y"))

# ------------------------------------------------------------------------------
# Supabase client (optional; used by fetch & lexical fallback)
# ------------------------------------------------------------------------------

_SUPABASE: Optional[SupabaseClient] = None
if create_client and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        _SUPABASE = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("[BDB] Supabase client initialized.", file=sys.stderr)
    except Exception as e:
        print("[BDB] Supabase client init failed:", e, file=sys.stderr)
else:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[BDB] Supabase client not initialized (missing SUPABASE_URL or SERVICE_ROLE_KEY).", file=sys.stderr)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _truncate(s: str, n: int = SNIPPET_LENGTH) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"

def _is_host_allowlisted(url: str) -> Tuple[bool, str]:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        return (host in ALLOWLIST_HOSTS, host)
    except Exception:
        return (False, "")

def make_edge_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    if key:
        # Always include BOTH headers per BurnsDB policy
        headers["apikey"] = key
        headers["Authorization"] = f"Bearer {key}"
    return headers

async def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, Any], str]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        resp = await client.post(url, json=body, headers=headers)
        text = resp.text
        try:
            data = await resp.json()
        except Exception:
            # fall back to text if invalid JSON
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
    if any(k in t for k in ("contract", "agreement", "msa", "sow")):
        return "KEY_CONTRACT"
    if any(k in t for k in ("email", "correspondence", "thread", "inbox")):
        return "CORRESPONDENCE"
    if any(k in t for k in ("invoice", "statement", "bank", "wire", "payment")):
        return "FINANCIAL"
    if any(k in t for k in ("technical", "spec", "design", "architecture", "diagram")):
        return "TECHNICAL"
    if any(k in t for k in ("image", "photo", "screenshot", "media", "audio", "video")):
        return "MEDIA"
    return "OTHER"

# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

class SearchParams(BaseModel):
    q: str = Field(..., description="User query")
    mode: str = Field(SEARCH_MODE, description="vector|bm25|hybrid|semantic")
    top_k: int = Field(SEARCH_TOP_K, ge=1, le=200)
    min_score: float = Field(SEARCH_MIN_SCORE, ge=0.0)
    rerank: bool = Field(True, description="Enable rerank if Edge supports it")
    label: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None  # e.g., {"category": "KEY_CONTRACT"}

class SetDefaultsParams(BaseModel):
    mode: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=200)
    min_score: Optional[float] = Field(None, ge=0.0)
    lexical_k: Optional[int] = None

# ------------------------------------------------------------------------------
# MCP server and tools
# ------------------------------------------------------------------------------

mcp = FastMCP(server_name=SERVER_NAME, server_version=SERVER_VERSION)

@mcp.tool(description="Echo input for debugging.")
async def echo(text: str) -> Dict[str, Any]:
    return {"ok": True, "echo": text}

@mcp.tool(description="List basic capabilities.")
async def list_capabilities() -> Dict[str, Any]:
    return {
        "ok": True,
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA, "mcp_path": MCP_PATH},
        "transport": "http_streamable",
        "stateless_http": True,
        "allowlist_hosts": ALLOWLIST_HOSTS,
    }

@mcp.tool(description="Version info.")
async def version_info() -> Dict[str, Any]:
    return {"ok": True, "version": SERVER_VERSION, "git_sha": GIT_SHA}

@mcp.tool(description="Adjust default search behavior.")
async def set_defaults(params: SetDefaultsParams) -> Dict[str, Any]:
    global SEARCH_MODE, SEARCH_TOP_K, SEARCH_MIN_SCORE, LEXICAL_K
    if params.mode:
        SEARCH_MODE = params.mode.lower().strip()
    if params.top_k is not None:
        SEARCH_TOP_K = params.top_k
    if params.min_score is not None:
        SEARCH_MIN_SCORE = params.min_score
    if params.lexical_k is not None:
        LEXICAL_K = params.lexical_k
    return {"ok": True, "defaults": {
        "mode": SEARCH_MODE, "top_k": SEARCH_TOP_K, "min_score": SEARCH_MIN_SCORE, "lexical_k": LEXICAL_K
    }}

def _resolve_edge_url() -> str:
    return EDGEFN_URL

async def _edge_search_core(params: SearchParams) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}

    top_k = params.top_k or SEARCH_TOP_K
    body: Dict[str, Any] = {
        "query": params.q,
        "q": params.q,
        "mode": (params.mode or SEARCH_MODE).lower(),
        "searchType": (params.mode or SEARCH_MODE).lower(),
        "k": top_k,
        "topK": top_k,
        "limit": top_k,
        "min_score": params.min_score if params.min_score is not None else SEARCH_MIN_SCORE,
        "lexical_k": LEXICAL_K,
        "rerank": bool(params.rerank),
    }
    if params.label:
        body["label"] = params.label
    if params.metadata_filter:
        body["filter"] = params.metadata_filter

    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, body, headers)
    out: Dict[str, Any] = {"ok": (status == 200), "status": status, "url": edge_url}
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

@mcp.tool(description="Hybrid (sparse+dense) search via Edge.")
async def hybrid_search(params: SearchParams) -> Dict[str, Any]:
    params.mode = "hybrid"
    return await _edge_search_core(params)

@mcp.tool(description="Semantic (dense only) search via Edge.")
async def semantic_search(params: SearchParams) -> Dict[str, Any]:
    params.mode = "semantic"
    return await _edge_search_core(params)

@mcp.tool(description="Request embeddings for a query from Edge (action=embed_query).")
async def embed_query(q: str) -> Dict[str, Any]:
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
# Canonical `search` with legal-discovery defaults
# ---------------------------

@mcp.tool(description="Canonical search: returns `ids` + scored snippets for follow‑up `fetch()`.")
async def search(query: str, limit: int = 10, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sp = SearchParams(q=query, mode=SEARCH_MODE, top_k=limit, min_score=SEARCH_MIN_SCORE, rerank=True)
    base = await edge_search(sp)
    items: List[Dict[str, Any]] = []

    # Prefer Edge results if ok
    if base.get("ok"):
        for r in (base.get("raw", {}) or {}).get("results", []) or []:
            ex_id = r.get("exhibit_id") or r.get("id") or _normalize_exhibit_id(r.get("doc_id", "") or "")
            if not ex_id:
                continue
            snippet = _truncate(r.get("snippet") or r.get("text") or r.get("chunk") or r.get("content") or "")
            score = r.get("score") or r.get("similarity")
            items.append({"id": ex_id, "snippet": snippet, "score": score})

    # Fallback lexical if Edge returned nothing and Supabase client exists
    if not items and _SUPABASE:
        try:
            # Basic BM25-ish fallback using text_search on vector_embeddings (schema may vary)
            resp = (
                _SUPABASE.table("vector_embeddings")
                .select("exhibit_id,text")
                .text_search("text", query)
                .limit(limit)
                .execute()
            )
            for rec in resp.data or []:
                ex_id = _normalize_exhibit_id(rec.get("exhibit_id") or "")
                snippet = _truncate(rec.get("text") or "")
                items.append({"id": ex_id, "snippet": snippet, "score": None})
        except Exception as e:
            return {"ok": False, "error": f"lexical_fallback_failed: {e}"}

    return {"ok": True, "results": items, "ids": [it["id"] for it in items]}

# ---------------------------
# Spec-compliant `fetch` (wraps fetch_exhibit)
# ---------------------------

@mcp.tool(description="Fetch a complete record by ID for ChatGPT/Deep Research.")
async def fetch(id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a single record with id/title/text/metadata."""
    rec = await fetch_exhibit(id)
    if not rec.get("ok"):
        return rec
    pages = rec.get("pages") or []
    text = "\n\n".join(pages)

    # optional metadata
    meta: Dict[str, Any] = {"exhibit_id": _normalize_exhibit_id(id)}
    if _SUPABASE:
        try:
            q = _SUPABASE.table("exhibits").select("description,filename").eq("exhibit_id", _normalize_exhibit_id(id)).limit(1)
            resp = q.execute()
            if resp.data:
                row = resp.data[0] or {}
                meta.update({k: row.get(k) for k in ("description", "filename") if row.get(k)})
        except Exception:
            pass

    title = (meta.get("description") or str(id)).strip()
    return {"ok": True, "id": _normalize_exhibit_id(id), "title": title, "text": text, "metadata": meta}

# ---------------------------
# Full-text pages by exhibit (Supabase REST)
# ---------------------------

@mcp.tool(description="Retrieve full text pages for an exhibit (server-side PostgREST).")
async def fetch_exhibit(exhibit_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "pages": []}
    eid = _normalize_exhibit_id(exhibit_id)
    try:
        # Try most specific schema first: vector_embeddings has one row per chunk/page
        # We attempt to sort by a `page` or `chunk_index` field if present.
        q = _SUPABASE.table("vector_embeddings").select("text,exhibit_id,page,chunk_index").eq("exhibit_id", eid)
        # some deployments use page, others chunk_index; order won't fail if column missing
        try:
            q = q.order("page")  # will no-op if missing
        except Exception:
            pass
        try:
            q = q.order("chunk_index")
        except Exception:
            pass
        resp = q.limit(5000).execute()  # generous ceiling for legal exhibits
        pages: List[str] = []
        for rec in resp.data or []:
            t = rec.get("text") or ""
            if t:
                pages.append(t)
        return {"ok": True, "exhibit_id": eid, "pages": pages[: MAX_FETCH_SIZE]}
    except Exception as e:
        return {"ok": False, "error": f"fetch_exhibit_failed: {e}", "pages": []}

@mcp.tool(description="List exhibits, optionally with simple categories.")
async def list_exhibits(withLabels: bool = False) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "exhibits": []}
    try:
        data = (_SUPABASE.table("exhibits").select("exhibit_id,description,filename").limit(5000).execute()).data or []
        if withLabels:
            for rec in data:
                rec["category"] = _categorize_exhibit(rec.get("description", ""), rec.get("filename", ""))
        return {"ok": True, "exhibits": data}
    except Exception as e:
        return {"ok": False, "error": f"list_exhibits_failed: {e}", "exhibits": []}

@mcp.tool(description="Derive simple legal-discovery labels for results.")
async def label_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled: List[Dict[str, Any]] = []
    for r in results or []:
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

# ---------------------------
# Diagnostics
# ---------------------------

@mcp.tool(description="Show config and defaults (raw).")
async def diag_config() -> Dict[str, Any]:
    return {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA},
        "edge": {"url": _resolve_edge_url()},
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "supabase_url_set": bool(SUPABASE_URL),
        "openai_key_set": bool(OPENAI_API_KEY),
        "defaults": {"mode": SEARCH_MODE, "top_k": SEARCH_TOP_K, "min_score": SEARCH_MIN_SCORE, "lexical_k": LEXICAL_K},
        "stateless_http": True,
        "enable_diag": ENABLE_DIAG,
    }

@mcp.tool(description="Detailed Edge diagnostic POST with optional payload.")
async def diag_edge(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}
    status, data, text = await _post_json(edge_url, payload or {"action": "ping", "ping": True}, headers)
    return {"ok": status == 200, "status": status, "url": edge_url, "json": data if data else {}, "text": _truncate(text)}

@mcp.tool(description="Basic health check.")
async def health() -> Dict[str, Any]:
    return {"ok": True, "server": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA}

# ------------------------------------------------------------------------------
# Extra HTTP routes (/health and diagnostics)
# ------------------------------------------------------------------------------

async def _root_ok(request: Request):
    return JSONResponse({
        "ok": True, "name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA,
        "ts": int(time.time()), "mcp_path": MCP_PATH
    })

async def _health_get(request: Request):
    # include a quick Edge ping and Supabase REST HEAD to catch misconfig early
    edge_status, auth_status, rest_status = 0, 0, 0

    # Edge ping
    try:
        headers = make_edge_headers()
        status, _, _ = await _post_json(_resolve_edge_url(), {"action": "ping", "ping": True}, headers)
        edge_status = status
    except Exception:
        edge_status = 0

    # Supabase REST & Auth endpoints (HEAD/GET)
    try:
        headers = make_edge_headers()
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            if SUPABASE_URL:
                au = f"{SUPABASE_URL.rstrip('/')}/auth/v1/health"
                ru = f"{SUPABASE_URL.rstrip('/')}/rest/v1/"
                try:
                    r1 = await client.get(au, headers=headers)
                    auth_status = r1.status_code
                except Exception:
                    auth_status = 0
                try:
                    r2 = await client.head(ru, headers=headers)
                    rest_status = r2.status_code
                except Exception:
                    rest_status = 0
    except Exception:
        pass

    return JSONResponse({
        "ok": True,
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA},
        "edge": {"url": _resolve_edge_url(), "status": edge_status, "allowlisted": _is_host_allowlisted(_resolve_edge_url())[0]},
        "supabase": {
            "url_set": bool(SUPABASE_URL),
            "auth_status": auth_status,
            "rest_status": rest_status,
        },
        "allowlist_hosts": ALLOWLIST_HOSTS,
    })

async def _health_head(request: Request):
    return PlainTextResponse("", status_code=200)

# Routes & middleware
_routes = [
    Route("/", endpoint=_root_ok, methods=["GET"]),
    Route("/health", endpoint=_health_get, methods=["GET"]),
    Route("/health", endpoint=_health_head, methods=["HEAD"]),
]
_middleware = [
    Middleware(CORSMiddleware,
               allow_origins=[CORS_ALLOWED_ORIGINS] if CORS_ALLOWED_ORIGINS != "*" else ["*"],
               allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
               allow_headers=["Content-Type", "Authorization", "Accept", "Origin"]),
]

# Build the Streamable HTTP app (no shim; fail loudly if this fails)
app = create_streamable_http_app(
    mcp,
    streamable_http_path=MCP_PATH,
    auth=None,              # FastCloud owns OAuth
    json_response=True,
    stateless_http=True,
    routes=_routes,
    middleware=_middleware,
)

# Local dev only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
