# server.py
# Burns Database MCP Server — Streamable HTTP + Supabase Edge / PostgREST
# Compatible with fastmcp 2.12.x (no server_name/server_version kwargs in FastMCP.__init__)

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import asyncio
import httpx

# FastMCP
from fastmcp import FastMCP
from fastmcp.server.http import create_streamable_http_app

# Starlette for extra routes
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Optional Supabase client (PostgREST)
try:
    from supabase import create_client, Client as SupabaseClient  # type: ignore
except Exception as e:
    create_client = None
    SupabaseClient = None
    print("[BDB] Supabase SDK import failed (Edge-only mode still works):", e, file=sys.stderr)

# ------------------------------------------------------------------------------
# Canonical BDB facts / env
# ------------------------------------------------------------------------------

SERVER_NAME = os.getenv("APP_NAME", "BDB").strip()
SERVER_VERSION = os.getenv("APP_VERSION", "6.1").strip()
GIT_SHA = os.getenv("GIT_SHA", "").strip()
MCP_PATH = "/mcp"  # mount path

PORT = int(os.getenv("PORT", "8080"))
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nqkzqcsqfvpquticvwmk.supabase.co").strip()
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
)
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()  # unused if service role present

EDGEFN_URL = os.getenv(
    "EDGEFN_URL",
    "https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search",
).strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

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

MAX_FETCH_SIZE = int(os.getenv("MAX_FETCH_SIZE", "1000000"))
SNIPPET_LENGTH = int(os.getenv("SNIPPET_LENGTH", "500"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "25"))

SEARCH_MODE = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
LEXICAL_K = int(os.getenv("LEXICAL_K", "100"))
ENABLE_DIAG = (os.getenv("ENABLE_DIAG", "1").strip().lower() in ("1", "true", "yes", "y"))

# ------------------------------------------------------------------------------
# Supabase client (optional; used for fetch + lexical fallback)
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
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

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
# Retrieval core (no decorators here)
# ------------------------------------------------------------------------------

def _resolve_edge_url() -> str:
    return EDGEFN_URL

async def _edge_search_core(
    q: str,
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    rerank: Optional[bool] = True,
    label: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}

    m = (mode or SEARCH_MODE).lower()
    k = int(top_k or SEARCH_TOP_K)
    body: Dict[str, Any] = {
        "query": q,
        "q": q,
        "mode": m,
        "searchType": m,
        "k": k,
        "topK": k,
        "limit": k,
        "min_score": float(min_score if min_score is not None else SEARCH_MIN_SCORE),
        "lexical_k": LEXICAL_K,
        "rerank": bool(rerank),
    }
    if label:
        body["label"] = label
    if metadata_filter:
        body["filter"] = metadata_filter

    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, body, headers)
    out: Dict[str, Any] = {"ok": (status == 200), "status": status, "url": edge_url}
    if not out["ok"]:
        out["error"] = data if data else _truncate(text)
        return out
    out["raw"] = data
    return out

async def _fetch_exhibit_pages(exhibit_id: str) -> Tuple[bool, List[str], str]:
    """
    Returns (ok, pages, error_message).
    """
    if not _SUPABASE:
        return (False, [], "supabase_client_not_initialized")
    eid = _normalize_exhibit_id(exhibit_id)
    try:
        q = (
            _SUPABASE.table("vector_embeddings")
            .select("text,exhibit_id,page,chunk_index")
            .eq("exhibit_id", eid)
        )
        # best-effort ordering
        try:
            q = q.order("page")
        except Exception:
            pass
        try:
            q = q.order("chunk_index")
        except Exception:
            pass
        resp = q.limit(5000).execute()
        pages: List[str] = []
        for rec in resp.data or []:
            t = rec.get("text") or ""
            if t:
                pages.append(t)
        return (True, pages[:MAX_FETCH_SIZE], "")
    except Exception as e:
        return (False, [], f"fetch_exhibit_failed: {e}")

# ------------------------------------------------------------------------------
# MCP server + tools
# ------------------------------------------------------------------------------

mcp = FastMCP()  # IMPORTANT: no kwargs for 2.12.0 compatibility

@mcp.tool(description="Echo input for debugging.")
async def echo(text: str) -> Dict[str, Any]:
    return {"ok": True, "echo": text}

@mcp.tool(description="List capabilities and server info.")
async def list_capabilities(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA, "mcp_path": MCP_PATH},
        "transport": "http_streamable",
        "stateless_http": True,
        "allowlist_hosts": ALLOWLIST_HOSTS,
    }

@mcp.tool(description="Adjust default search behavior.")
async def set_defaults(
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    lexical_k: Optional[int] = None,
) -> Dict[str, Any]:
    global SEARCH_MODE, SEARCH_TOP_K, SEARCH_MIN_SCORE, LEXICAL_K
    if mode:
        SEARCH_MODE = mode.lower().strip()
    if top_k is not None:
        SEARCH_TOP_K = int(top_k)
    if min_score is not None:
        SEARCH_MIN_SCORE = float(min_score)
    if lexical_k is not None:
        LEXICAL_K = int(lexical_k)
    return {"ok": True, "defaults": {
        "mode": SEARCH_MODE, "top_k": SEARCH_TOP_K, "min_score": SEARCH_MIN_SCORE, "lexical_k": LEXICAL_K
    }}

# ----- Search family (use private helpers; DO NOT call decorated tools) -----

@mcp.tool(description="Canonical search: returns `ids` and scored snippets for follow‑up `fetch()`.")
async def search(
    query: str,
    limit: int = 10,
    mode: Optional[str] = None,
    min_score: Optional[float] = None,
    rerank: Optional[bool] = True,
    label: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = await _edge_search_core(
        q=query,
        mode=mode,
        top_k=limit,
        min_score=min_score,
        rerank=rerank,
        label=label,
        metadata_filter=metadata_filter,
    )
    items: List[Dict[str, Any]] = []
    if base.get("ok"):
        for r in (base.get("raw", {}) or {}).get("results", []) or []:
            ex_id = r.get("exhibit_id") or r.get("id") or _normalize_exhibit_id(r.get("doc_id", "") or "")
            if not ex_id:
                continue
            snippet = _truncate(r.get("snippet") or r.get("text") or r.get("chunk") or r.get("content") or "")
            score = r.get("score") or r.get("similarity")
            items.append({"id": _normalize_exhibit_id(ex_id), "snippet": snippet, "score": score})

    # Lexical fallback if nothing returned and PostgREST is available
    if not items and _SUPABASE:
        try:
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

@mcp.tool(description="Hybrid search via Edge (explicit control of params).")
async def edge_search(
    q: str,
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    rerank: Optional[bool] = True,
    label: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await _edge_search_core(q, mode, top_k, min_score, rerank, label, metadata_filter)

@mcp.tool(description="Request query embedding from Edge (action=embed_query).")
async def embed_query(q: str) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, {"action": "embed_query", "query": q, "q": q}, headers)
    if status != 200:
        return {"ok": False, "status": status, "error": data if data else _truncate(text)}
    emb = data.get("embedding") or data.get("vector") or []
    return {"ok": True, "dim": len(emb), "embedding": emb}

# ----- Spec‑compliant fetch -----

@mcp.tool(description="Fetch a complete record by ID (Deep Research compatible).")
async def fetch(id: str) -> Dict[str, Any]:
    ok, pages, err = await _fetch_exhibit_pages(id)
    if not ok:
        return {"ok": False, "error": err, "id": _normalize_exhibit_id(id), "text": "", "metadata": {}}
    text = "\n\n".join(pages)
    meta: Dict[str, Any] = {"exhibit_id": _normalize_exhibit_id(id)}
    if _SUPABASE:
        try:
            resp = (
                _SUPABASE.table("exhibits")
                .select("description,filename")
                .eq("exhibit_id", _normalize_exhibit_id(id))
                .limit(1)
                .execute()
            )
            if resp.data:
                row = resp.data[0] or {}
                meta.update({k: row.get(k) for k in ("description", "filename") if row.get(k)})
        except Exception:
            pass
    title = (meta.get("description") or str(id)).strip()
    return {"ok": True, "id": _normalize_exhibit_id(id), "title": title, "text": text, "metadata": meta}

# ----- Additional helpers -----

@mcp.tool(description="Retrieve full text pages for an exhibit (server-side PostgREST).")
async def fetch_exhibit(exhibit_id: str) -> Dict[str, Any]:
    ok, pages, err = await _fetch_exhibit_pages(exhibit_id)
    return {"ok": ok, "exhibit_id": _normalize_exhibit_id(exhibit_id), "pages": pages, "error": (None if ok else err)}

@mcp.tool(description="List exhibits; optionally attach simple categories.")
async def list_exhibits(withLabels: bool = False) -> Dict[str, Any]:
    if not _SUPABASE:
        return {"ok": False, "error": "supabase_client_not_initialized", "exhibits": []}
    try:
        data = (
            _SUPABASE.table("exhibits")
            .select("exhibit_id,description,filename")
            .limit(5000)
            .execute()
        ).data or []
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
                resp = (
                    _SUPABASE.table("exhibits")
                    .select("description,filename")
                    .eq("exhibit_id", _normalize_exhibit_id(ex))
                    .limit(1)
                    .execute()
                )
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

# ----- Diagnostics (accept loose payloads to satisfy MCP Inspector) -----

@mcp.tool(description="Show config and defaults.")
async def diag_config(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA},
        "edge": {"url": _resolve_edge_url()},
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "supabase_url_set": bool(SUPABASE_URL),
        "openai_key_set": bool(OPENAI_API_KEY),
        "defaults": {"mode": SEARCH_MODE, "top_k": SEARCH_TOP_K, "min_score": SEARCH_MIN_SCORE, "lexical_k": LEXICAL_K},
        "stateless_http": True,
        "enable_diag": ENABLE_DIAG,
    }

@mcp.tool(description="POST a custom payload to Edge (diagnostic).")
async def diag_edge(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}
    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, payload or {"action": "ping", "ping": True}, headers)
    return {"ok": status == 200, "status": status, "url": edge_url, "json": data if data else {}, "text": _truncate(text)}

@mcp.tool(description="Basic health check (accepts optional message/payload).")
async def health(message: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "server": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA}

# ------------------------------------------------------------------------------
# Extra HTTP routes (/ and /health)
# ------------------------------------------------------------------------------

async def _root_ok(request: Request):
    return JSONResponse({
        "ok": True, "name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA,
        "ts": int(time.time()), "mcp_path": MCP_PATH
    })

async def _health_get(request: Request):
    edge_status, auth_status, rest_status = 0, 0, 0
    # Edge ping (safe)
    try:
        headers = make_edge_headers()
        status, _, _ = await _post_json(_resolve_edge_url(), {"action": "ping", "ping": True}, headers)
        edge_status = status
    except Exception:
        edge_status = 0

    # Supabase auth/rest probes (safe; tolerate failures)
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

_routes = [
    Route("/", endpoint=_root_ok, methods=["GET"]),
    Route("/health", endpoint=_health_get, methods=["GET"]),
    Route("/health", endpoint=_health_head, methods=["HEAD"]),
]
_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=[CORS_ALLOWED_ORIGINS] if CORS_ALLOWED_ORIGINS != "*" else ["*"],
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=["Content-Type", "Authorization", "Accept", "Origin"],
    ),
]

# Build streamable HTTP app mounted at /mcp (no shims; fail loudly on import errors)
app = create_streamable_http_app(
    mcp,
    streamable_http_path=MCP_PATH,
    routes=_routes,
    middleware=_middleware,
)

# Local dev convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
