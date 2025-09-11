#!/usr/bin/env python3
# server.py
import os
from typing import Optional, Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse, JSONResponse

# FastMCP: support both >=2.12.2 (has http_app) and 2.12.0 (no http_app)
try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # very early/defensive
    raise RuntimeError(f"FastMCP import failed: {e}")

# --- Optional Python client; gracefully degrade to REST if unavailable ---
try:
    from supabase import create_client
except Exception:
    create_client = None

# ----------------------------
# Environment & configuration
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")  # optional alias
    or SUPABASE_SERVICE_ROLE_KEY
    or SUPABASE_ANON_KEY
)

EDGE_FUNCTION_PATH = os.getenv("EDGE_FUNCTION_PATH", "")
EDGE_FUNCTION_NAME = os.getenv("EDGE_FUNCTION_NAME", "advanced_semantic_search")
EDGEFN_AUTH_MODE = os.getenv("EDGEFN_AUTH_MODE", "supabase").lower()  # "none"|"supabase"|"service_role"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWLIST_HOSTS = os.getenv("ALLOWLIST_HOSTS", "")
ENABLE_DIAG = os.getenv("ENABLE_DIAG", "0") == "1"
try:
    PORT = int(os.getenv("PORT", "8080"))
except ValueError:
    PORT = 8080

# HTTP client tuning
try:
    HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "30"))
except ValueError:
    HTTP_TIMEOUT_S = 30.0
try:
    HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "20"))
except ValueError:
    HTTP_MAX_CONNECTIONS = 20
try:
    HTTP_MAX_KEEPALIVE = int(os.getenv("HTTP_MAX_KEEPALIVE", "10"))
except ValueError:
    HTTP_MAX_KEEPALIVE = 10
USER_AGENT = os.getenv("USER_AGENT", "BurnsLegal-MCP/1.0")

# ----------------------------
# Utilities
# ----------------------------
def _split_csv(v: str) -> List[str]:
    return [s.strip() for s in v.split(",") if s.strip()]

def _origin_of(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else url

def _host_of(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url if "://" in url else f"https://{url}")
    return p.netloc

def is_allowlisted(url: str) -> bool:
    """Accept entries as full origins (https://host) or bare domains (host.tld). '*' or empty => allow all."""
    if not ALLOWLIST_HOSTS or ALLOWLIST_HOSTS.strip() == "":
        return True
    allowed = _split_csv(ALLOWLIST_HOSTS)
    if "*" in allowed or not allowed:
        return True
    url_origin = _origin_of(url)
    url_host = _host_of(url)
    for entry in allowed:
        entry_origin = _origin_of(entry)
        entry_host = _host_of(entry)
        if url_origin == entry_origin or url_host == entry_host:
            return True
    return False

def resolve_edge_function_url(function_name: Optional[str] = None) -> str:
    """
    Resolve a concrete Edge Function URL.
    Precedence: explicit EDGE_FUNCTION_PATH (full or base), then SUPABASE_URL/functions/v1/<name>, then <ref>.functions.supabase.co/<name>.
    If EDGE_FUNCTION_PATH looks like a base (ends with '/functions/v1' or contains no function segment), append '/<function_name>'.
    """
    name = (function_name or EDGE_FUNCTION_NAME or "advanced_semantic_search").strip("/")
    if EDGE_FUNCTION_PATH:
        base = EDGE_FUNCTION_PATH.rstrip("/")
        # heuristics: treat as base if endswith functions/v1 or host only
        if base.endswith("/functions/v1") or base.count("/") <= 2:
            return f"{base}/{name}"
        return base
    if SUPABASE_URL:
        return f"{SUPABASE_URL.rstrip('/')}/functions/v1/{name}"
    if SUPABASE_PROJECT_REF:
        return f"https://{SUPABASE_PROJECT_REF}.functions.supabase.co/{name}"
    return ""

def supabase_auth_headers(key: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "apikey": key}

def make_edge_headers(mode: str = EDGEFN_AUTH_MODE) -> Dict[str, str]:
    """
    - 'none'         -> no Authorization/apikey
    - 'supabase'     -> use anon (least privilege) or SUPABASE_KEY
    - 'service_role' -> use service role key
    """
    mode = (mode or "").lower()
    base: Dict[str, str] = {"Content-Type": "application/json"}
    key: Optional[str] = None
    if mode == "service_role":
        key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY
    elif mode in ("supabase", "apikey"):
        key = SUPABASE_ANON_KEY or SUPABASE_KEY
    else:
        key = None
    if key:
        base.update(supabase_auth_headers(key))
    return base

# Optional Python client
_supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        _supabase = None

# ----------------------------
# Build FastMCP server
# ----------------------------
mcp = FastMCP("burns-legal-streamable")

# MCP root (when mounted at /mcp this becomes /mcp/)
@mcp.custom_route("/", methods=["GET"])
async def _root_ok(_req):
    return PlainTextResponse("ok", status_code=200)

# OpenAI wrapper helper for ChatGPT hosted MCP
@mcp.custom_route("/openai/hosted-tool", methods=["GET"])
async def _openai_tool(_request):
    """
    Returns a JSON block that ChatGPT's hostedMcpTool() can use
    to connect to this server without additional headers. Auth is
    handled by the hosting layer (e.g., FastCloud).
    """
    base_url = os.getenv("FASTCLOUD_URL", "https://burnsdb.fastmcp.app/mcp").strip()
    tool_def = {
        "label": "BurnsDB",
        "url": base_url,
    }
    return JSONResponse(tool_def, status_code=200)

# Optional diagnostic route
if ENABLE_DIAG:
    @mcp.custom_route("/diag/supabase", methods=["GET"])
    async def _diag_supabase(_req):
        if not SUPABASE_URL or not (SUPABASE_ANON_KEY or SUPABASE_KEY):
            return JSONResponse({"ok": False, "error": "supabase_not_configured"}, status_code=200)
        headers = supabase_auth_headers(SUPABASE_ANON_KEY or SUPABASE_KEY)
        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/documents"
        params = {"select": "id", "limit": "1"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, params=params, headers=headers)
                ok = (r.status_code in (200, 206))
                return JSONResponse({"ok": ok, "status": r.status_code})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)})

async def _call_edge_function(query: str, mode: str, top_k: int, min_score: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call Supabase Edge Function if configured and allowlisted.

    Returns (data, error):
    - data: dict result if successful (or structured error dict on invalid JSON)
    - error: string code if the call could not be made or failed non-200
    """
    edge_url = resolve_edge_function_url()
    if not edge_url or not is_allowlisted(edge_url):
        return None, "edgefn_not_configured_or_not_allowlisted"

    headers = {"User-Agent": USER_AGENT}
    headers.update(make_edge_headers())
    try:
        client = _http_client or httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, limits=http_limits, headers={"User-Agent": USER_AGENT})
        close_after = client is not _http_client
        resp = await client.post(
            edge_url,
            json={"query": query, "mode": mode, "top_k": top_k, "min_score": min_score},
            headers=headers,
        )
        if close_after:
            await client.aclose()
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                # treat invalid JSON as a terminal edge error response
                return {"error": "invalid_edge_json", "text": resp.text[:500]}, None
            return data if isinstance(data, dict) else {"results": data}, None
        return None, f"edgefn_{resp.status_code}"
    except Exception as e:
        return None, f"edgefn_exc_{e}"

async def _keyword_search_fallback(query: str, top_k: int, edge_err: str) -> Dict[str, Any]:
    """Fallback to PostgREST keyword search against documents table."""
    if not SUPABASE_URL or not (SUPABASE_ANON_KEY or SUPABASE_KEY):
        return {"ok": False, "error": f"supabase_not_configured; edge_err={edge_err}", "items": []}
    if not is_allowlisted(SUPABASE_URL):
        return {"ok": False, "error": f"supabase_url_not_allowlisted; edge_err={edge_err}", "items": []}

    headers = supabase_auth_headers(SUPABASE_ANON_KEY or SUPABASE_KEY)
    params = {
        "select": "id,title,preview",
        "content": f"ilike.*{query}*",
        "limit": str(top_k),
    }
    try:
        client = _http_client or httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, limits=http_limits, headers={"User-Agent": USER_AGENT})
        close_after = client is not _http_client
        r = await client.get(f"{SUPABASE_URL.rstrip('/')}/rest/v1/documents", params=params, headers=headers)
        if close_after:
            await client.aclose()
        if r.status_code not in (200, 206):
            return {"ok": False, "error": f"rest_{r.status_code}; edge_err={edge_err}", "items": []}
        items = r.json() or []
        return {"ok": True, "items": items, "fallback": True, "edge_err": edge_err}
    except Exception as e:
        return {"ok": False, "error": f"rest_exc_{e}; edge_err={edge_err}", "items": []}

@mcp.tool(description="Hybrid search via Supabase EdgeFn with safe fallback to keyword PostgREST.")
async def search(query: str, mode: str = "hybrid", top_k: int = 10, min_score: float = 0.0) -> Dict[str, Any]:
    data, err = await _call_edge_function(query=query, mode=mode, top_k=top_k, min_score=min_score)
    if data is not None:
        return data
    edge_err = err or "edgefn_unknown"
    return await _keyword_search_fallback(query=query, top_k=top_k, edge_err=edge_err)

# ----------------------------
# Host FastAPI and mount MCP at /mcp
# ----------------------------
main_app = FastAPI(title="Burns Legal MCP", version="1.0.0")
allow_origins = ["*"] if ALLOWED_ORIGINS.strip() == "*" else _split_csv(ALLOWED_ORIGINS)
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

http_limits = httpx.Limits(max_connections=HTTP_MAX_CONNECTIONS, max_keepalive_connections=HTTP_MAX_KEEPALIVE)
_http_client: Optional[httpx.AsyncClient] = None

@main_app.on_event("startup")
async def _startup():
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, limits=http_limits, headers={"User-Agent": USER_AGENT})

@main_app.on_event("shutdown")
async def _shutdown():
    global _http_client
    try:
        if _http_client is not None:
            await _http_client.aclose()
    finally:
        _http_client = None

@main_app.get("/")
async def root():
    return {"ok": True}

@main_app.get("/health")
async def health():
    return {"status": "healthy", "app": "burns-legal-streamable"}

# --- FastMCP 2.12.x compatibility shim ---
def _compat_http_app(server: FastMCP, path: str = "/mcp"):
    """
    Compatibility shim for FastMCP 2.12.0 environments that lack `server.http_app`.

    When FastCloud (or other deploy targets) import an MCP server module, they may
    expect a callable ASGI app mounted at a path. FastMCP gained `http_app()` in
    >= 2.12.2; older 2.12.0 builds do not have this attribute. In those cases:

    - We expose a minimal FastAPI app so HTTP probing and import inspection succeed.
    - The actual transport wiring is performed by the FastMCP CLI when the platform
      executes `fastmcp run` with HTTP transport.
    - This avoids double event loop issues (e.g., "Already running asyncio in this thread")
      that can occur if the module tries to start its own runner during import time.

    If `server.http_app` exists, we simply delegate to it.
    """
    if hasattr(server, "http_app"):
        return server.http_app(path=path)  # type: ignore[attr-defined]
    # Minimal shim: mount a root "ok" while FastCloud uses the CLI to wire HTTP transport.
    # This prevents 'inspect' from failing and keeps /mcp/ok probe working pre-run.
    from fastapi import FastAPI
    shim = FastAPI()
    @shim.get("/")
    async def ok():
        return {"ok": True, "shim": "fastmcp-2.12.0"}
    return shim

try:
    mcp_app = mcp.http_app(path="/mcp")  # >=2.12.2
except Exception:
    mcp_app = _compat_http_app(mcp, path="/mcp")  # 2.12.0 shim

main_app.mount("/mcp", mcp_app)
app = main_app  # ASGI export for local uvicorn if used

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
