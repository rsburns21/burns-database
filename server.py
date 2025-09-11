"""
BDB (BurnsDB) — FastCloud MCP Server
------------------------------------

Deployment-safe server module for FastCloud:

- Exports `app` at module import time (NO uvicorn.run(), NO event-loop starts).
- Uses FastMCP SDK HTTP app directly; FastCloud will host it at port 8080.
- Auth is handled by FastCloud (no bearer checks here).
- Supabase access uses SERVICE ROLE KEY (from FastCloud secrets) for Edge Function calls & DB.
- Edge Function endpoint is STATIC and singular (advanced_semantic_search).

Environment (minimal; set via FastCloud):
  SUPABASE_URL=https://nqkzqcsqfvpquticvwmk.supabase.co
  SUPABASE_SERVICE_ROLE_KEY=<set in FastCloud Secrets>
Optional:
  ALLOWLIST_HOSTS=nqkzqcsqfvpquticvwmk.supabase.co,nqkzqcsqfvpquticvwmk.functions.supabase.co,api.openai.com,chatgpt.com,raw.githubusercontent.com,httpbin.org,playground.ai.cloudflare.com,example.com
  MAX_FETCH_SIZE=1000000
  SNIPPET_LENGTH=500
  ENABLE_DIAG=1
  FASTCLOUD_MCP_URL=https://burnsdb.fastmcp.app/mcp   # used by /openai/hosted-tool helper

NOTE: This file purposefully avoids any FastAPI/Starlette `on_event` startup/shutdown handlers
to prevent the "Already running asyncio in this thread" crash under FastCloud.
"""

import os
import sys
from typing import Dict, Optional, List, Any

import httpx
from fastmcp import FastMCP
from starlette.responses import PlainTextResponse, JSONResponse

# ---------- Supabase (optional direct table access) ----------
try:
    from supabase import create_client, Client as SupabaseClient  # type: ignore
except Exception as e:
    print("Supabase SDK not installed or import failed:", e, file=sys.stderr)
    create_client = None
    SupabaseClient = None  # type: ignore

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nqkzqcsqfvpquticvwmk.supabase.co").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# This server intentionally uses ONLY the service role key for Supabase access.
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY

_supabase: Optional["SupabaseClient"] = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[BDB] Supabase client initialized.")
    except Exception as e:
        print("[BDB] Supabase client init failed:", e, file=sys.stderr)
else:
    print("[BDB] Supabase client not initialized (missing SDK or env). Some tools will be inactive.", file=sys.stderr)

# ---------- Static Edge Function config ----------
# Single, fixed Edge Function used by all semantic/hybrid search tools.
EDGE_FUNCTION_URL = "https://nqkzqcsqfvpquticvwmk.functions.supabase.co/advanced_semantic_search"

def _edge_headers() -> Dict[str, str]:
    """
    Always include apikey + Authorization: Bearer <service_role_key> when key is available.
    Works with opaque service keys (non-JWT).
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    key = SUPABASE_SERVICE_ROLE_KEY
    if key:
        headers["apikey"] = key
        headers["Authorization"] = f"Bearer {key}"
    return headers

# ---------- Allowlist and fetch limits ----------
_allowlist_env = os.getenv("ALLOWLIST_HOSTS", "")
_ALLOWED: Optional[set] = None if _allowlist_env in ("", "*") else {h.strip().lower() for h in _allowlist_env.split(",") if h.strip()}
_MAX_FETCH_SIZE = int(os.getenv("MAX_FETCH_SIZE", "1000000"))
_SNIPPET_LENGTH = int(os.getenv("SNIPPET_LENGTH", "500"))
_ENABLE_DIAG = os.getenv("ENABLE_DIAG", "0") == "1"

# ---------- Utilities ----------
def _host_allowed(url: str) -> bool:
    try:
        host = httpx.URL(url).host or ""
    except Exception:
        return False
    if _ALLOWED is None:
        return True
    return host.lower() in _ALLOWED

def _snippet(txt: str, n: int = _SNIPPET_LENGTH) -> str:
    if len(txt) <= n:
        return txt
    return txt[:n] + "..."

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

# ---------- FastMCP Server ----------
mcp = FastMCP("BDB")

# Root and health (for probes)
@mcp.custom_route("/", methods=["GET"])
async def _root_ok(_request):
    return PlainTextResponse("ok", status_code=200)

@mcp.custom_route("/health", methods=["GET"])
async def _health(_request):
    info = {
        "server": {"name": "BDB", "version": "1.0.0"},
        "edge": {"url": EDGE_FUNCTION_URL, "configured": bool(EDGE_FUNCTION_URL), "auth": bool(SUPABASE_SERVICE_ROLE_KEY)},
        "supabase_client": bool(_supabase is not None)
    }
    return JSONResponse(info, status_code=200)

# Helper for OpenAI Agents SDK / GPT Builder wiring
@mcp.custom_route("/openai/hosted-tool", methods=["GET"])
async def _openai_hosted_tool(_request):
    """
    Convenience endpoint: returns a minimal object you can pass to OpenAI
    Agents SDK `hostedMcpTool({ label, url })`, or configure in a GPT tool.
    """
    url = os.getenv("FASTCLOUD_MCP_URL", "https://burnsdb.fastmcp.app/mcp")
    return JSONResponse({"label": "BurnsDB", "url": url}, status_code=200)

# ---------- Optional diagnostics ----------
if _ENABLE_DIAG:
    @mcp.custom_route("/diag/edge", methods=["GET"])
    async def _diag_edge(_request):
        if not EDGE_FUNCTION_URL:
            return JSONResponse({"ok": False, "error": "no_edge_url"}, status_code=200)
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(EDGE_FUNCTION_URL, json={"ping": True}, headers=_edge_headers())
            try:
                body = r.json()
            except Exception:
                body = {"_text": r.text[:500]}
            return JSONResponse({"ok": r.status_code == 200, "status": r.status_code, "body": body}, status_code=200)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

    @mcp.custom_route("/diag/supabase", methods=["GET"])
    async def _diag_supabase(_request):
        if not _supabase:
            return JSONResponse({"ok": False, "error": "no_client"}, status_code=200)
        try:
            # lightweight existence check
            resp = _supabase.table("vector_embeddings").select("exhibit_id").limit(1).execute()
            count = 0
            if hasattr(resp, "data") and isinstance(resp.data, list):
                count = len(resp.data)
            return JSONResponse({"ok": True, "vector_embeddings_sample_rows": count}, status_code=200)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

# ---------- Core Tools (search & retrieval) ----------

@mcp.tool()
async def edge_search(query: str, top_k: int = 10, mode: str = "hybrid") -> Dict[str, Any]:
    """
    Call the Supabase Edge Function for semantic / hybrid / BM25 search.
    mode: "hybrid" | "vector" | "bm25"
    """
    if not EDGE_FUNCTION_URL:
        return {"ok": False, "error": "edge_function_not_configured", "results": []}
    payload = {"query": query, "matchCount": max(1, min(int(top_k), 100)), "mode": (mode or "hybrid").lower()}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(EDGE_FUNCTION_URL, json=payload, headers=_edge_headers())
    except Exception as e:
        return {"ok": False, "error": f"request_error: {e}", "results": []}
    if resp.status_code != 200:
        return {"ok": False, "error": f"edge_function_error_{resp.status_code}", "text": resp.text[:500], "results": []}
    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "error": "invalid_json_from_edge", "text": resp.text[:500], "results": []}
    if not isinstance(data, list):
        return {"ok": False, "error": "invalid_response_format", "results": []}
    return {"ok": True, "results": data}

# Canonical search tool expected by some ChatGPT connectors
@mcp.tool()
async def search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Canonical search: returns IDs + snippets so `fetch()` can retrieve full content.
    """
    base = await edge_search(query=query, top_k=limit, mode="hybrid")
    if not base.get("ok"):
        return {"ok": False, "error": base.get("error", "search_failed"), "results": []}
    items: List[Dict[str, Any]] = []
    for r in base.get("results", []):
        ex = r.get("exhibit_id") or r.get("id") or "Unknown"
        text = r.get("text") or r.get("chunk") or r.get("content") or ""
        score = r.get("score") or r.get("similarity")
        items.append({"id": ex, "snippet": _snippet(str(text)), "score": score})
    return {"ok": True, "results": items}

@mcp.tool()
async def search_legal(query: str, top_k: int = 10) -> Dict[str, Any]:
    """High-level semantic/hybrid search with concise snippets."""
    res = await edge_search(query=query, top_k=top_k, mode="hybrid")
    if not res.get("ok"):
        return res
    items: List[Dict[str, Any]] = []
    for r in res.get("results", []):
        ex = r.get("exhibit_id") or r.get("id") or "Unknown"
        text = r.get("text") or r.get("chunk") or r.get("content") or ""
        score = r.get("score") or r.get("similarity")
        items.append({"exhibit_id": ex, "snippet": _snippet(str(text)), "score": score})
    return {"ok": True, "items": items}

@mcp.tool()
async def bm25_search(query: str, top_k: int = 10) -> Dict[str, Any]:
    """BM25-only search via Edge Function."""
    return await edge_search(query=query, top_k=top_k, mode="bm25")

@mcp.tool()
async def vector_search(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Vector-only similarity search via Edge Function."""
    return await edge_search(query=query, top_k=top_k, mode="vector")

@mcp.tool()
async def rerank(results: List[Dict[str, Any]], method: str = "relevance") -> Dict[str, Any]:
    """
    Simple reranking of already-fetched results.
    method: "relevance" | "recency" | "rrf"
    """
    if not isinstance(results, list):
        return {"ok": False, "error": "invalid_results_format", "results": []}
    m = (method or "relevance").lower()
    out = results
    if m == "relevance":
        try:
            out = sorted(results, key=lambda x: x.get("score", 0.0) or 0.0, reverse=True)
        except Exception:
            out = results
    elif m == "recency":
        out = results
    elif m in ("rrf", "reciprocal rank fusion"):
        out = results
    else:
        return {"ok": False, "error": f"unknown_method:{method}", "results": results}
    return {"ok": True, "results": out}

@mcp.tool()
async def label_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assign lightweight labels to results (e.g., CONTRACT / EVIDENCE / DAMAGES)."""
    labeled = []
    for r in results:
        ex = r.get("exhibit_id") or r.get("id") or ""
        desc, fname = "", ""
        if _supabase and ex:
            try:
                q = _supabase.table("exhibits").select("description, filename").eq("Exhibit_ID", _normalize_exhibit_id(ex))
                resp = q.limit(1).execute()
                if resp.data:
                    desc = (resp.data[0] or {}).get("description") or ""
                    fname = (resp.data[0] or {}).get("filename") or ""
            except Exception:
                pass
        label = _categorize_exhibit(desc or r.get("snippet", ""), fname)
        lr = dict(r)
        lr["category"] = label
        labeled.append(lr)
    return {"ok": True, "results": labeled}

# ---------- Retrieval Tools ----------

@mcp.tool()
def fetch_exhibit(exhibit_id: str) -> Dict[str, Any]:
    """Retrieve the full text pages for a given exhibit."""
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "pages": []}
    eid = _normalize_exhibit_id(exhibit_id)
    try:
        chunks = (_supabase
                  .table("vector_embeddings")
                  .select("text,page_start,page_end,chunk_index")
                  .eq("exhibit_id", eid)
                  .order("chunk_index", asc=True)
                  .execute()).data or []
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
        meta_resp = _supabase.table("exhibits").select("description,filename").eq("Exhibit_ID", eid).limit(1).execute()
        if meta_resp.data:
            meta.update({k: v for k, v in meta_resp.data[0].items() if k in ("description", "filename")})
    except Exception:
        pass
    return {"ok": True, **meta, "pages": pages}

@mcp.tool()
def list_exhibits(withLabels: bool = False) -> Dict[str, Any]:
    """List all exhibits with optional category labeling."""
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "exhibits": []}
    try:
        data = (_supabase.table("exhibits")
                .select("Exhibit_ID,description,filename")
                .execute()).data or []
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "exhibits": []}
    out = []
    for rec in data:
        ex = rec.get("Exhibit_ID")
        item = {"exhibit_id": ex, "description": rec.get("description", "")}
        if withLabels:
            item["category"] = _categorize_exhibit(rec.get("description", ""), rec.get("filename", ""))
        out.append(item)
    return {"ok": True, "exhibits": out}

# ---------- Keyword (BM25) direct via PostgREST (optional) ----------
@mcp.tool()
def keyword_search(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Keyword/BM25-style search using PostgREST full text on text column.
    Requires a text search index on vector_embeddings.text in your DB.
    """
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "items": []}
    try:
        resp = (_supabase.table("vector_embeddings")
                .select("exhibit_id,text,similarity")
                .text_search("text", query, {"config": "english", "type": "websearch"})
                .limit(max(1, min(int(top_k), 100)))
                .execute())
        rows = resp.data or []
    except Exception as e:
        return {"ok": False, "error": f"search_error:{e}", "items": []}
    items = [{"exhibit_id": r.get("exhibit_id", "Unknown"), "snippet": _snippet(r.get("text", ""))} for r in rows]
    return {"ok": True, "items": items}

# ---------- HTTP fetch utilities (non-canonical) ----------

@mcp.tool()
def http_fetch(url: str) -> Dict[str, Any]:
    """HTTP GET with allowlist + size cap (renamed to avoid clashing with canonical `fetch`)."""
    if not _host_allowed(url):
        return {"ok": False, "error": "host_not_allowed"}
    try:
        r = httpx.get(url, follow_redirects=True, timeout=10.0)
    except Exception as e:
        return {"ok": False, "error": f"request_error:{e}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"http_{r.status_code}"}
    b = r.content[:_MAX_FETCH_SIZE]
    text = b.decode("utf-8", errors="ignore")
    return {"ok": True, "content_snippet": _snippet(text), "content_length": len(text)}

@mcp.tool()
def fetch_allowed(url: str) -> Dict[str, Any]:
    """Alias of http_fetch() with allowlist enforced."""
    return http_fetch(url)

# ---------- Case data helpers ----------

@mcp.tool()
def list_claims(status: Optional[str] = None) -> Dict[str, Any]:
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "claims": []}
    try:
        q = _supabase.table("claims").select("*")
        if status:
            q = q.ilike("status", status)
        rows = q.execute().data or []
        return {"ok": True, "claims": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "claims": []}

@mcp.tool()
def get_facts_by_claim(claim_id: str) -> Dict[str, Any]:
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "facts": []}
    try:
        rows = (_supabase.table("facts").select("*").eq("claim_id", claim_id).execute()).data or []
        return {"ok": True, "facts": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "facts": []}

@mcp.tool()
def get_facts_by_exhibit(exhibit_id: str) -> Dict[str, Any]:
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "facts": []}
    eid = _normalize_exhibit_id(exhibit_id)
    try:
        rows = (_supabase.table("facts").select("*").eq("exhibit_id", eid).execute()).data or []
        return {"ok": True, "facts": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "facts": []}

@mcp.tool()
def get_case_statistics() -> Dict[str, Any]:
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized"}
    stats: Dict[str, Any] = {}
    try:
        c = _supabase.table("claims").select("id", count="exact").execute()
        stats["num_claims"] = c.count if hasattr(c, "count") and c.count is not None else len(c.data or [])
    except Exception as e:
        stats["num_claims"] = f"error:{e}"
    try:
        e = _supabase.table("exhibits").select("id", count="exact").execute()
        stats["num_exhibits"] = e.count if hasattr(e, "count") and e.count is not None else len(e.data or [])
    except Exception as e2:
        stats["num_exhibits"] = f"error:{e2}"
    try:
        f = _supabase.table("facts").select("id", count="exact").execute()
        stats["num_facts"] = f.count if hasattr(f, "count") and f.count is not None else len(f.data or [])
    except Exception as e3:
        stats["num_facts"] = f"error:{e3}"
    return {"ok": True, "statistics": stats}

@mcp.tool()
def get_case_timeline() -> Dict[str, Any]:
    # Static example; wire to DB if timeline table exists.
    timeline = [
        {"date": "2020-11-06", "event": "Operating Agreement for Floorable LLC signed"},
        {"date": "2021-09-15", "event": "Employee termination that led to breach allegation"},
        {"date": "2022-01-05", "event": "Legal complaint filed by R. Burns"},
        {"date": "2023-03-10", "event": "Discovery phase begins, key evidence collected"},
        {"date": "2024-07-22", "event": "Trial scheduled in California Superior Court"},
    ]
    return {"ok": True, "timeline": timeline}

@mcp.tool()
def get_entities() -> Dict[str, Any]:
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "entities": []}
    try:
        rows = (_supabase.table("entities").select("*").execute()).data or []
        return {"ok": True, "entities": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "entities": []}

@mcp.tool()
def get_individuals(role: Optional[str] = None) -> Dict[str, Any]:
    if not _supabase:
        return {"ok": False, "error": "supabase_client_not_initialized", "individuals": []}
    try:
        q = _supabase.table("individuals").select("*")
        if role:
            q = q.ilike("role", role)
        rows = q.execute().data or []
        return {"ok": True, "individuals": rows}
    except Exception as e:
        return {"ok": False, "error": f"fetch_error:{e}", "individuals": []}

# ---------- Canonical fetch tool (single id -> full content) ----------

@mcp.tool()
def fetch(id: str) -> Dict[str, Any]:
    """
    Canonical fetch for ChatGPT connectors.
    Given a single exhibit ID, return full document content and metadata.
    """
    if not id:
        return {"ok": False, "error": "missing_id"}
    res = fetch_exhibit(id)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "fetch_failed"), "id": id}
    pages = res.get("pages", []) or []
    content = "\n\n".join(pages) if isinstance(pages, list) else str(pages)
    meta = {k: v for k, v in res.items() if k in ("exhibit_id", "description", "filename")}
    return {"ok": True, "id": res.get("exhibit_id", id), "content": content, "metadata": meta}

# ---------- Utility/Info Tools ----------

@mcp.tool()
def list_capabilities() -> Dict[str, Any]:
    """Enumerate tool names & brief descriptions."""
    tools = [
        {"name": "search", "desc": "Canonical search: returns IDs + snippets"},
        {"name": "fetch", "desc": "Canonical fetch: returns full content by ID"},
        {"name": "edge_search", "desc": "Edge Function search (hybrid/vector/bm25)"},
        {"name": "search_legal", "desc": "Hybrid search with snippets"},
        {"name": "bm25_search", "desc": "BM25-only search via Edge Function"},
        {"name": "vector_search", "desc": "Vector-only search via Edge Function"},
        {"name": "rerank", "desc": "Rerank an existing result list"},
        {"name": "label_results", "desc": "Assign simple labels to results"},
        {"name": "fetch_exhibit", "desc": "Retrieve full pages for an exhibit"},
        {"name": "list_exhibits", "desc": "List exhibits (optionally labeled)"},
        {"name": "keyword_search", "desc": "Direct BM25-ish keyword search via PostgREST"},
        {"name": "http_fetch", "desc": "HTTP GET with allowlist & size caps"},
        {"name": "fetch_allowed", "desc": "Alias of http_fetch"},
        {"name": "list_claims", "desc": "List all claims (optional status filter)"},
        {"name": "get_facts_by_claim", "desc": "Facts for a given claim"},
        {"name": "get_facts_by_exhibit", "desc": "Facts for a given exhibit"},
        {"name": "get_case_statistics", "desc": "Counts of key tables"},
        {"name": "get_case_timeline", "desc": "Static example timeline"},
        {"name": "get_entities", "desc": "Entities list"},
        {"name": "get_individuals", "desc": "Individuals list"},
        {"name": "list_capabilities", "desc": "This list"},
    ]
    return {"ok": True, "tools": tools}

# ---------- Export HTTP app for FastCloud ----------
# IMPORTANT: Do NOT start uvicorn here; FastCloud will import `app` and run it.
app = mcp.http_app()
