# server.py
# Burns Database MCP Server — Streamable HTTP + Supabase Edge / PostgREST + optional manifest fallback
# Compatible with fastmcp 2.12.x (instantiate FastMCP() with no kwargs)

from __future__ import annotations

import os
import sys
import time
import csv
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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
SERVER_VERSION = os.getenv("APP_VERSION", "6.3").strip()
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

# Optional manifest (CSV) mapping exhibits; either PATH or URL
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "").strip()
MANIFEST_URL = os.getenv("MANIFEST_URL", "").strip()

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
# Manifest (optional)
# ------------------------------------------------------------------------------

_MANIFEST: List[Dict[str, Any]] = []

def _is_host_allowlisted(url: str) -> Tuple[bool, str]:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        return (host in ALLOWLIST_HOSTS, host)
    except Exception:
        return (False, "")

def _load_manifest() -> None:
    global _MANIFEST
    try:
        if MANIFEST_PATH and os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                _manifest_from_csv(f.read())
                print(f"[BDB] Manifest loaded from file: {MANIFEST_PATH} ({len(_MANIFEST)} rows)", file=sys.stderr)
                return
        if MANIFEST_URL:
            ok, host = _is_host_allowlisted(MANIFEST_URL)
            if not ok:
                print(f"[BDB] Manifest URL host not allowlisted: {host}", file=sys.stderr)
                return
            try:
                import asyncio
                async def _fetch():
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                        r = await client.get(MANIFEST_URL, headers={"Accept": "text/csv"})
                        r.raise_for_status()
                        return r.text
                text = asyncio.get_event_loop().run_until_complete(_fetch())
            except RuntimeError:
                # If event loop is already running (some platforms), do a simple sync fallback
                with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
                    r = client.get(MANIFEST_URL, headers={"Accept": "text/csv"})
                    r.raise_for_status()
                    text = r.text
            _manifest_from_csv(text)
            print(f"[BDB] Manifest loaded from URL: {MANIFEST_URL} ({len(_MANIFEST)} rows)", file=sys.stderr)
    except Exception as e:
        print("[BDB] Manifest load failed:", e, file=sys.stderr)

def _manifest_from_csv(text: str) -> None:
    global _MANIFEST
    _MANIFEST = []
    if not text:
        return
    try:
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            # normalize keys
            nrow = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k,v in row.items() }
            _MANIFEST.append(nrow)
    except Exception as e:
        print("[BDB] Manifest parse failed:", e, file=sys.stderr)

_load_manifest()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _truncate(s: str, n: int = SNIPPET_LENGTH) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

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

# --- Edge result parsing (robust across shapes) --------------------------------

_ID_KEYS = (
    "exhibit_id", "exhibitId", "id", "doc_id", "document_id", "documentId",
    "key", "pk", "uuid"
)
_SNIPPET_KEYS = (
    "snippet", "text", "chunk", "content", "passage", "preview", "summary"
)
_SCORE_KEYS = ("score", "similarity", "relevance", "rank", "distance")

def _prefer_list(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(d, dict):
        return []
    for k in ("results", "matches", "hits", "items", "data", "documents", "docs", "rows", "records"):
        v = d.get(k)
        if isinstance(v, list):
            return v
    # nested one deeper
    for v in d.values():
        if isinstance(v, dict):
            lst = _prefer_list(v)
            if lst:
                return lst
    return []

def _get_first(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    # case-insensitive fallback
    for k in list(d.keys()):
        lk = k.lower()
        for key in keys:
            if lk == key.lower():
                return d[k]
    return None

def _extract_results_from_edge(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    items = _prefer_list(payload)
    for r in items:
        if not isinstance(r, dict):
            continue
        rid = _get_first(r, _ID_KEYS)
        if not rid:
            continue
        rid = _normalize_exhibit_id(str(rid))
        snippet = _get_first(r, _SNIPPET_KEYS) or ""
        score = _get_first(r, _SCORE_KEYS)
        # distance -> convert to similarity-ish
        if score is None and "distance" in r:
            try:
                dist = float(r.get("distance"))
                score = 1.0 - dist
            except Exception:
                score = None
        out.append({"id": rid, "snippet": _truncate(str(snippet)), "score": score})
    return out

# ------------------------------------------------------------------------------
# Retrieval core (no decorators here)
# ------------------------------------------------------------------------------

def _resolve_edge_url() -> str:
    return EDGEFN_URL

async def _edge_search_core(
    q: str,
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    limit: Optional[int] = None,          # accept both names; MCP Inspector uses "limit"
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
    k = int(limit if limit is not None else (top_k if top_k is not None else SEARCH_TOP_K))

    # Build liberal body (tolerates different EF parameter names)
    body: Dict[str, Any] = {
        "action": "search",
        "query": q,
        "q": q,
        "mode": m,
        "searchType": m,
        "k": k, "topK": k, "limit": k,
        "min_score": float(min_score if min_score is not None else SEARCH_MIN_SCORE),
        "lexical_k": LEXICAL_K, "lexicalK": LEXICAL_K,
        "rerank": bool(rerank),
    }
    if label:
        body["label"] = label
    if metadata_filter:
        body["filter"] = metadata_filter
        body["filters"] = metadata_filter

    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, body, headers)
    out: Dict[str, Any] = {"ok": (status == 200), "status": status, "url": edge_url}
    if not out["ok"]:
        out["error"] = data if data else _truncate(text)
        return out
    out["raw"] = data
    return out

async def _lexical_fallback_embeddings(q: str, limit: int) -> List[Dict[str, Any]]:
    """
    Fallback 1: vector_embeddings.text ILIKE '%q%'
    """
    if not _SUPABASE:
        return []
    pattern = f"%{(q or '').strip().replace('%','')[:128]}%"
    try:
        resp = (
            _SUPABASE.table("vector_embeddings")
            .select("exhibit_id,text")
            .ilike("text", pattern)
            .limit(limit)
            .execute()
        )
        out: List[Dict[str, Any]] = []
        for rec in resp.data or []:
            eid = _normalize_exhibit_id(rec.get("exhibit_id") or "")
            snip = _truncate(rec.get("text") or "")
            out.append({"id": eid, "snippet": snip, "score": None})
        return out
    except Exception as e:
        print("[BDB] lexical fallback embeddings failed:", e, file=sys.stderr)
        return []

async def _metadata_probe_exhibits(q: str, limit: int) -> List[str]:
    """
    Fallback helper: search exhibits.description/filename ILIKE.
    Returns list of exhibit_ids.
    """
    ids: List[str] = []
    if not _SUPABASE:
        return ids
    pattern = f"%{(q or '').strip().replace('%','')[:128]}%"
    try:
        # description
        r1 = (
            _SUPABASE.table("exhibits")
            .select("exhibit_id,description")
            .ilike("description", pattern)
            .limit(limit * 2)
            .execute()
        )
        for rec in (r1.data or []):
            eid = _normalize_exhibit_id(rec.get("exhibit_id") or "")
            if eid and eid not in ids:
                ids.append(eid)
        # filename
        r2 = (
            _SUPABASE.table("exhibits")
            .select("exhibit_id,filename")
            .ilike("filename", pattern)
            .limit(limit * 2)
            .execute()
        )
        for rec in (r2.data or []):
            eid = _normalize_exhibit_id(rec.get("exhibit_id") or "")
            if eid and eid not in ids:
                ids.append(eid)
    except Exception as e:
        print("[BDB] exhibits metadata probe failed:", e, file=sys.stderr)
    return ids[: max(1, limit * 2)]

async def _first_page_snippet_for(eid: str) -> str:
    """
    Fetches first page/chunk for an exhibit to make a snippet.
    """
    if not _SUPABASE:
        return ""
    try:
        q = (
            _SUPABASE.table("vector_embeddings")
            .select("text,page,chunk_index")
            .eq("exhibit_id", _normalize_exhibit_id(eid))
            .limit(1)
        )
        try:
            q = q.order("page")
        except Exception:
            pass
        try:
            q = q.order("chunk_index")
        except Exception:
            pass
        r = q.execute()
        if r.data:
            return _truncate(r.data[0].get("text") or "")
    except Exception:
        pass
    return ""

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
# Manifest helpers
# ------------------------------------------------------------------------------

def _manifest_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Very simple match over manifest fields: exhibit_id, description, filename, title, tags, custodian.
    Returns list of {id, snippet, score=None}.
    """
    if not _MANIFEST or not query:
        return []
    q = (query or "").strip().lower()
    fields = ("exhibit_id","description","filename","title","tags","custodian")
    hits: List[Tuple[str, str]] = []  # (id, snippet)
    seen: set = set()
    for row in _MANIFEST:
        try:
            text_parts = []
            exhibit_id = ""
            for key in row.keys():
                lk = key.lower()
                if lk in fields:
                    val = str(row.get(key) or "")
                    text_parts.append(val)
                    if lk == "exhibit_id" and not exhibit_id:
                        exhibit_id = val
            blob = " ".join(text_parts).lower()
            if q in blob:
                eid = _normalize_exhibit_id(exhibit_id or "")
                if eid and eid not in seen:
                    snippet = row.get("description") or row.get("title") or row.get("filename") or ""
                    hits.append((eid, snippet))
                    seen.add(eid)
            if len(hits) >= limit:
                break
        except Exception:
            continue
    return [{"id": eid, "snippet": _truncate(sn), "score": None} for (eid, sn) in hits[:limit]]

# ------------------------------------------------------------------------------
# MCP server + tools
# ------------------------------------------------------------------------------

mcp = FastMCP()  # IMPORTANT: no kwargs for 2.12.x compatibility

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
        "manifest_rows": len(_MANIFEST),
    }

@mcp.tool(description="Adjust default search behavior.")
async def set_defaults(
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    limit: Optional[int] = None,            # accept limit too
    min_score: Optional[float] = None,
    lexical_k: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,  # MCP Inspector sometimes supplies this
) -> Dict[str, Any]:
    global SEARCH_MODE, SEARCH_TOP_K, SEARCH_MIN_SCORE, LEXICAL_K
    if mode:
        SEARCH_MODE = mode.lower().strip()
    # prefer explicit limit if provided
    if limit is not None:
        SEARCH_TOP_K = int(limit)
    if top_k is not None:
        SEARCH_TOP_K = int(top_k)
    if min_score is not None:
        SEARCH_MIN_SCORE = float(min_score)
    if lexical_k is not None:
        LEXICAL_K = int(lexical_k)
    return {"ok": True, "defaults": {
        "mode": SEARCH_MODE, "top_k": SEARCH_TOP_K, "min_score": SEARCH_MIN_SCORE, "lexical_k": LEXICAL_K
    }}

# ----- Search family (robust EF parse + Supabase + manifest fallbacks) ---------

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
    # 1) Try Edge Function
    base = await _edge_search_core(
        q=query,
        mode=mode,
        top_k=limit,
        limit=limit,
        min_score=min_score,
        rerank=rerank,
        label=label,
        metadata_filter=metadata_filter,
    )
    items: List[Dict[str, Any]] = []
    if base.get("ok"):
        items = _extract_results_from_edge(base.get("raw", {}) or {})

    # 2) Fallback: lexical within vector_embeddings
    if not items:
        embeds = await _lexical_fallback_embeddings(query, limit)
        items.extend(embeds)

    # 3) Fallback: exhibits metadata probe → first-page snippets
    if not items:
        eids = await _metadata_probe_exhibits(query, limit)
        for eid in eids[:limit]:
            snip = await _first_page_snippet_for(eid)
            if snip:
                items.append({"id": _normalize_exhibit_id(eid), "snippet": snip, "score": None})

    # 4) Fallback: manifest search
    if not items:
        items = _manifest_search(query, limit)

    # Dedup and cap
    dedup: Dict[str, Dict[str, Any]] = {}
    for it in items:
        eid = _normalize_exhibit_id(it.get("id") or "")
        if not eid:
            continue
        if eid not in dedup:
            dedup[eid] = {"id": eid, "snippet": it.get("snippet") or "", "score": it.get("score")}
    final = list(dedup.values())[:max(1, limit)]

    return {"ok": True, "results": final, "ids": [it["id"] for it in final]}

@mcp.tool(description="Hybrid search via Edge (returns raw edge payload alongside parsed results).")
async def edge_search(
    q: str,
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
    limit: Optional[int] = None,            # MCP Inspector passes "limit"
    min_score: Optional[float] = None,
    rerank: Optional[bool] = True,
    label: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = await _edge_search_core(q, mode, top_k, limit, min_score, rerank, label, metadata_filter)
    parsed = _extract_results_from_edge(base.get("raw", {}) or {}) if base.get("ok") else []
    return {"ok": base.get("ok", False), "status": base.get("status"), "url": base.get("url"),
            "parsed": parsed, "raw": base.get("raw") if ENABLE_DIAG else {}}

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

# ----- Spec‑compliant fetch ----------------------------------------------------

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

# ----- Additional helpers ------------------------------------------------------

@mcp.tool(description="Retrieve full text pages for an exhibit (server-side PostgREST).")
async def fetch_exhibit(exhibit_id: str) -> Dict[str, Any]:
    ok, pages, err = await _fetch_exhibit_pages(exhibit_id)
    return {"ok": ok, "exhibit_id": _normalize_exhibit_id(exhibit_id), "pages": pages, "error": (None if ok else err)}

@mcp.tool(description="List exhibits; optionally attach simple categories.")
async def list_exhibits(withLabels: bool = False) -> Dict[str, Any]:
    def _cat(desc: str, fn: str = "") -> str:
        t = (desc or "").lower() + " " + (fn or "").lower()
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
                rec["category"] = _cat(rec.get("description", ""), rec.get("filename", ""))
        return {"ok": True, "exhibits": data}
    except Exception as e:
        return {"ok": False, "error": f"list_exhibits_failed: {e}", "exhibits": []}

# ----- Diagnostics (accept loose payloads for MCP Inspector) -------------------

@mcp.tool(description="Show config and defaults.")
async def diag_config(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA},
        "edge": {"url": _resolve_edge_url()},
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "supabase_url_set": bool(SUPABASE_URL),
        "openai_key_set": bool(OPENAI_API_KEY),
        "manifest_rows": len(_MANIFEST),
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

@mcp.tool(description="Basic health check (accepts optional text/message/payload).")
async def health(
    text: Optional[str] = None,
    message: Optional[str] = None,
    input: Optional[str] = None,                  # some inspectors send 'input'
    prompt: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"ok": True, "server": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA}

# ------------------------------------------------------------------------------
# Extra HTTP routes (/ and /health)
# ------------------------------------------------------------------------------

async def _root_ok(request: Request):
    return JSONResponse({
        "ok": True, "name": SERVER_NAME, "version": SERVER_VERSION, "git_sha": GIT_SHA,
        "ts": int(time.time()), "mcp_path": MCP_PATH, "manifest_rows": len(_MANIFEST)
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
        "manifest_rows": len(_MANIFEST),
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
