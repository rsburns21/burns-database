"""
BDB (BurnsDB) — FastMCP server (single‑file edition)
====================================================

This module exposes a FastMCP server for the Burns Database.  The primary
goal is to provide a minimal, deployment‑safe implementation that will work
across multiple versions of the underlying ``fastmcp`` package.  In
particular, earlier releases of ``fastmcp`` do not support the
``description`` keyword argument on the ``FastMCP`` constructor, and some
releases omit the ``version`` keyword entirely.  To remain compatible,
we wrap construction in a helper that tries the most specific signature
first and falls back to simpler signatures if necessary.

The server proxies a single Supabase Edge Function (``advanced_semantic_search``)
for legal discovery search and exposes a suite of tools (search variants,
rerank, labeling, diagnostics, allowlist fetchers, etc.) via the FastMCP
framework.  See the README for deployment instructions and the healthcheck
script for connectivity verification.
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP
from fastapi import FastAPI
from pydantic import BaseModel, Field

###############################################################################
# Server Identity & Defaults
###############################################################################

SERVER_NAME = "BDB"
# Bump the minor version to indicate this file includes compatibility fixes.
SERVER_VERSION = "4.3.1"

# Hard‑wired Edge Function (override with EDGEFN_URL if needed)
EDGEFN_URL_DEFAULT = (
    "https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search"
)
EDGEFN_URL = os.getenv("EDGEFN_URL", EDGEFN_URL_DEFAULT).strip()

# Keys (service role only for server‑to‑server)
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
    "SUPABASE_SERVICE_KEY"
) or ""
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # optional catch‑all

# OpenAI (used for rerank)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Strict allowlist for outbound calls (scheme ignored; host matched).  Hosts
# listed here are considered safe to contact via HTTP.
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
ALLOWLIST_HOSTS = [
    h.strip().lower()
    for h in (os.getenv("ALLOWLIST_HOSTS", ",".join(ALLOWLIST_DEFAULT))).split(",")
    if h.strip()
]

# Fetch and snippet sizing
MAX_FETCH_SIZE = int(os.getenv("MAX_FETCH_SIZE", "1000000"))
SNIPPET_LENGTH = int(os.getenv("SNIPPET_LENGTH", "500"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "20"))

# Search defaults
SEARCH_MODE = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()  # vector|bm25|hybrid|semantic
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
LEXICAL_K = int(os.getenv("LEXICAL_K", "0"))  # reserved

# Embedding dimension (vector_embeddings: 384)
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

###############################################################################
# Helpers
###############################################################################

def _resolve_edge_url() -> str:
    """Return the configured edge function URL."""
    return EDGEFN_URL


def make_edge_headers() -> Dict[str, str]:
    """
    Always include ``apikey`` and ``Authorization`` when a key is available.
    This supports both opaque service‑role strings (non‑JWT) and JWTs alike.
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY
    if key:
        headers.update({"apikey": key, "Authorization": f"Bearer {key}"})
    return headers


def _is_host_allowlisted(url: str) -> Tuple[bool, str]:
    """Check if the host in the URL is on the allowlist."""
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        return (host in ALLOWLIST_HOSTS, host)
    except Exception:
        return (False, "")


async def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, Any], str]:
    """Helper to POST JSON and parse the response as JSON or text."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        resp = await client.post(url, json=body, headers=headers)
        text = resp.text
        try:
            data = await resp.json()
        except Exception:
            data = {}
        return (resp.status_code, data, text)


def _truncate(s: str, n: int = SNIPPET_LENGTH) -> str:
    """Truncate a string to at most ``n`` characters, appending an ellipsis."""
    if not s:
        return s
    return s if len(s) <= n else s[: n - 3] + "..."


def _now_ms() -> int:
    """Return the current time in milliseconds since epoch."""
    return int(time.time() * 1000)


# Shared in‑memory defaults (tunable via tools).  Using a Pydantic model
# ensures type safety and simple serialization.
class Defaults(BaseModel):
    mode: str = SEARCH_MODE
    top_k: int = SEARCH_TOP_K
    min_score: float = SEARCH_MIN_SCORE
    lexical_k: int = LEXICAL_K
    max_fetch_size: int = MAX_FETCH_SIZE
    snippet_length: int = SNIPPET_LENGTH


DEFAULTS = Defaults()

###############################################################################
# FastMCP Server Construction
###############################################################################

def _create_mcp(name: str, version: str) -> FastMCP:
    """
    Create a FastMCP instance while gracefully handling version mismatches.

    The ``fastmcp`` library has evolved over time.  Some releases accept a
    ``version`` keyword argument, while others require only the name.  Newer
    releases may also accept a ``description`` keyword but older versions
    reject it, causing a ``TypeError``.  To maintain broad compatibility,
    this helper tries the most explicit constructor first and falls back to
    simpler forms as necessary.
    """
    try:
        # Attempt to use the full signature.  If the installed fastmcp
        # supports ``version`` but not ``description``, this should work.
        return FastMCP(name, version=version)
    except TypeError:
        # Fall back to passing only the name (oldest API).
        return FastMCP(name)


# Instantiate the MCP server.  The description is intentionally omitted to
# avoid issues with older fastmcp versions that do not accept it.
mcp = _create_mcp(SERVER_NAME, SERVER_VERSION)

###############################################################################
# Schemas
###############################################################################

# Pydantic models capturing the shape of various tool parameters.  See the
# corresponding tool functions below for usage.
class SearchParams(BaseModel):
    q: str = Field(..., description="User query")
    mode: str = Field(SEARCH_MODE, description="vector|bm25|hybrid|semantic")
    top_k: int = Field(SEARCH_TOP_K, ge=1, le=200)
    min_score: float = Field(SEARCH_MIN_SCORE, ge=0.0, description="Score threshold (edge-defined)")
    rerank: bool = Field(False, description="Apply OpenAI-based reranking (requires OPENAI_API_KEY)")
    label: Optional[str] = Field(None, description="Optional label to attach to this search intent")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None, description="Optional filter dict passed to edge function"
    )


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
    action: str = Field(..., description="Edge function action (e.g., 'embed_query')")
    payload: Dict[str, Any] = Field(default_factory=dict)


###############################################################################
# Utility Tools (1-7)
###############################################################################

@mcp.tool(description="Echo text back.")
async def echo(text: str) -> Dict[str, Any]:
    return {"echo": text, "ts": _now_ms()}


@mcp.tool(description="List server capabilities and config.")
async def list_capabilities() -> Dict[str, Any]:
    return {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "edge": {"url": _resolve_edge_url()},
        "defaults": DEFAULTS.model_dump(),
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "embed_dim": EMBED_DIM,
        "openai_enabled": bool(OPENAI_API_KEY),
    }


@mcp.tool(description="Return version info.")
async def version_info() -> Dict[str, Any]:
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
        body = r.text or ""
        return {"ok": r.status_code == 200, "status": r.status_code, "host": host, "text": _truncate(body)}


@mcp.tool(description="HTTP HEAD (allowlisted only).")
async def http_head_allowed(params: FetchParams) -> Dict[str, Any]:
    ok, host = _is_host_allowlisted(params.url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        r = await client.head(params.url)
        return {"ok": r.status_code == 200, "status": r.status_code, "headers": dict(r.headers)}


@mcp.tool(description="HTTP POST JSON (allowlisted only). Returns JSON or text snippet.")
async def http_post_allowed(url: str, body_json: Dict[str, Any]) -> Dict[str, Any]:
    ok, host = _is_host_allowlisted(url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        r = await client.post(url, json=body_json)
        try:
            data = await r.json()
            return {"ok": r.status_code == 200, "status": r.status_code, "json": data}
        except Exception:
            return {"ok": r.status_code == 200, "status": r.status_code, "text": _truncate(r.text)}


###############################################################################
# Diagnostics (8-13)
###############################################################################

@mcp.tool(description="Check server + edge + OpenAI health quickly.")
async def health() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION, "ts": _now_ms()},
        "edge": {},
        "openai": {},
    }

    # Edge ping
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        out["edge"] = {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}
    else:
        headers = make_edge_headers()
        status, data, text = await _post_json(edge_url, {"ping": True}, headers)
        out["edge"] = {
            "ok": status == 200,
            "status": status,
            "url": edge_url,
            "body": data if data else _truncate(text),
        }

    # OpenAI sanity if key available
    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                r = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                )
                out["openai"] = {"ok": r.status_code in (200, 401, 403), "status": r.status_code}
        except Exception as e:
            out["openai"] = {"ok": False, "error": str(e)}
    else:
        out["openai"] = {"ok": False, "error": "OPENAI_API_KEY not set"}

    return out


@mcp.tool(description="Detailed edge diagnostic POST with optional payload.")
async def diag_edge(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}
    status, data, text = await _post_json(edge_url, payload or {"ping": True}, headers)
    return {
        "ok": status == 200,
        "status": status,
        "url": edge_url,
        "json": data if data else {},
        "text": _truncate(text),
    }


@mcp.tool(description="Show config and defaults (raw).")
async def diag_config() -> Dict[str, Any]:
    return {
        "server": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "edge": {"url": _resolve_edge_url()},
        "env_presence": {
            "SUPABASE_SERVICE_ROLE_KEY": bool(SUPABASE_SERVICE_ROLE_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        },
        "allowlist_hosts": ALLOWLIST_HOSTS,
        "defaults": DEFAULTS.model_dump(),
        "http_timeout_s": HTTP_TIMEOUT_S,
    }


@mcp.tool(description="Quick allowlist/host report.")
async def diag_allowlist() -> Dict[str, Any]:
    return {"allowlist_hosts": ALLOWLIST_HOSTS}


@mcp.tool(description="OpenAI connectivity smoke check.")
async def diag_openai() -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"ok": False, "error": "OPENAI_API_KEY not set"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            r = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            )
            return {"ok": r.status_code in (200, 401, 403), "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


###############################################################################
# Search & Retrieval (14-26)
###############################################################################

async def _edge_search_core(params: SearchParams) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist", "url": edge_url}

    body = {
        "query": params.q,
        "mode": (params.mode or DEFAULTS.mode).lower(),
        "top_k": params.top_k or DEFAULTS.top_k,
        "min_score": params.min_score
        if params.min_score is not None
        else DEFAULTS.min_score,
        "lexical_k": DEFAULTS.lexical_k,
    }
    if params.label:
        body["label"] = params.label
    if params.metadata_filter:
        body["filter"] = params.metadata_filter

    headers = make_edge_headers()
    status, data, text = await _post_json(edge_url, body, headers)
    out: Dict[str, Any] = {
        "ok": status == 200,
        "status": status,
        "mode": body["mode"],
        "url": edge_url,
    }
    if not out["ok"]:
        out["error"] = data if data else _truncate(text)
        return out

    results = data.get("results") or data.get("matches") or []
    out["raw"] = data

    # Optional rerank using OpenAI (if enabled & key present)
    if params.rerank and OPENAI_API_KEY and results:
        try:
            out["reranked"] = await _rerank_openai(params.q, results)
        except Exception as e:
            out["rerank_error"] = str(e)

    return out


async def _rerank_openai(query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Simple rerank: send query + candidates to OpenAI for scoring using a lightweight model.
    Requires ``OPENAI_API_KEY``.
    """
    cands: List[str] = []
    for r in results:
        txt = r.get("text") or r.get("content") or json.dumps(r)
        cands.append(txt)

    limit = min(len(cands), top_k or DEFAULTS.top_k)
    cands = cands[:limit]

    prompt = (
        "You are a ranking function. Score each candidate passage for relevance to the user query on a 0-100 scale.\n"
        f"Query: {query}\n\nCandidates:\n"
    )
    for i, c in enumerate(cands, 1):
        prompt += f"[{i}] {c}\n"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "input": [
            {
                "role": "user",
                "content": prompt
                + "\nReturn a JSON array of numbers only (scores aligned with index order).",
            }
        ],
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        j = await r.json()
        out_text = ""
        try:
            out_text = j["output"][0]["content"][0]["text"]  # new Responses format
        except Exception:
            try:
                out_text = j["choices"][0]["message"]["content"]  # legacy compat
            except Exception:
                out_text = ""
        try:
            scores = json.loads(out_text)
            if not isinstance(scores, list):
                raise ValueError("scores not a list")
        except Exception:
            # fallback: uniform scores if parse fails
            scores = [50] * len(cands)

    scored: List[Dict[str, Any]] = []
    for idx, r in enumerate(results[:limit]):
        rr = dict(r)
        rr["_rerank"] = float(scores[idx] if idx < len(scores) else 50.0)
        scored.append(rr)
    scored.sort(key=lambda x: x.get("_rerank", 0.0), reverse=True)
    return scored


@mcp.tool(description="General search via Edge (vector|bm25|hybrid|semantic). Supports OpenAI rerank.")
async def edge_search(params: SearchParams) -> Dict[str, Any]:
    return await _edge_search_core(params)


@mcp.tool(description="BM25 search via Edge.")
async def bm25_search(
    q: str, top_k: int = SEARCH_TOP_K, min_score: float = SEARCH_MIN_SCORE, rerank: bool = False
) -> Dict[str, Any]:
    return await _edge_search_core(
        SearchParams(q=q, mode="bm25", top_k=top_k, min_score=min_score, rerank=rerank)
    )


@mcp.tool(description="Vector similarity search via Edge.")
async def vector_search(
    q: str, top_k: int = SEARCH_TOP_K, min_score: float = SEARCH_MIN_SCORE, rerank: bool = False
) -> Dict[str, Any]:
    return await _edge_search_core(
        SearchParams(q=q, mode="vector", top_k=top_k, min_score=min_score, rerank=rerank)
    )


@mcp.tool(description="Hybrid search (vector + BM25) via Edge.")
async def hybrid_search(
    q: str, top_k: int = SEARCH_TOP_K, min_score: float = SEARCH_MIN_SCORE, rerank: bool = False
) -> Dict[str, Any]:
    return await _edge_search_core(
        SearchParams(q=q, mode="hybrid", top_k=top_k, min_score=min_score, rerank=rerank)
    )


@mcp.tool(description="Semantic search (alias to hybrid unless Edge defines differently).")
async def semantic_search(
    q: str, top_k: int = SEARCH_TOP_K, min_score: float = SEARCH_MIN_SCORE, rerank: bool = False
) -> Dict[str, Any]:
    return await _edge_search_core(
        SearchParams(q=q, mode="semantic", top_k=top_k, min_score=min_score, rerank=rerank)
    )


@mcp.tool(description="Request embeddings for a query from Edge (action=embed_query).")
async def embed_query(q: str) -> Dict[str, Any]:
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


@mcp.tool(description="Label a document or match by posting to Edge (action=label).")
async def label_result(params: LabelParams) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    body = {"action": "label", "doc_id": params.doc_id, "label": params.label}
    status, data, text = await _post_json(edge_url, body, headers)
    return {
        "ok": status == 200,
        "status": status,
        "json": data if data else {},
        "text": _truncate(text),
    }


@mcp.tool(description="List collections/sources from Edge if supported.")
async def list_collections() -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    status, data, text = await _post_json(edge_url, {"action": "list_collections"}, headers)
    return {
        "ok": status == 200,
        "status": status,
        "json": data if data else {},
        "text": _truncate(text),
    }


@mcp.tool(description="Fetch a document (by id) via Edge if supported.")
async def get_document(doc_id: str) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    status, data, text = await _post_json(edge_url, {"action": "get_document", "doc_id": doc_id}, headers)
    return {
        "ok": status == 200,
        "status": status,
        "json": data if data else {},
        "text": _truncate(text),
    }


@mcp.tool(description="Reciprocal Rank Fusion of two result sets (client-side).")
async def combine_results(
    set_a: List[Dict[str, Any]], set_b: List[Dict[str, Any]], k: int = 60
) -> Dict[str, Any]:
    """
    Combine two ranked result sets using Reciprocal Rank Fusion (RRF).  The
    RRF score is the sum of ``1 / (k + rank)`` across both sets for each
    document identifier.
    """
    def index_by_id(items: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for it in items:
            ids.append(it.get("id") or it.get("doc_id") or json.dumps(it)[:64])
        return ids

    ids_a = index_by_id(set_a)
    ids_b = index_by_id(set_b)

    scores: Dict[str, float] = {}
    for rank, id_ in enumerate(ids_a, 1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    for rank, id_ in enumerate(ids_b, 1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)

    fused: List[Dict[str, Any]] = []
    seen: Dict[str, bool] = {}

    def find_row(id_: str) -> Dict[str, Any]:
        for it in set_a:
            if (it.get("id") or it.get("doc_id")) == id_:
                return dict(it)
        for it in set_b:
            if (it.get("id") or it.get("doc_id")) == id_:
                return dict(it)
        return {"id": id_}

    for id_, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        if id_ in seen:
            continue
        row = find_row(id_)
        row["_rrf"] = sc
        fused.append(row)
        seen[id_] = True

    return {"ok": True, "count": len(fused), "results": fused}


###############################################################################
# Rerank Utilities (27-29)
###############################################################################

@mcp.tool(description="Rerank a set of results with OpenAI (expects 'text' per item).")
async def rerank_by_openai(query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"ok": False, "error": "OPENAI_API_KEY not set"}
    ranked = await _rerank_openai(query, results, top_k=top_k)
    return {"ok": True, "results": ranked}


@mcp.tool(description="BM25-like lightweight rerank (client-side heuristic).")
async def rerank_by_bm25(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Very simple BM25-ish scoring (heuristic). Not a full implementation; useful as a
    client-side tiebreaker when exact BM25 is unavailable.
    """
    q_terms = [t for t in query.lower().split() if t]
    def score(txt: str) -> float:
        t = txt.lower()
        hits = sum(t.count(qt) for qt in q_terms)
        return float(hits)
    scored: List[Dict[str, Any]] = []
    for r in results:
        txt = r.get("text") or r.get("content") or ""
        rr = dict(r)
        rr["_bm25h"] = score(txt)
        scored.append(rr)
    scored.sort(key=lambda x: x.get("_bm25h", 0.0), reverse=True)
    return {"ok": True, "results": scored}


@mcp.tool(description="Combine OpenAI rerank (first) then BM25 heuristic as a tie resolver.")
async def rerank_hybrid(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = await rerank_by_openai(query, results)
    if not first.get("ok"):
        return first
    ranked = first["results"]
    if len(ranked) >= 2 and ranked[0].get("_rerank") == ranked[1].get("_rerank"):
        tie = await rerank_by_bm25(query, ranked[:2])
        anchored = tie["results"] + ranked[2:]
        return {"ok": True, "results": anchored}
    return {"ok": True, "results": ranked}


###############################################################################
# Defaults Management (30-32)
###############################################################################

@mcp.tool(description="Set session defaults (mode/top_k/min_score/etc).")
async def set_defaults(params: SetDefaultsParams) -> Dict[str, Any]:
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
async def get_defaults() -> Dict[str, Any]:
    return {"ok": True, "defaults": DEFAULTS.model_dump()}


@mcp.tool(description="Reset session defaults to startup values.")
async def reset_defaults() -> Dict[str, Any]:
    DEFAULTS.mode = (os.getenv("SEARCH_MODE") or "hybrid").strip().lower()
    DEFAULTS.top_k = int(os.getenv("SEARCH_TOP_K", "10"))
    DEFAULTS.min_score = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
    DEFAULTS.lexical_k = int(os.getenv("LEXICAL_K", "0"))
    globals()["MAX_FETCH_SIZE"] = int(os.getenv("MAX_FETCH_SIZE", "1000000"))
    globals()["SNIPPET_LENGTH"] = int(os.getenv("SNIPPET_LENGTH", "500"))
    return {"ok": True, "defaults": DEFAULTS.model_dump()}


###############################################################################
# Generic Edge Actions (33)
###############################################################################

@mcp.tool(description="Call Edge with a raw action/payload (advanced).")
async def edge_action(params: EdgeActionParams) -> Dict[str, Any]:
    edge_url = _resolve_edge_url()
    headers = make_edge_headers()
    body = dict(params.payload or {})
    body["action"] = params.action
    ok, host = _is_host_allowlisted(edge_url)
    if not ok:
        return {"ok": False, "error": f"host '{host}' not in allowlist"}
    status, data, text = await _post_json(edge_url, body, headers)
    return {
        "ok": status == 200,
        "status": status,
        "json": data if data else {},
        "text": _truncate(text),
    }


###############################################################################
# Codex Background Agent Management (34-40)
###############################################################################

class CodexAgentParams(BaseModel):
    action: str = Field(..., description="Agent action: start|stop|status|restart|add_task")
    task_data: Optional[Dict[str, Any]] = Field(None, description="Task data for add_task action")
    task_type: Optional[str] = Field(
        None,
        description="Task type: codex_analysis|document_processing|search_optimization|embedding_generation",
    )
    config_override: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")


@mcp.tool(
    description="Manage the codex background agent (start/stop/status/restart/add_task)."
)
async def codex_agent_control(params: CodexAgentParams) -> Dict[str, Any]:
    import uuid
    action = params.action.lower()
    try:
        if action == "start":
            result = subprocess.run(
                [sys.executable, "launch_codex_agent.py", "--daemon"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "ok": result.returncode == 0,
                "action": "start",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        elif action == "stop":
            result = subprocess.run(
                [sys.executable, "launch_codex_agent.py", "stop"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "ok": result.returncode == 0,
                "action": "stop",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        elif action == "status":
            result = subprocess.run(
                [sys.executable, "launch_codex_agent.py", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "ok": result.returncode == 0,
                "action": "status",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "running": result.returncode == 0,
            }
        elif action == "restart":
            result = subprocess.run(
                [sys.executable, "launch_codex_agent.py", "restart"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "ok": result.returncode == 0,
                "action": "restart",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        elif action == "add_task":
            if not params.task_type or not params.task_data:
                return {
                    "ok": False,
                    "error": "task_type and task_data are required for add_task action",
                }
            task_id = str(uuid.uuid4())
            task_data = {
                "task_id": task_id,
                "task_type": params.task_type,
                "payload": params.task_data,
                "created_at": _now_ms(),
            }
            tmpdir = tempfile.gettempdir()
            task_file = os.path.join(tmpdir, f"codex_task_{task_id}.json")
            try:
                with open(task_file, "w") as f:
                    json.dump(task_data, f, indent=2)
                return {
                    "ok": True,
                    "action": "add_task",
                    "task_id": task_id,
                    "task_file": task_file,
                    "message": "Task queued for processing",
                }
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Failed to create task file: {e}",
                }
        else:
            return {
                "ok": False,
                "error": f"Unknown action: {action}. Supported actions: start, stop, status, restart, add_task",
            }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Operation timed out"}
    except Exception as e:
        return {"ok": False, "error": f"Agent control error: {e}"}


@mcp.tool(description="Get codex agent configuration and status.")
async def codex_agent_info() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, "launch_codex_agent.py", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        is_running = result.returncode == 0
        config_info = {
            "agent_name": os.getenv("CODEX_AGENT_NAME", "burns-codex-agent"),
            "max_workers": os.getenv("CODEX_MAX_WORKERS", "4"),
            "check_interval": os.getenv("CODEX_CHECK_INTERVAL", "30.0"),
            "enable_health_checks": os.getenv("CODEX_ENABLE_HEALTH_CHECKS", "true"),
            "log_level": os.getenv("CODEX_LOG_LEVEL", "INFO"),
            "log_file": os.getenv("CODEX_LOG_FILE"),
            "pid_file": os.getenv(
                "CODEX_PID_FILE",
                os.path.join(tempfile.gettempdir(), "codex_agent.pid"),
            ),
            "supabase_configured": bool(
                os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            ),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "codex_model": os.getenv("CODEX_MODEL", "gpt-4"),
            "codex_endpoint": os.getenv(
                "CODEX_ENDPOINT",
                "https://api.openai.com/v1/chat/completions",
            ),
            "max_task_duration": os.getenv("CODEX_MAX_TASK_DURATION", "300"),
            "task_retry_attempts": os.getenv("CODEX_TASK_RETRY_ATTEMPTS", "3"),
            "task_retry_delay": os.getenv("CODEX_TASK_RETRY_DELAY", "5.0"),
        }
        return {
            "ok": True,
            "running": is_running,
            "status_output": result.stdout or "",
            "config": config_info,
            "supported_task_types": [
                "codex_analysis",
                "document_processing",
                "search_optimization",
                "embedding_generation",
            ],
        }
    except Exception as e:
        return {"ok": False, "error": f"Failed to get agent info: {e}"}


@mcp.tool(description="Create a sample codex agent configuration file.")
async def codex_agent_create_config() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, "launch_codex_agent.py", "create-config"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "message": "Sample configuration file created: codex_agent_config.json",
        }
    except Exception as e:
        return {"ok": False, "error": f"Failed to create config: {e}"}


###############################################################################
# ASGI app export for FastCloud
###############################################################################

def build_app() -> FastAPI:
    """Construct the FastAPI app and mount the MCP under ``/mcp``."""
    app = FastAPI()

    @app.get("/health")
    async def health_http():
        edge_url = _resolve_edge_url()
        ok, host = _is_host_allowlisted(edge_url)
        if not ok:
            return {
                "ok": False,
                "edge": {"url": edge_url, "error": f"host '{host}' not in allowlist"},
            }
        headers = make_edge_headers()
        status, data, text = await _post_json(edge_url, {"ping": True}, headers)
        return {
            "ok": status == 200,
            "edge": {
                "url": edge_url,
                "status": status,
                "body": data if data else _truncate(text),
            },
            "server": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    app.mount("/mcp", mcp.http_app())
    return app


# Exported for platform discovery (FastCloud)
app = build_app()

if __name__ == "__main__":
    # Local dev only (FastCloud uses its own runner; this block won't execute there)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
