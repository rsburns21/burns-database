# server.py
# Server: BDB
# Version: 4.4.0

from __future__ import annotations

import os
import httpx
from typing import Optional
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

SERVER_NAME = "BDB"
SERVER_VERSION = "4.4.0"

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_SECRET", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EXPECTED_BEARER = os.getenv("BDB_AUTH_TOKEN", "").strip()

# Allowed hosts for outbound calls
ALLOWLIST_DEFAULT = [
    "nqkzqcsqfvpquticvwmk.supabase.co",
    "nqkzqcsqfvpquticvwmk.functions.supabase.co",
    "api.openai.com",
    "chatgpt.com",
    "fastmcp.cloud",
    "burns-database.fastmcp.app",
    "raw.githubusercontent.com",
    "httpbin.org",
    "playground.ai.cloudflare.com",
    "example.com",
]

HTTP_TIMEOUT_S = 20.0

app = FastAPI(title=SERVER_NAME)

# CORS configuration for ChatGPT and FastMCP
ALLOWED_ORIGINS = [
    "https://chat.openai.com",
    "https://fastmcp.cloud",
    "https://burns-database.fastmcp.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["*"],
)

@app.options("/mcp")
@app.options("/mcp/{path:path}")
async def _options_ok(path: str = "") -> Response:
    """Return empty response for OPTIONS preflight."""
    return Response(status_code=204)

@app.head("/mcp")
@app.head("/mcp/{path:path}")
async def _head_ok(path: str = "") -> Response:
    """Return empty response for HEAD requests."""
    return Response(status_code=204)

def require_bearer(request: Request) -> None:
    """Verify Authorization Bearer token matches EXPECTED_BEARER"""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = auth.split(" ", 1)[1].strip()
    if not token or not EXPECTED_BEARER or token != EXPECTED_BEARER:
        raise HTTPException(status_code=401, detail="Invalid token")

AuthDep = Depends(require_bearer)

# Diagnostics endpoints
class HealthOut(BaseModel):
    name: str
    version: str
    ok: bool

@app.get("/mcp/health")
async def health(_: None = AuthDep) -> HealthOut:
    """Server health check"""
    return HealthOut(name=SERVER_NAME, version=SERVER_VERSION, ok=True)

class DiagConfigOut(BaseModel):
    name: str
    version: str
    allowlist: list[str]
    supabase_url: Optional[str] = None
    openai: bool = False

@app.get("/mcp/diag_config")
async def diag_config(_: None = AuthDep) -> DiagConfigOut:
    """Return configuration details for diagnostics"""
    return DiagConfigOut(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        allowlist=ALLOWLIST_DEFAULT,
        supabase_url=SUPABASE_URL or None,
        openai=bool(OPENAI_API_KEY),
    )

class SBHealthOut(BaseModel):
    auth_ok: bool
    rest_ok: bool
    status_auth: int
    status_rest: int

@app.post("/mcp/tools/supabase_health")
async def supabase_health(_: None = AuthDep) -> SBHealthOut:
    """Check Supabase Auth and REST endpoints"""
    headers = {}
    key = SUPABASE_SERVICE_ROLE_KEY
    if key:
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    auth_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/health" if SUPABASE_URL else ""
    rest_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/" if SUPABASE_URL else ""
    status_auth = 0
    status_rest = 0

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        if auth_url:
            try:
                resp = await client.get(auth_url, headers=headers)
                status_auth = resp.status_code
            except Exception:
                status_auth = 0
        if rest_url:
            try:
                resp2 = await client.head(rest_url, headers=headers)
                status_rest = resp2.status_code
            except Exception:
                status_rest = 0

    return SBHealthOut(
        auth_ok=(status_auth == 200),
        rest_ok=(status_rest in (200, 204, 405)),
        status_auth=status_auth,
        status_rest=status_rest,
    )

# Note: Additional /mcp/* tool endpoints should import and check AuthDep accordingly
