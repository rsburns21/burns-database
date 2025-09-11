# Burns-Database MCP Server

This repo hosts the BurnsDB MCP server (FastMCP over HTTP) that proxies Supabase Edge functions for legal data search.

- Remote MCP endpoint (after deploy): `https://BurnsDB.fastmcp.app/mcp`
- Recommended client: `npx -y mcp-remote@0.1.29 https://BurnsDB.fastmcp.app/mcp`

Quick start (local):
- `python -m venv .venv && .\\.venv\\Scripts\\Activate.ps1`
- `pip install -r requirements.txt`
- `uvicorn server:app --host 0.0.0.0 --port 8080`

FastCloud deploy:
- Configure GitHub Variables: `SUPABASE_URL`, `PORT` (8080), `ENABLE_DIAG` (0/1), `APP_NAME` (burnsdb)
- Configure GitHub Secrets: `SUPABASE_SERVICE_ROLE_KEY` (required), optionally `OPENAI_API_KEY`
- Push to `main` to trigger deploy

ChatGPT MCP connector (client-side):
- `npx -y mcp-remote@0.1.29 https://BurnsDB.fastmcp.app/mcp`
- Or Codex CLI config override:
  `codex -c "mcp_servers.BurnsDB.command=npx" -c "mcp_servers.BurnsDB.args=[\"-y\", \"mcp-remote@0.1.29\", \"https://BurnsDB.fastmcp.app/mcp\"]"`

Health check:
- GET `https://BurnsDB.fastmcp.app/mcp` should return an MCP hello/initialize contract or redirect depending on platform.

Security:
- For production, set `auth_mode: bearer` in `fastmcp.yaml` and add `MCP_API_KEY` secret; require `Authorization: Bearer <token>` in clients.

Notes:
- Server exports `app` (ASGI). Do not self-run in cloud; workflow starts `uvicorn server:app`.
- Outbound `ALLOWLIST_HOSTS` restricts HTTP calls; configure as needed.
