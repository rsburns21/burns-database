# Burns-Database MCP Server

This repo hosts the BurnsDB MCP server (FastMCP over HTTP) that proxies a single Supabase Edge Function (`advanced_semantic_search`) for legal discovery search.

- Remote MCP endpoint (after deploy): `https://BurnsDB.fastmcp.app/mcp`
- Recommended client: `npx -y mcp-remote@0.1.29 https://BurnsDB.fastmcp.app/mcp`

Quick start (local):
- `python -m venv .venv && .\\.venv\\Scripts\\Activate.ps1`
- `pip install -r requirements.txt`
- `uvicorn server:app --host 0.0.0.0 --port 8080`

FastCloud deploy:
- GitHub Variables: `PORT` (8080), `APP_NAME` (burnsdb)
- GitHub Secrets: `SUPABASE_SERVICE_ROLE_KEY` (required), optionally `OPENAI_API_KEY`
- Push to `main` to trigger deploy

Auth policy (important):
- Service‑role only for server↔Supabase. No ANON keys anywhere.
- The server always includes `Authorization: Bearer <service_role>` and `apikey: <service_role>` when calling Supabase Edge.

Edge Function ping (prod):
- `curl -i --location --request POST "https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search" \`
  `--header "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \`
  `--header "Content-Type: application/json" \`
  `--data '{"query":"connectivity smoke test","k":3}'`

Local vs prod sanity:
- Local (when serving functions): `http://localhost:54321/functions/v1/advanced_semantic_search`
- Start with: `supabase start && supabase functions serve advanced_semantic_search`
- Use service‑role only for testing from this server.

Healthcheck script:
- New `bdb_healthcheck.sh` verifies Edge reachability, pgvector, ANN indexes, and FTS prerequisites. Edit placeholders and run: `bash bdb_healthcheck.sh`.

SQL: hybrid + TAR scaffolding:
- `sql/hybrid_prereqs.sql` — pgvector HNSW + FTS indexes (GIN/tsvector)
- `sql/hybrid_search_example.sql` — normalized hybrid scoring CTE (BM25 × cosine)
- `sql/tar_schema.sql` — active‑learning/TAR tables (labels, models, predictions, audit)
- Apply via `scripts/apply_sql.sh` (requires `psql` and DB env vars).

Supabase logs (connectivity reference):
- 200 OK invocation confirms function availability:
  - POST | 200 | https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search
  - sb-request-id: 01993b10-2473-7d76-8444-aa0a931cc3c4; region: us-west-1; runtime: deno v2.1.x

Notes:
- Server exports `app` (ASGI). Do not self-run in cloud; workflows launch `uvicorn server:app`.
- Outbound `ALLOWLIST_HOSTS` restricts HTTP calls; configure as needed.

# Burns-Database MCP Server

This repo hosts the BurnsDB MCP server (FastMCP over HTTP) that proxies a single Supabase Edge Function (`advanced_semantic_search`) for legal discovery search.

- Remote MCP endpoint (after deploy): `https://BurnsDB.fastmcp.app/mcp`
- Recommended client: `npx -y mcp-remote@0.1.29 https://BurnsDB.fastmcp.app/mcp`

## Quick start (local)
- `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`
- `uvicorn server:app --host 0.0.0.0 --port 8080`

## FastCloud deploy
- **GitHub Variables:** `PORT` (8080), `APP_NAME` (burnsdb)
- **GitHub Secrets:** `SUPABASE_SERVICE_ROLE_KEY` (required), optionally `OPENAI_API_KEY`
- Push to `main` to trigger deploy

## Auth policy (important)
- Service-role only for server↔Supabase. No ANON keys anywhere.
- The server always includes `Authorization: Bearer <service_role>` and `apikey: <service_role>` when calling Supabase Edge.

## Edge Function ping (prod)
```bash
curl -i --location --request POST "https://nqkzqcsqfvpquticvwmk.supabase.co/functions/v1/advanced_semantic_search" \
  --header "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
  --header "Content-Type: application/json" \
  --data '{"query":"connectivity smoke test","k":3}'

