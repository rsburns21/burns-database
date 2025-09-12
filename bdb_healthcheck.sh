#!/usr/bin/env bash
set -euo pipefail

### ---------- CONFIG ----------
# Supabase project + keys
PROJECT_REF="<YOUR_PROJECT_REF>"                               # e.g., abcdefghijklmnop
FUNCTION_NAME="advanced_semantic_search"
# Server-side testing: use service role
SERVICE_ROLE_KEY="<YOUR_SERVICE_ROLE_KEY>"

# DB connection (psql)
PGHOST="<YOUR_DB_HOST>"        # e.g., db.<region>.supabase.co
PGPORT="${PGPORT:-5432}"
PGDATABASE="<YOUR_DB_NAME>"
PGUSER="<YOUR_DB_USER>"
PGPASSWORD="<YOUR_DB_PASSWORD>"

# Basic query to semantic-search
SMOKE_QUERY="connectivity smoke test"
K=3

printf "\n== BDB Healthcheck ==\n"

### ---------- 0) Resolve URLs ----------
REMOTE_FUNC_URL="https://${PROJECT_REF}.supabase.co/functions/v1/${FUNCTION_NAME}"
LOCAL_FUNC_URL="http://localhost:54321/functions/v1/${FUNCTION_NAME}"

printf "\n[0] Derived URLs:\n  Remote: %s\n  Local : %s\n" "$REMOTE_FUNC_URL" "$LOCAL_FUNC_URL"

### ---------- 1) Edge Function connectivity ----------
printf "\n[1] Edge Function (REMOTE) ping...\n"
set +e
HTTP_CODE=$(curl -s -o /tmp/bdb_func_remote.json -w "%{http_code}" \
  -X POST "$REMOTE_FUNC_URL" \
  -H "Authorization: Bearer ${SERVICE_ROLE_KEY}" \
  -H "Content-Type: application/json" \
  --data "{\"query\":\"${SMOKE_QUERY}\",\"k\":${K}}")
set -e

if [[ "$HTTP_CODE" == "200" ]]; then
  echo "  ✅ Remote function reachable (200)."
else
  echo "  ⚠️ Remote function responded with HTTP ${HTTP_CODE}."
  echo "     If you recently switched to publishable/secret keys, deploy with --no-verify-jwt or pass a valid JWT."
fi

printf "\n[1b] Edge Function (LOCAL) ping (if serving locally)...\n"
set +e
HTTP_CODE_LOCAL=$(curl -s -o /tmp/bdb_func_local.json -w "%{http_code}" \
  -X POST "$LOCAL_FUNC_URL" \
  -H "Authorization: Bearer ${SERVICE_ROLE_KEY}" \
  -H "Content-Type: application/json" \
  --data "{\"query\":\"local smoke test\",\"k\":${K}}")
set -e
if [[ "$HTTP_CODE_LOCAL" == "200" ]]; then
  echo "  ✅ Local function reachable (200)."
else
  echo "  ℹ️  Local not serving or invalid key (HTTP ${HTTP_CODE_LOCAL}). Start with: supabase start && supabase functions serve"
fi

### ---------- 2) Database + pgvector health ----------
export PGPASSWORD
echo -e "\n[2] Database checks (pgvector, schema, counts)..."
psql "host=${PGHOST} port=${PGPORT} dbname=${PGDATABASE} user=${PGUSER} sslmode=require" <<'SQL'
-- Versions and extensions
SELECT current_setting('server_version') AS postgres_version;
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector');  -- pgvector present?

-- Expect vector_embeddings table & column
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name='vector_embeddings';

-- Embedding dims & cardinality (if using typed 'vector')
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='vector_embeddings' AND column_name='embedding') THEN
    RAISE NOTICE 'vector_embeddings count -> %', (SELECT COUNT(*) FROM vector_embeddings);
  END IF;
END$$;

-- Join integrity (adjust table/column names if needed)
-- Expect: exhibits(exhibit_id) <- chunks(exhibit_id, chunk_id) <- vector_embeddings(chunk_id)
WITH j AS (
  SELECT COUNT(*) AS n
  FROM vector_embeddings ve
  JOIN chunks c ON c.chunk_id = ve.chunk_id
  JOIN exhibits e ON e.exhibit_id = c.exhibit_id
)
SELECT 'join_integrity' AS check, n FROM j;
SQL

### ---------- 3) ANN index presence ----------
echo -e "\n[3] ANN indexes on embeddings..."
psql "host=${PGHOST} port=${PGPORT} dbname=${PGDATABASE} user=${PGUSER} sslmode=require" <<'SQL'
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename='vector_embeddings';

-- Example create (commented):
-- CREATE INDEX IF NOT EXISTS ve_hnsw_cos ON vector_embeddings USING hnsw (embedding vector_cosine_ops);
-- CREATE INDEX IF NOT EXISTS ve_hnsw_ip  ON vector_embeddings USING hnsw (embedding vector_ip_ops);
SQL

### ---------- 4) Full-text (hybrid pre-req) ----------
echo -e "\n[4] Full-text indexes present?"
psql "host=${PGHOST} port=${PGPORT} dbname=${PGDATABASE} user=${PGUSER} sslmode=require" <<'SQL'
-- Expect a tsvector column or expression index on chunks/exhibits text
-- Example discovery:
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('chunks','exhibits')
  AND indexdef ILIKE '%USING gin%tsvector%';
SQL

echo -e "\n[Done] Review any ⚠️ notes above.\n"

