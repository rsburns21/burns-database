#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PGHOST:-}" || -z "${PGDATABASE:-}" || -z "${PGUSER:-}" || -z "${PGPASSWORD:-}" ]]; then
  echo "Set PGHOST, PGDATABASE, PGUSER, PGPASSWORD (and optionally PGPORT)." >&2
  exit 1
fi

psql "host=${PGHOST} port=${PGPORT:-5432} dbname=${PGDATABASE} user=${PGUSER} sslmode=require" -v ON_ERROR_STOP=1 -f sql/hybrid_prereqs.sql
psql "host=${PGHOST} port=${PGPORT:-5432} dbname=${PGDATABASE} user=${PGUSER} sslmode=require" -v ON_ERROR_STOP=1 -f sql/tar_schema.sql

echo "Done. You can run hybrid examples by passing parameters to sql/hybrid_search_example.sql."

