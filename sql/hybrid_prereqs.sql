-- pgvector + FTS prerequisites for hybrid search
CREATE EXTENSION IF NOT EXISTS vector;

-- HNSW index (cosine) for fast ANN (adjust ops as needed)
-- Requires pgvector >= 0.6.0 on managed Supabase
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE tablename = 'vector_embeddings' AND indexname = 've_hnsw_cos'
  ) THEN
    EXECUTE 'CREATE INDEX ve_hnsw_cos ON vector_embeddings USING hnsw (embedding vector_cosine_ops)';
  END IF;
END $$;

-- FTS: materialized tsvector and GIN index (example on chunks)
ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS tsv tsvector;
UPDATE chunks SET tsv = to_tsvector('english', coalesce(text, '')) WHERE tsv IS NULL;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin(tsv);

