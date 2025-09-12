-- Hybrid ranking (BM25/FTS × cosine similarity)
-- $1 = query text
-- $2 = query embedding (vector)

WITH
params AS (
  SELECT 
    plainto_tsquery('english', $1)      AS q_ts,
    $2::vector                          AS q_vec,
    0.5::float4                         AS w_lex,
    0.5::float4                         AS w_sem
),
candidates AS (
  -- Lexical candidates
  SELECT c.chunk_id, c.exhibit_id,
         ts_rank_cd(c.tsv, p.q_ts)                    AS lex_score,
         1 - (ve.embedding <=> p.q_vec)               AS sem_score
  FROM params p
  JOIN chunks c ON c.tsv @@ p.q_ts
  JOIN vector_embeddings ve ON ve.chunk_id = c.chunk_id

  UNION

  -- Semantic candidates (top-N by ANN)
  SELECT c.chunk_id, c.exhibit_id,
         ts_rank_cd(c.tsv, p.q_ts)                    AS lex_score,
         1 - (ve.embedding <=> p.q_vec)               AS sem_score
  FROM params p
  JOIN vector_embeddings ve ON true
  JOIN chunks c ON c.chunk_id = ve.chunk_id
  ORDER BY ve.embedding <=> p.q_vec
  LIMIT 500
),
norm AS (
  SELECT *,
         CASE WHEN max(lex_score) OVER() = min(lex_score) OVER() THEN 0
              ELSE (lex_score - min(lex_score) OVER()) / NULLIF(max(lex_score) OVER() - min(lex_score) OVER(),0)
         END AS lex_norm,
         CASE WHEN max(sem_score) OVER() = min(sem_score) OVER() THEN 0
              ELSE (sem_score - min(sem_score) OVER()) / NULLIF(max(sem_score) OVER() - min(sem_score) OVER(),0)
         END AS sem_norm
  FROM candidates
),
scored AS (
  SELECT exhibit_id, chunk_id,
         lex_norm, sem_norm,
         (p.w_lex * lex_norm + p.w_sem * sem_norm) AS hybrid_score
  FROM norm, params p
)
SELECT s.*
FROM scored s
JOIN exhibits e USING (exhibit_id)
ORDER BY hybrid_score DESC
LIMIT 50;

