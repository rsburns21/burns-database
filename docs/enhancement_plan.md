# BurnsDB Enhancement Plan

This document compiles recommendations derived from external research to improve the Burns Database (BDB) MCP server and discovery workflow. The items below highlight future directions and do not represent completed features.

## 1. Advanced Query Parsing and UI Filters
- Support fielded queries using `field:value` syntax and expose common filters (claims, issues, individuals, entities).
- Document Boolean operators and phrase search; consider wildcard and proximity support.

## 2. Vector Search Optimisation
- Index embeddings with approximate nearest‑neighbour algorithms (e.g. pgvector IVF, Faiss) to keep semantic search fast at scale.
- Cache frequent query embeddings to reduce repeated computation.

## 3. Document Ingestion and OCR
- Add an ingestion pipeline that OCRs PDFs/images and automatically chunks, embeds and indexes new exhibits.
- Track ingestion status and surface errors (e.g. missing files).

## 4. Machine Learning‑Assisted Review
- Use existing relevance scores as training data for active learning or predictive coding.
- Suggest additional relevant documents by nearest‑neighbour similarity or trained classifiers.

## 5. Multi‑User Collaboration
- Implement user accounts with roles and audit logs.
- Allow tagging and annotation of exhibits per user to support coordinated review.

## 6. Reporting and Export
- Provide exports (CSV/PDF) of filtered exhibit sets and generate case statistics reports.
- Support standard eDiscovery formats for interchange with other platforms.

## 7. Continuous Index Updates
- Automatically refresh text and vector indexes when exhibits or metadata change.
- Ensure semantic search uses the latest embeddings.

## 8. Documentation and Training
- Create user‑facing guides describing search syntax, tagging workflows and defensibility considerations.
- Include troubleshooting steps for common deployment issues (network, schema mismatch, etc.).

These enhancements aim to align BDB with best practices observed in high‑value litigation discovery platforms.
