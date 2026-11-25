# GA5 Q1 — Render deployment (no local Docker)

This project serves the supplied 50-passage knowledge base through a FastAPI `POST /query` endpoint.

## Runtime

- Hosting: Render native Python Web Service (Free plan)
- API: FastAPI + Uvicorn, one worker
- Dense embeddings: FastEmbed with `BAAI/bge-small-en-v1.5`
- Retrieval: dense cosine similarity + BM25 + entity-aware / attribute-aware evidence ranking
- Answering: extractive and evidence-grounded; no external LLM is required
- Security: strips common prompt-injection attempts from user queries and never executes retrieved passage text
- CORS: `Access-Control-Allow-Origin: *`

The local regression test can use TF-IDF when model downloads are unavailable; Render is configured to require FastEmbed (`EMBEDDING_BACKEND=fastembed`).

## Files to keep in the GitHub repository root

- `app.py`
- `knowledge_base.json`
- `requirements.txt`
- `.python-version`
- `render.yaml`
- `README.md`

The `tests` directory is optional for deployment but useful for local verification. Do not add a Dockerfile or install Docker/WSL locally.

## Deploy/update on Render

1. Push the repository files to GitHub.
2. In Render, open the existing `ga5-q1-rag` service.
3. Choose **Manual Deploy → Deploy latest commit**.
4. Wait until the service reports `Live`.
5. Open `/health` and verify:
   - `status` is `ok`
   - `passages` is `50`
   - `embedding_backend` is `fastembed`

## Endpoints

- `GET /`
- `GET /health`
- `GET /docs`
- `POST /query`

Example request:

```json
{
  "question": "What quorum does Raft require to commit a log entry?",
  "top_k": 3
}
```

Use the public service base URL in the assignment form, without `/query`.

## Local tests

Install dependencies from `requirements.txt`, then run:

```bash
EMBEDDING_BACKEND=tfidf python tests/smoke_test.py
```

This exercises 44 answerable questions, 5 multi-hop questions, 6 abstention questions, 2 injection questions, CORS, schema validation, and 20-way concurrent requests. The local fallback test does not substitute for checking the production FastEmbed backend on Render.

## Free Render caveat

Render Free Web Services can spin down after inactivity. Open `/health` shortly before the exam probe so the service is warm. Cold-start timing remains a hosting limitation relative to the exam's 8-second per-request target.
