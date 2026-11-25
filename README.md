# GA5 Q1 — Render native Python deployment

This package deploys the FastAPI RAG service to Render as a native Python Web Service. It does **not** require Docker or local WSL.

## Files to put in a GitHub repository

- `app.py`
- `knowledge_base.json`
- `requirements.txt`
- `.python-version`
- `render.yaml` (optional Blueprint configuration)

Do not add a Dockerfile to the Render repository. Render should use the native Python runtime.

## Deploy

1. Create a GitHub repository and upload the files above to its root.
2. In Render, choose **New → Blueprint** and select that GitHub repository, or choose **New → Web Service** and connect it.
3. For a Web Service, select **Python** runtime, Free instance, build command `pip install -r requirements.txt`, and start command `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`.
4. Set environment variables: `EMBEDDING_BACKEND=fastembed`, `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, `MAX_PARALLEL_REQUESTS=16`.
5. Wait for the deploy to become Live. The first startup downloads the embedding model and indexes the 50 passages.
6. Check `/health`. It must return `status: ok`, `passages: 50`, and `embedding_backend: fastembed`.
7. Submit the base HTTPS URL only, without `/query`.

## Important free-tier caveat

Render Free Web Services spin down after 15 minutes without inbound traffic. Waking a sleeping service can take about a minute, which may violate the assignment's 8-second response limit on the first request. Open `/health` immediately before triggering the exam probe; keep the service warm until all rounds finish. If the exam probe may occur more than 15 minutes later, a local server plus Cloudflare Tunnel is the safer no-cost alternative.

## API

`POST /query` with JSON body:

```json
{"question":"What quorum does Raft require?","top_k":3}
```

`GET /health` reports readiness and the active embedding backend.
