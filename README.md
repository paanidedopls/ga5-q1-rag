# GA5 Q1 — Render free-tier optimized native Python deployment

This version is configured for Render's 512 MB / low-CPU free web service. It does not require Docker or local WSL.

## Why this version is optimized

- Uses `BAAI/bge-small-en-v1.5`, a compact 384-dimensional dense embedding model supported by FastEmbed (about 67 MB model files).
- Restricts the ONNX Runtime session to one thread (`FASTEMBED_THREADS=1`).
- Embeds the small corpus in batches of 16 to limit startup working memory.
- Restricts concurrent API processing to four requests and uses one Uvicorn worker.
- Caps common BLAS/OpenMP thread pools at one thread.
- Uses an instruction prefix for BGE query embeddings.

Model size reference: https://qdrant.github.io/fastembed/examples/Supported_Models/
FastEmbed threads parameter: https://github.com/qdrant/fastembed/blob/main/fastembed/text/text_embedding.py

## GitHub repository files

Upload the contents of this folder to the repository root:

- `app.py`
- `knowledge_base.json`
- `requirements.txt`
- `.python-version`
- `render.yaml`

The `tests` folder is optional for deployment. Do not upload a Dockerfile.

## Deploy

1. Push these files to the connected GitHub repository.
2. In Render, use **Manual Deploy → Deploy latest commit** (or trigger a new deploy after connecting the repo).
3. Keep the Free plan and allow the service to build and start.
4. The first boot downloads the BGE-small model and embeds all 50 passages. This may take a few minutes on Render's free CPU.
5. Check `/health`. It must return `status: ok`, `passages: 50`, and `embedding_backend: fastembed`.
6. Test `POST /query` and confirm the response includes `answer`, `sources`, `scores`, and `confidence`.

## Environment variables

`render.yaml` declares all recommended memory/thread limits. Confirm these are applied in the Render service's Environment tab if you created the service manually rather than via Blueprint.

## Important free-tier caveat

Render Free Web Services have 512 MB RAM, spin down after 15 minutes of inactivity, and can take about one minute to wake. Keep the service warm immediately before the exam probe. A cold start can violate an 8-second first-request requirement.

## Endpoints

- `GET /health`
- `POST /query`
- `GET /docs`

Use the service base URL in the exam submission, without `/query`.
