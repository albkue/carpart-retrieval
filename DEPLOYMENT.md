# Deployment Guide

## Architecture

```
GitHub                →  source code (version control)
Hugging Face Hub      →  best.pt model weights (auto-downloaded on startup)
Senior's Server       →  runs Docker container (compute + public URL)
```

---

## Prerequisites

- Docker installed on the server
- `MAIN_API_URL` of the backend API (ask your senior)
- A generated API key (see Step 2)

---

## Step 1 — Model Weights (Already Done)

Model is hosted at `hf://albkue/car-parts-yolov8/best.pt`.
The container downloads it automatically on first startup. Nothing to do here.

---

## Step 2 — Generate an API Key

Run this once to generate a secure key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Save the output — you need to give it to your senior and set it in the server env vars.

---

## Step 3 — Environment Variables

Set these on the server (or in a `.env` file on the server):

```
SERVICE_NAME=ml-search-service
SERVICE_PORT=8001
YOLO_MODEL=hf://albkue/car-parts-yolov8/best.pt
CLIP_MODEL=openai/clip-vit-large-patch14
YOLO_CONFIDENCE_THRESHOLD=0.5
OCR_CONFIDENCE_THRESHOLD=0.6
FAISS_INDEX_PATH=/app/data/faiss_index
USE_GPU=false
CORS_ORIGINS=*
MAIN_API_URL=https://backend-url-from-senior    ← fill this in
API_KEY=your-generated-key-here                 ← fill this in
```

---

## Step 4 — Run on the Server

**Option A — docker-compose (recommended)**

```bash
git clone https://github.com/services-management/image-search
cd ml_service
# Create .env from the variables above
docker-compose up -d
```

**Option B — plain Docker**

```bash
docker build -t car-parts-ml .
docker run -d \
  --env-file .env \
  -p 8001:8001 \
  -v faiss_data:/app/data \
  car-parts-ml
```

---

## Step 5 — Build the FAISS Search Index

Run this once after the container is up (rebuilds the product search index from the backend):

```bash
curl -X POST http://server-url:8001/api/v1/rebuild-index \
  -H "X-API-Key: your-generated-key-here"
```

---

## Step 6 — Verify

```bash
# Health check (no API key needed)
curl http://server-url:8001/health

# Test image search (API key required)
curl -X POST http://server-url:8001/api/v1/search-by-image \
  -H "X-API-Key: your-generated-key-here" \
  -F "file=@test_image.jpg"
```

---

## API Endpoints

All endpoints under `/api/v1/` require the `X-API-Key` header.
`/health` and `/` are public.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check service is live |
| POST | `/api/v1/search-by-image` | Search products by image |
| POST | `/api/v1/rebuild-index` | Rebuild FAISS index from backend |
| POST | `/api/v1/index-product` | Add single product to index |
| GET | `/api/v1/index/stats` | FAISS index statistics |
| GET | `/api/v1/brands` | List known brands |
| GET | `/api/v1/categories` | List supported part categories |
| GET | `/api/v1/health/catalog` | Check backend connection |

---

## Connecting to the Backend

The ML service connects to the backend via `MAIN_API_URL`. The backend must expose:

| Endpoint | Used for |
|----------|----------|
| `GET /category/` | Resolve part category names to IDs |
| `GET /product/search` | Search products by brand/category/name |
| `GET /product/{id}` | Get product details |
| `GET /product/` | Get all products (for index rebuild) |
| `GET /health` | Backend health check |

Your senior's backend must also send the API key on every request to the ML service:
```
X-API-Key: your-generated-key-here
```

---

## Updating the Model

When you retrain and get a new `best.pt`:

```bash
# Upload new weights to HF Hub
huggingface-cli login
huggingface-cli upload albkue/car-parts-yolov8 runs/detect/car_parts_v1/weights/best.pt best.pt

# Restart the container to pick up new weights
docker-compose restart

# Rebuild FAISS index if product catalog changed
curl -X POST http://server-url:8001/api/v1/rebuild-index \
  -H "X-API-Key: your-generated-key-here"
```

---

## Notes

- Never commit `.env` — it contains real secrets
- `API_KEY` is optional for local dev — leave it empty to skip auth
- Cold start downloads ~2.4 GB of models (YOLO + CLIP + BGE-M3 + PaddleOCR) — allow 5–10 min on first run
- `data/` volume persists the FAISS index across container restarts
