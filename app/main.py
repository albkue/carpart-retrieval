import os

# Disable OneDNN/MKL on Windows before any paddle/paddleocr imports
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""ML Search Service - Main Application Entry Point."""
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.endpoints import router
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME}...")
    logger.info(f"Main API URL: {settings.MAIN_API_URL}")
    logger.info(f"YOLO Model: {settings.YOLO_MODEL}")
    logger.info(f"CLIP Model: {settings.CLIP_MODEL}")
    logger.info(f"GPU Enabled: {settings.USE_GPU}")

    if settings.API_KEY:
        logger.info("API key authentication is ENABLED.")
    else:
        logger.warning(
            "API_KEY is not set — every endpoint is OPEN and unauthenticated. "
            "Set API_KEY in .env for any deployment reachable beyond localhost."
        )


    # Pre-load core models only; OCR is lazy-loaded on first request to save RAM
    try:
        logger.info("Pre-loading models...")
        from .api.endpoints import components

        _ = components.detector
        logger.info(f"YOLO model loaded: {settings.YOLO_MODEL}")

        _ = components.embedder.model
        _ = components.embedder.processor
        logger.info(f"CLIP model loaded: {settings.CLIP_MODEL}")

        # PaddleOCR (~3-4 GB) is intentionally NOT pre-loaded here.
        # It loads on the first search request instead.
        logger.info("OCR will load on first request (lazy) to reduce startup RAM.")

    except Exception as e:
        logger.warning(f"Could not pre-load models: {e}")
        logger.info("Models will be loaded on first request.")
    
    logger.info(f"{settings.SERVICE_NAME} is ready!")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.SERVICE_NAME}...")


app = FastAPI(
    title="ML Search Service",
    description="Machine Learning service for image-based product search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
# credentials=True is incompatible with wildcard origins per the CORS spec
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
_allow_credentials = "*" not in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins else ["*"],
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting — sliding-window counter per client IP.
# Limits the expensive /search-by-image endpoint to 20 req/min.
_rate_counts: dict = defaultdict(list)
_RATE_LIMIT = 20
_RATE_WINDOW = 60  # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/v1/search-by-image":
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - _RATE_WINDOW
        _rate_counts[client_ip] = [t for t in _rate_counts[client_ip] if t > cutoff]
        if len(_rate_counts[client_ip]) >= _RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {_RATE_LIMIT} requests per minute."}
            )
        _rate_counts[client_ip].append(now)
    return await call_next(request)

# API key middleware — rejects requests missing the correct key
# Skips check if API_KEY is not configured (open mode for local dev)
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    # /health and / are always public
    if settings.API_KEY and request.url.path not in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
        key = request.headers.get("X-API-Key")
        if key != settings.API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    """Root endpoint - service health check."""
    return {
        "service": settings.SERVICE_NAME,
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
