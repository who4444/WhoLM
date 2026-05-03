
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .routes import router
from .rate_limiter import RateLimitMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    app = FastAPI(
        title="WhoLM API",
        description="AI-powered Video and Document Q&A System",
        version="1.0.0"
    )

    # CORS: configurable origins (comma-separated env var)
    # In production, set CORS_ORIGINS="https://yourdomain.com"
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://localhost:3000"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Admin-Key",
            "X-Request-ID",
            "Accept",
        ],
    )

    # Global rate limit: 60 requests/min per client IP
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

    app.include_router(router)

    return app


app = create_app()