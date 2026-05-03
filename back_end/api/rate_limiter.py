import time
import logging
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple sliding-window rate limiter. Not distributed — single process only."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.window = 60.0  # seconds
        self._clients: dict[str, list[float]] = defaultdict(list)

    def _clean(self, client_id: str, now: float) -> None:
        cutoff = now - self.window
        self._clients[client_id] = [t for t in self._clients[client_id] if t > cutoff]
        if not self._clients[client_id]:
            del self._clients[client_id]

    def is_allowed(self, client_id: str) -> bool:
        now = time.monotonic()
        self._clean(client_id, now)
        if client_id not in self._clients:
            return True
        return len(self._clients[client_id]) < self.rpm

    def hit(self, client_id: str) -> None:
        self._clients[client_id].append(time.monotonic())

    @property
    def tracked_clients(self) -> int:
        return len(self._clients)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limiting per client IP."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute=requests_per_minute)

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        if not self.limiter.is_allowed(client):
            logger.warning(f"Rate limit hit for {client}")
            raise HTTPException(status_code=429, detail="Too many requests. Slow down.")

        self.limiter.hit(client)
        return await call_next(request)


# Stricter limiter for expensive AI endpoints
chat_limiter = RateLimiter(requests_per_minute=20)


def check_chat_rate_limit(request: Request) -> None:
    """Call at start of /chat handler for stricter per-endpoint limiting."""
    client = request.client.host if request.client else "unknown"
    if not chat_limiter.is_allowed(client):
        logger.warning(f"Chat rate limit hit for {client}")
        raise HTTPException(status_code=429, detail="Too many chat requests. Slow down.")
    chat_limiter.hit(client)
