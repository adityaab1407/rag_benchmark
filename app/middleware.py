"""
app/middleware.py
=================
FastAPI middleware for request tracing and observability.

Adds request_id to every request so logs can be correlated.
Every log line for a request will have [req-abc123] prefix.

Also tracks:
  - request start/end time
  - endpoint called
  - response status code
  - total wall time
"""

import uuid
import time
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from contextvars import ContextVar

# Context variable so request_id flows through async code
request_id_var: ContextVar[str] = ContextVar("request_id", default="no-request")


def get_request_id() -> str:
    return request_id_var.get()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Generates a unique request_id for every request
    2. Injects it into context so all log lines include it
    3. Logs request start, end, duration, status
    4. Adds X-Request-ID header to response
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate request ID
        req_id = str(uuid.uuid4())[:8]
        request_id_var.set(req_id)

        # Store on request state so route handlers can access it
        request.state.request_id = req_id

        start_time = time.time()

        logger.info(
            "[{}] → {} {}",
            req_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)
            duration = round(time.time() - start_time, 3)

            logger.info(
                "[{}] ← {} {} {}s",
                req_id,
                response.status_code,
                request.url.path,
                duration,
            )

            # Add request ID to response headers for client-side tracing
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Duration-Ms"] = str(int(duration * 1000))

            return response

        except Exception as e:
            duration = round(time.time() - start_time, 3)
            logger.error(
                "[{}] ✗ {} {} {}s — {}",
                req_id,
                request.method,
                request.url.path,
                duration,
                str(e)[:100],
            )
            raise