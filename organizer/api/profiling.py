"""
Profiling middleware for FastAPI application.

This module provides middleware for profiling API requests using pyinstrument.
Profile results can be viewed by adding ?profile=true to any request.
"""

import logging
import os
from pathlib import Path

from fastapi import Request, Response
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ProfilingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to profile API requests.

    When enabled, this middleware will profile requests and save the results
    to HTML files that can be viewed in a browser.

    Usage:
        Add ?profile=true to any request URL to enable profiling for that request.
        Profile results are saved to the profiles/ directory.
    """

    def __init__(self, app, enabled: bool = True, output_dir: str = "profiles"):
        super().__init__(app)
        self.enabled = enabled
        self.output_dir = Path(output_dir)

        if self.enabled:
            self.output_dir.mkdir(exist_ok=True)
            logger.info(
                f"Profiling middleware enabled. Output directory: {self.output_dir}"
            )

    async def dispatch(self, request: Request, call_next):
        # Check if profiling is requested via query parameter
        should_profile = request.query_params.get("profile") == "true"

        if not self.enabled or not should_profile:
            return await call_next(request)

        # Start profiling
        profiler = Profiler(interval=0.0001)  # Sample every 0.1ms
        profiler.start()

        try:
            # Process the request
            response: Response = await call_next(request)
        finally:
            # Stop profiling
            profiler.stop()

            # Generate profile output
            path = request.url.path.replace("/", "_")
            if path == "_":
                path = "_root"

            output_file = self.output_dir / f"{path}.html"

            # Save HTML profile
            with open(output_file, "w") as f:
                f.write(profiler.output_html())

            logger.info(f"Profile saved to {output_file}")

            # Add header to response indicating where profile was saved
            response.headers["X-Profile-Output"] = str(output_file)

        return response


def is_profiling_enabled() -> bool:
    """
    Check if profiling should be enabled based on environment variables.

    Returns:
        True if ENABLE_PROFILING env var is set to 'true' or '1', False otherwise.
    """
    return os.getenv("ENABLE_PROFILING", "false").lower() in ("true", "1")


class ProfilerContext:
    """
    Context manager for profiling a block of code.

    Usage:
        with ProfilerContext("task_name") as profiler:
            # code to profile
            pass
    """

    def __init__(self, name: str, output_dir: str = "profiles", enabled: bool = False):
        self.name = name
        self.output_dir = Path(output_dir)
        self.enabled = is_profiling_enabled()
        self.profiler = None

    def __enter__(self):
        if not self.enabled:
            return None

        self.output_dir.mkdir(exist_ok=True)
        # Use async_mode="disabled" to allow multiple profilers to coexist
        # This is fine for background tasks where the main work is synchronous
        self.profiler = Profiler(interval=0.0001, async_mode="disabled")
        self.profiler.start()
        logger.info(f"Started profiling: {self.name}")
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.profiler:
            return

        self.profiler.stop()

        # Generate output filename
        safe_name = self.name.replace("/", "_").replace(" ", "_")
        output_file = self.output_dir / f"{safe_name}.html"

        # Save HTML profile
        with open(output_file, "w") as f:
            f.write(self.profiler.output_html())

        logger.info(f"Profile saved to {output_file}")
