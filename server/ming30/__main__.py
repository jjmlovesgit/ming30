"""``python -m ming30`` launches the adapter with uvicorn."""

from __future__ import annotations

import os

import uvicorn

from .app import create_app


def main() -> None:
    host = os.environ.get("MING30_HOST", "127.0.0.1")
    port = int(os.environ.get("MING30_PORT", "8070"))
    log_level = os.environ.get("MING30_LOG_LEVEL", "info")
    uvicorn.run(create_app(), host=host, port=port, log_level=log_level, access_log=False)


if __name__ == "__main__":
    main()
