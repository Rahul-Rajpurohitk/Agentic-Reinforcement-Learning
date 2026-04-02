"""OpenEnv-compliant server entry point.

This module provides the main() function required by openenv for
multi-mode deployment (uv run server, python -m, Docker).
It delegates to the FastAPI app defined in src/agentic_rl/server/app.py.
"""

import uvicorn


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Fish Farm OpenEnv server."""
    uvicorn.run(
        "agentic_rl.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
