"""Launch the V5 read-only HTTP API + dashboard.

Default bind: 127.0.0.1:8765 (loopback only). Override with
``--host`` / ``--port``. The server reads run artefacts from
``cfg.run_root`` and is safe to run in parallel with the scanner /
exec loops since it never writes to that tree.

Examples:
    .venv/Scripts/python scripts/run_api.py
    .venv/Scripts/python scripts/run_api.py --port 9000 --reload
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.server import create_app  # noqa: E402
from core.config import load_config  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_host = os.getenv("API_HOST", "127.0.0.1")
    default_port = int(os.getenv("API_PORT", "8765"))
    p = argparse.ArgumentParser(description="V5 dashboard / API")
    p.add_argument("--host", default=default_host)
    p.add_argument("--port", type=int, default=default_port)
    p.add_argument("--log-level", default="info")
    p.add_argument("--reload", action="store_true",
                   help="dev-mode autoreload (requires uvicorn[standard])")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write("uvicorn is required: pip install uvicorn\n")
        return 2

    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
