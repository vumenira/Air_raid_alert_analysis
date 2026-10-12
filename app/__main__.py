"""Usage: uv run python -m app [--port 8000]"""
import argparse

import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="127.0.0.1")
args = ap.parse_args()
print(f"Dashboard: http://{args.host}:{args.port}")
uvicorn.run("app.server:app", host=args.host, port=args.port)
