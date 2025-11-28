"""
Serve built frontend assets (default: ../frontend/dist) on Jetson.
Run: python serve_static.py --dir ../frontend/dist --port 4173
"""

import argparse
import functools
import http.server
import socketserver
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve static frontend files (Jetson combined deploy)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Directory to serve (default: ../frontend/dist)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=4173, help="Port to serve on (default: 4173)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not args.dir.exists():
        raise SystemExit(f"Directory not found: {args.dir}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.dir))

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {args.dir} at http://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    main()
