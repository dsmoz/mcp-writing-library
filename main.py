#!/usr/bin/env python3
"""
MCP Writing Library Server

Stores and semantically searches exemplary writing passages and terminology
for use by the copywriter agent and editorial-review skill.

Collections:
    writing_passages — exemplary paragraphs by doc type, language, domain
    writing_terms    — terminology dictionary (preferred vs avoid)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# kbase-core: vendored in vendor/ (Railway) or installed locally from libraries/
vendor_path = project_root / 'vendor'
if vendor_path.exists():
    sys.path.insert(0, str(vendor_path))
else:
    kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
    if kbase_core_path.exists():
        sys.path.insert(0, str(kbase_core_path))

# Load environment variables
env_file = os.getenv('ENV_FILE', str(project_root / '.env'))
if Path(env_file).exists():
    load_dotenv(env_file)
else:
    load_dotenv(project_root / '.env.example')
    print(f"⚠️  {env_file} not found, using .env.example", file=sys.stderr)

# Initialise Sentry (no-op if SENTRY_DSN is not set)
_sentry_dsn = os.getenv("SENTRY_DSN")
if _sentry_dsn:
    import sentry_sdk

    _SDK_NOISE_LOGGERS = frozenset({
        "mcp.server.streamable_http",
        "mcp.server.lowlevel.server",
    })
    _SDK_NOISE_MESSAGES = ("ClientDisconnect", "Received exception from stream:")

    def _before_send(event, hint):
        logger = event.get("logger", "")
        if logger in _SDK_NOISE_LOGGERS:
            msg = event.get("message") or ""
            exc_values = event.get("exception", {}).get("values", [])
            exc_type = exc_values[0].get("type", "") if exc_values else ""
            if any(m in msg for m in _SDK_NOISE_MESSAGES) or exc_type == "ClientDisconnect":
                return None

        # Drop benign anyio stream-teardown noise (check all values in the exception chain)
        exc_values = event.get("exception", {}).get("values", [])
        if any(v.get("type") in ("ClosedResourceError", "BrokenResourceError") for v in exc_values):
            return None

        return event

    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0.1,
        environment=os.getenv("RAILWAY_ENVIRONMENT", "development"),
        release=f"mcp-writing-library@{os.getenv('RAILWAY_GIT_COMMIT_SHA', 'unknown')}",
        before_send=_before_send,
    )
    print("✅ Sentry error tracking enabled", file=sys.stderr)


def main():
    """Main entry point for MCP server."""
    transport = os.getenv("TRANSPORT", "stdio")
    try:
        from src.server import mcp
        print(f"🚀 Starting MCP Writing Library Server (transport={transport})...", file=sys.stderr)
        print(f"📍 Project: {project_root}", file=sys.stderr)
        if transport == "http":
            port = int(os.getenv("PORT", "8000"))
            host = os.getenv("HOST", "0.0.0.0")
            print(f"🌐 HTTP transport on {host}:{port}", file=sys.stderr)
            import uvicorn
            from src.server import BearerAuthMiddleware
            app = BearerAuthMiddleware(mcp.streamable_http_app())
            uvicorn.run(app, host=host, port=port)
        else:
            mcp.run()
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
