#!/usr/bin/env python3
"""
Launch the WebUI FastAPI server for mtdata.

Usage:
  python webui.py            # serves on http://127.0.0.1:8000
  python -m uvicorn src.mtdata.core.web_api:app --reload

This does not replace the MCP server in server.py. Both can run separately.
"""

import os
import sys

def main():
    # Ensure environment variables are loaded from .env if present
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv()
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
    except Exception:
        pass

    # Auto-detect MT5 server offset if not configured
    if "MT5_TIME_OFFSET_MINUTES" not in os.environ and "MT5_SERVER_TZ" not in os.environ:
        try:
            # Add src to path to import utils
            sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
            from mtdata.utils.mt5 import estimate_server_offset
            
            print("Detecting MT5 server timezone offset...", file=sys.stderr)
            offset_sec = estimate_server_offset()
            if offset_sec != 0:
                offset_min = int(offset_sec / 60)
                print(f"Detected offset: {offset_min} minutes. Applying to session.", file=sys.stderr)
                os.environ["MT5_TIME_OFFSET_MINUTES"] = str(offset_min)
            else:
                print("Could not detect offset (market closed or connection failed). Defaulting to 0.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Offset detection failed: {e}", file=sys.stderr)

    try:
        import uvicorn
    except Exception as e:
        print("ERROR: uvicorn not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        raise

    app_path = "src.mtdata.core.web_api:app"
    host = os.environ.get("MTDATA_WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("MTDATA_WEBUI_PORT", "8000"))
    reload = os.environ.get("MTDATA_WEBUI_RELOAD", "0") == "1"

    uvicorn.run(app_path, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

