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

