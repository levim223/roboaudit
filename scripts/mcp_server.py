#!/usr/bin/env python3
"""Standalone entrypoint for RoboAudit MCP Server (Lesson 6).

Ensures the project root is in sys.path so the server runs reliably
under any environment (Kiro IDE, VS Code, CI, or command line).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboaudit.server import main

if __name__ == "__main__":
    main()
