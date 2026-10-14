"""Streamlit entry point.

Usage:
    streamlit run main.py

(For a quick terminal-only way to use/test the assistant without the UI,
see scripts/cli.py.)
"""

from app.ui.app import main

if __name__ == "__main__":
    main()
