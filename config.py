"""API configuration via environment variables.

Users should set the following environment variables before running experiments:

    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL=""
    export OPENAI_MODEL="gpt-4o"

Or create an api.json file in the project root (see api.json.example).
"""
import json, os


def load():
    """Load API configuration. Prefers api.json, falls back to env vars."""
    # Try api.json first
    for candidate in ["api.json", "../api.json"]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                data = json.load(f)
            active = data.get("active", data)
            return {
                "base_url": active["base_url"],
                "api_key": active["api_key"],
                "model": active.get("model", "gpt-4o"),
            }

    # Fall back to environment variables
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    }
