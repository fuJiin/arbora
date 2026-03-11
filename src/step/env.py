"""Load environment variables from .env file."""

import os
from pathlib import Path


def load_hf_token() -> None:
    """Load HF_TOKEN from .env file if not already in environment."""
    if os.environ.get("HF_TOKEN"):
        return
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_file = parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key == "HF_TOKEN":
                        os.environ["HF_TOKEN"] = value
                        return
            break


load_hf_token()
