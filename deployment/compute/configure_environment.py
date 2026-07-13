from __future__ import annotations

import argparse
import secrets
from collections.abc import Sequence
from pathlib import Path


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository_root", type=Path)
    options = parser.parse_args(arguments)
    repository_root = options.repository_root.resolve()
    environment_path = repository_root / ".env.compute"
    if environment_path.exists():
        raise ValueError(f"Refusing to overwrite {environment_path}.")
    compute_token = secrets.token_urlsafe(48)
    environment_path.write_text(
        "\n".join(
            (
                f"VOICE_LIGHT_COMPUTE_TOKEN={compute_token}",
                "VOICE_LIGHT_COMPUTE_HOST=0.0.0.0",
                "VOICE_LIGHT_COMPUTE_PORT=8000",
                "VOICE_LIGHT_COMPUTE_LOG_DIR=logs/compute",
                f"HF_HOME={repository_root / '.cache/compute/huggingface'}",
                f"HF_HUB_CACHE={repository_root / '.cache/compute/huggingface/hub'}",
                f"TORCH_HOME={repository_root / '.cache/compute/torch'}",
                f"XDG_CACHE_HOME={repository_root / '.cache/compute'}",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment_path.chmod(0o600)


if __name__ == "__main__":
    main()
