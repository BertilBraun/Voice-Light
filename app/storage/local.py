from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class LocalStorageBackend:
    def walk(self, root: str) -> Iterator[str]:
        root_path = Path(root).resolve()
        if not root_path.exists():
            raise ValueError(f"Storage root does not exist: {root}")
        for path in sorted(root_path.rglob("*")):
            if path.is_file():
                yield path.as_posix()

    def open(self, path: str) -> BinaryIO:
        return Path(path).open("rb")

    def read(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def access_uri(self, path: str) -> str:
        return str(Path(path).resolve())
