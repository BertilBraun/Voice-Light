from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    def walk(self, root: str) -> Iterator[str]: ...

    def open(self, path: str) -> object: ...

    def read(self, path: str) -> bytes: ...

    def access_uri(self, path: str) -> str: ...

    def local_path(self, path: str) -> Path: ...
