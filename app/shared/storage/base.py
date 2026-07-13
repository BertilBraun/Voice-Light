from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO, Protocol


class StorageBackend(Protocol):
    def walk(self, root: str) -> Iterator[str]: ...

    def open(self, path: str) -> BinaryIO: ...

    def read(self, path: str) -> bytes: ...

    def access_uri(self, path: str) -> str: ...
