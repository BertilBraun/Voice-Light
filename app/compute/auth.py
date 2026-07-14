from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException


class BearerTokenAuthorizer:
    def __init__(self, token: str) -> None:
        self.expected_authorization = f"Bearer {token}"

    def authorize_http(
        self,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if authorization is None or not secrets.compare_digest(
            authorization,
            self.expected_authorization,
        ):
            raise HTTPException(status_code=401, detail="Invalid compute bearer token.")
