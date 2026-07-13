from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, WebSocket


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

    async def authorize_websocket(self, websocket: WebSocket) -> bool:
        authorization = websocket.headers.get("authorization")
        if authorization is not None and secrets.compare_digest(
            authorization,
            self.expected_authorization,
        ):
            return True
        await websocket.close(code=1008, reason="Invalid compute bearer token.")
        return False
