from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, Response, WebSocket
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from app.compute.asr.service import transcribe_request
from app.compute.auth import BearerTokenAuthorizer
from app.compute.config import ComputeSettings
from app.compute.quality.router import analyze_uploaded_quality
from app.compute.runtime import ComputeRuntime
from app.compute.telemetry import RequestIdScope, configure_logging, gpu_memory
from app.compute.voice.router import run_voice_session
from app.shared.asr import RemoteAsrRequest, RemoteAsrResponse
from app.shared.compute_api import HealthStatus, LivenessResponse, ReadinessResponse

logger = logging.getLogger(__name__)


def create_compute_app(settings: ComputeSettings) -> FastAPI:
    configure_logging(settings.log_directory)
    runtime = ComputeRuntime()
    authorizer = BearerTokenAuthorizer(settings.token)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        del application
        logger.info("compute server startup initiated")
        runtime.start_loading()
        try:
            yield
        finally:
            logger.info("compute server graceful shutdown initiated")
            await runtime.shutdown()

    application = FastAPI(title="Voice Light Compute", lifespan=lifespan)

    @application.middleware("http")
    async def request_telemetry(
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        started = time.perf_counter()
        with RequestIdScope(request_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "%s %s completed with %d in %.3f seconds",
                request.method,
                request.url.path,
                response.status_code,
                time.perf_counter() - started,
            )
            return response

    @application.get("/health/live")
    def liveness() -> LivenessResponse:
        return LivenessResponse()

    @application.get(
        "/health/ready",
        dependencies=[Depends(authorizer.authorize_http)],
    )
    def readiness(response: Response) -> ReadinessResponse:
        if not runtime.ready:
            response.status_code = 503
        return ReadinessResponse(
            status=HealthStatus.READY if runtime.ready else HealthStatus.NOT_READY,
            stages=runtime.stages(),
            gpu_memory=gpu_memory(),
        )

    @application.post(
        "/v1/asr:batch",
        dependencies=[Depends(authorizer.authorize_http)],
    )
    async def batch_asr(request: RemoteAsrRequest) -> RemoteAsrResponse:
        return await asyncio.to_thread(transcribe_request, runtime.batch_asr_models, request)

    application.post(
        "/v1/quality:analyze",
        dependencies=[Depends(authorizer.authorize_http)],
    )(analyze_uploaded_quality)

    @application.websocket("/v1/voice")
    async def voice(websocket: WebSocket) -> None:
        if not runtime.ready:
            await websocket.close(code=1013, reason="Compute models are not ready.")
            return
        request_id = websocket.headers.get("x-request-id") or str(uuid4())
        await run_voice_session(
            websocket=websocket,
            runtime=runtime,
            request_id=request_id,
        )

    return application


def create_app_from_environment() -> FastAPI:
    return create_compute_app(ComputeSettings.from_environment(os.environ))
