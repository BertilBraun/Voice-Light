from __future__ import annotations

from typing import TypeVar

import httpx

from app.shared.base_model import FrozenBaseModel
from app.shared.compute_api import (
    DatasetAsrRequest,
    DatasetAsrResponse,
    DatasetLanguageRequest,
    DatasetLanguageResponse,
    DatasetQualityRequest,
    DatasetQualityResponse,
)

ResponseModel = TypeVar("ResponseModel", bound=FrozenBaseModel)
DATASET_ANALYSIS_TIMEOUT_SECONDS = 21_600.0


class HttpDatasetAnalysisClient:
    def __init__(self, compute_base_url: str, api_key: str) -> None:
        if not compute_base_url:
            raise ValueError("VOICE_LIGHT_COMPUTE_URL is required for manifest ingestion.")
        if not api_key:
            raise ValueError("VOICE_LIGHT_COMPUTE_TOKEN is required for manifest ingestion.")
        self.compute_base_url = compute_base_url.rstrip("/")
        self.api_key = api_key

    def assess_language(self, request: DatasetLanguageRequest) -> DatasetLanguageResponse:
        return self.post(
            endpoint="/v1/dataset:language",
            request=request,
            response_type=DatasetLanguageResponse,
        )

    def transcribe(self, request: DatasetAsrRequest) -> DatasetAsrResponse:
        return self.post(
            endpoint="/v1/dataset:asr",
            request=request,
            response_type=DatasetAsrResponse,
        )

    def analyze_quality(self, request: DatasetQualityRequest) -> DatasetQualityResponse:
        return self.post(
            endpoint="/v1/dataset:quality",
            request=request,
            response_type=DatasetQualityResponse,
        )

    def post(
        self,
        endpoint: str,
        request: FrozenBaseModel,
        response_type: type[ResponseModel],
    ) -> ResponseModel:
        response = httpx.post(
            f"{self.compute_base_url}{endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=request.model_dump(mode="json"),
            timeout=DATASET_ANALYSIS_TIMEOUT_SECONDS,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ValueError(
                f"Remote dataset analysis failed with HTTP {response.status_code}: {response.text}"
            ) from error
        return response_type.model_validate_json(response.text)
