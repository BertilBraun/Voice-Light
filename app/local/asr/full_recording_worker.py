from __future__ import annotations

import argparse
from dataclasses import dataclass
from uuid import UUID

from app.local.asr.client import HttpRemoteAsrClient
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.full_recording_service import run_full_recording_asr_batch
from app.local.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL


@dataclass(frozen=True)
class WorkerArguments:
    batch_id: UUID
    maximum_tracks: int | None
    recover_running: bool
    retry_failed: bool


def main() -> None:
    arguments = parse_arguments()
    repository = FullRecordingAsrRepository(database_url=DATABASE_URL)
    result = run_full_recording_asr_batch(
        batch_id=arguments.batch_id,
        store=repository,
        remote_client_factory=remote_asr_client,
        maximum_tracks=arguments.maximum_tracks,
        recover_running=arguments.recover_running,
        retry_failed=arguments.retry_failed,
    )
    print(result.model_dump_json(indent=2))


def parse_arguments() -> WorkerArguments:
    parser = argparse.ArgumentParser(
        description="Resume a queued full-recording ASR batch on the configured GPU service."
    )
    parser.add_argument("batch_id", type=UUID)
    parser.add_argument(
        "--maximum-tracks",
        type=int,
        default=None,
        help="Process at most this many tracks before returning.",
    )
    parser.add_argument(
        "--recover-running",
        action="store_true",
        help="Explicitly return interrupted running items to the pending queue.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly return failed items to the pending queue.",
    )
    namespace = parser.parse_args()
    return WorkerArguments(
        batch_id=namespace.batch_id,
        maximum_tracks=namespace.maximum_tracks,
        recover_running=namespace.recover_running,
        retry_failed=namespace.retry_failed,
    )


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )


if __name__ == "__main__":
    main()
