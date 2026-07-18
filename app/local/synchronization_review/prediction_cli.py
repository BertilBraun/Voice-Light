from __future__ import annotations

from app.local.config import DATABASE_URL
from app.local.data.sessions import SpeakerName, session_audio_path
from app.local.synchronization_review.models import AlignmentEstimateOrigin
from app.local.synchronization_review.optimized_alignment import (
    optimized_alignment_candidate,
)
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.local.synchronization_review.service import (
    ESTIMATOR_VERSION,
    synchronization_candidates,
)


def main() -> None:
    repository = SynchronizationReviewRepository(database_url=DATABASE_URL)
    response = synchronization_candidates(repository=repository)
    candidates = tuple(
        optimized_alignment_candidate(
            candidate=candidate,
            speaker1_path=session_audio_path(
                identifier=candidate.external_id,
                speaker_name=SpeakerName.SPEAKER1,
            ),
            speaker2_path=session_audio_path(
                identifier=candidate.external_id,
                speaker_name=SpeakerName.SPEAKER2,
            ),
        )
        if candidate.alignment_estimate_origin is AlignmentEstimateOrigin.PREDICTED
        else candidate
        for candidate in response.candidates
    )
    saved_count = repository.save_unreviewed_predictions(
        candidates=candidates,
        estimator_version=ESTIMATOR_VERSION,
    )
    print(f"Saved {saved_count} unreviewed synchronization predictions with {ESTIMATOR_VERSION}.")


if __name__ == "__main__":
    main()
