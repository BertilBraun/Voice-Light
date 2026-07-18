from __future__ import annotations

from app.local.config import DATABASE_URL
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.local.synchronization_review.service import (
    ESTIMATOR_VERSION,
    synchronization_candidates,
)


def main() -> None:
    repository = SynchronizationReviewRepository(database_url=DATABASE_URL)
    response = synchronization_candidates(repository=repository)
    saved_count = repository.save_unreviewed_predictions(
        candidates=response.candidates,
        estimator_version=ESTIMATOR_VERSION,
    )
    print(f"Saved {saved_count} unreviewed synchronization predictions with {ESTIMATOR_VERSION}.")


if __name__ == "__main__":
    main()
