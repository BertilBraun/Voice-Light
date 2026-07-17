# Full-recording ASR ingestion

Full-recording ASR is a resumable part of dataset ingestion. There is no separate batch queue or
worker.

For each selected sample, ingestion:

1. registers both audio tracks and their metadata;
2. hashes each source file and looks up an exact track, hash, and model cache entry;
3. encodes only missing tracks as temporary mono 16 kHz Ogg Opus;
4. streams them to the compute service, where Parakeet processes 30-second chunks in batches;
5. deletes temporary transport files after each request;
6. applies the reviewed speaker-2 offset, or an explicit unreviewed zero offset;
7. builds the full-duration annotation and quality result; and
8. persists the result before incrementing ingestion progress for the sample.

Re-running the same quality version skips completed samples. If a sample is interrupted after one
speaker transcript is saved, the next run reuses that exact source-hash transcript and requests
only the missing speaker.

Start or resume ingestion from `/datasets/ingest`. The job list on that page is the single progress
surface for registration, ASR, annotation, and quality processing.
