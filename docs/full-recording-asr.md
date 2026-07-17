# Full-recording ASR preparation

Full-recording ASR is deliberately separate from the three-minute analysis cache. A transcript
is linked to one `sample_tracks` row and one source-file SHA-256, so synchronization code never
has to infer its scope from a filename.

## Queue a batch

Run database migrations, then queue a batch without starting GPU work:

```powershell
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/asr/full-recording/batches' `
  -ContentType 'application/json' `
  -Body '{"idempotency_key":"luel-parakeet-full-v1","dataset_id":"DATASET_UUID","sample_scope":"quality_analyzed","models":["parakeet_tdt_0_6b_v3"]}'
```

Reusing the same idempotency key with the same canonical model set returns the existing batch.
Reusing it with different settings fails. Model selection is explicit: start with Parakeet for
long recordings because its implementation chunks audio, while Canary currently does not.

## Run or resume on the GPU day

Configure `VOICE_LIGHT_COMPUTE_URL` and `VOICE_LIGHT_COMPUTE_TOKEN`, then run:

```powershell
.\.venv\Scripts\python.exe -m app.local.asr.full_recording_worker BATCH_UUID
```

The worker claims one track at a time and records failures per track. Completed transcript rows
are reused by source hash, preventing duplicate GPU calls. An interrupted item remains `running`
so another worker cannot silently steal active work. After confirming the old worker is gone,
recover it explicitly:

```powershell
.\.venv\Scripts\python.exe -m app.local.asr.full_recording_worker BATCH_UUID --recover-running
```

Retry failed tracks explicitly with `--retry-failed`. Use `--maximum-tracks N` for a smoke run.
The status endpoint is `GET /api/asr/full-recording/batches/BATCH_UUID`.

Each source is encoded as temporary mono 16 kHz Ogg Opus at 32 kbps and streamed to the compute
service as a binary multipart upload. The compute service verifies the encoded file hash and
decodes it to canonical mono 16 kHz FLAC before model inference. The legacy JSON/base64 endpoint
remains available temporarily for workers started before this transport was introduced. Source
and prepared hashes, durations, channel count, and sample rate are validated before persistence.

## Ingestion lifecycle

`quality-conversation-full-parakeet-v5` registers samples and tracks before quality processing.
Tracks without an exact source-hash Parakeet transcript pair are added idempotently to the
dataset's ingestion ASR batch, and the ingestion job finishes as `waiting_for_asr`. Re-running
ingestion resumes from the same quality version and uses the cached transcript rows without
another GPU call.

Once both transcripts exist, ingestion builds the full-duration annotation and quality result.
Reviewed speaker-2 offsets are applied to audio and transcript timestamps; samples without a
review use an explicit zero shift. Each v5 quality row stores both transcript IDs, the applied
shift, and whether it was reviewed or `unreviewed_zero`. This intentionally mixed v5 timeline is
the input for full-recording offset estimation. After the remaining offsets are reviewed, v6 can
reuse the same transcript cache and recompute only alignment-dependent annotations and quality.
