# ASR Analysis

The ASR analysis page compares word-level transcripts and timing across remote ASR models and
local post-processing modes. Open:

```text
http://127.0.0.1:8000/analyses/asr
```

An analysis processes the first three minutes of one speaker track. The page shows synchronized
transcripts, stacked timing waveforms, and optional reference-based metrics. Adjacent words less
than 300 ms apart are displayed as one speech block.

## Processing

Raw Parakeet, faster-Whisper, Canary, and Nemotron inference runs on the compute backend. Results are cached in
Postgres by audio content hash and model ID, so repeated analysis does not invoke a cached model
again.

Local derived modes do not call the compute backend:

- The Parakeet + Canary union retains Parakeet words and adds non-overlapping Canary coverage.
- The merged consensus progressively aligns all four model transcripts.
- Crosstalk-filtered variants use a 30-second rolling active-speech power baseline and the other
  speaker channel. Words 12 dB below the baseline are removed; words below the baseline are also
  removed when the other channel dominates them by at least 6 dB.
- Filtered union and consensus modes filter each source transcript before combining them.

Raw cached transcripts remain unchanged, allowing raw and filtered results to be compared in the
same analysis.

## Configuration

The local app requires:

```text
VOICE_LIGHT_DATABASE_URL
VOICE_LIGHT_COMPUTE_URL
VOICE_LIGHT_COMPUTE_TOKEN
```

Reusable local APIs are `POST /api/asr/transcriptions`, `GET /api/asr/models`, and
`POST /api/asr/analyze`. Start the compute model server on the rented machine with:

```bash
bash deployment/compute/bootstrap.sh
bash deployment/compute/start.sh
```
