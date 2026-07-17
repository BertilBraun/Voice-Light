# Synchronization offset evaluation

The synchronization estimator is evaluated against every manually reviewed offset before its
parameters are changed. Database reviews take precedence over the older static calibration list,
and the report keeps their provenance separate.

Run the reproducible evaluation from the repository root:

```powershell
.\.venv\Scripts\python.exe -m app.local.synchronization_review.evaluation_cli `
  > .\synchronization-offset-evaluation.json
```

The report includes:

- the current estimator baseline;
- tuned out-of-fold predictions from deterministic five-fold cross-validation;
- combined-review and database-only error metrics;
- absolute errors and predictions for every reviewed sample;
- the configuration selected on all labels for later application to unreviewed samples.

The all-label configuration is a deployment candidate, not an unbiased evaluation. Only the
out-of-fold metrics measure whether tuning generalizes beyond the labels used to select each
fold's parameters.

The ASR estimator is based on complementary turn-taking activity: it searches for a shift that
reduces simultaneous speech and joint silence. It is not a same-signal correlation estimator.

Current evidence has the `initial_180_seconds` scope. Full-recording ASR can produce the same
typed evidence records with the `full_recording` scope. After the full ASR batch finishes, run:

```powershell
.\.venv\Scripts\python.exe -m app.local.synchronization_review.evaluation_cli `
  --evidence-scope full_recording `
  --models parakeet_tdt_0_6b_v3 `
  > .\synchronization-offset-full-asr-evaluation.json
```

The full-recording adapter reads the dedicated transcript table through sample-track IDs. It
requires both speakers for every requested model, reports incomplete coverage, and produces
whole-recording evidence plus three-minute windows across each recording. The report's drift
summaries show each source's first and last window estimates, end-to-start change, and maximum
window spread. Evidence scopes cannot be mixed in one report, preventing an ambiguous comparison
with the older filename-based, truncated transcript cache.

## Acoustic baseline

A separate CPU-only diagnostic tests normalized log-energy onset correlation between the raw
channels:

```powershell
.\.venv\Scripts\python.exe -m app.local.synchronization_review.acoustic_evaluation_cli `
  > .\synchronization-offset-acoustic-evaluation.json
```

On the 83 reviews available during implementation, this baseline was rejected: combined MAE was
8.676 seconds, RMSE was 9.369 seconds, and only 2.4% of predictions were within one second. The
correlation peaks were generally weak and aliased by conversational activity, so acoustic
predictions are not consumed by the synchronization estimator.
