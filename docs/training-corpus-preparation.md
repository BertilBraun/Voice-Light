# Training Corpus Preparation And Validation

## Decision

Prepare and validate the existing datasets before completing the trainer or processing more meeting
data. Keep every audio file already stored in the private Hugging Face repository. The training
corpus is a versioned selection that references eligible recordings; it is not a destructive cleanup
of the source archive.

Store each complete speaker track once as lossless FLAC. Do not store a separate audio file for each
20-second training window. Parquet rows are the materialized training examples: they reference a
source FLAC and a time interval and contain the aligned frame-level inputs, targets, and masks.

## Current source status

| Dataset | Local source | Current analysis status | Required work |
| --- | --- | --- | --- |
| `dataset_1` (LUEL) | 165 paired WAV recordings | Canonical quality analysis exists; 33 recordings passed the previous 0.95 export | Reuse the existing Hub FLAC, regenerate rich metadata, and revalidate the selected recordings |
| `dataset_2` (MagicHub) | 8 paired WAV recordings, about 2.77 hours | Canonical analysis and review are complete; all 8 recordings are accepted | Publish the staged lossless FLAC and rich metadata |
| `dataset_3` (Mundo TurnBench dev) | 38 paired FLAC recordings, about 7.31 hours | Canonical analysis and review are complete; 37 recordings are accepted and silent `sample_020` is excluded | Preserve the human evidence and publish only the accepted recording metadata/windows |
| `dataset_4` (meetings) | S3 source; 191 paired recordings already archived on Hub | Canonical quality analysis exists; 29 reviewed recordings remain after excluding `meeting-112` | Reference only the accepted recordings for the pilot; do not process or upload more meetings |

MagicHub and TurnBench were not rejected by the 0.95 filter. They were never candidates for the
previous export because they did not have the same completed quality analysis.

TurnBench's supplied human annotations must remain distinct from model-generated transcripts. The
existing source-materialization helper copies one human annotator into both Parakeet and Canary
transcript slots; using that route to compute the filtering score would create artificial agreement.
For a comparable quality score, run the real Parakeet and Canary analysis and use the human tracks
as independent inspection and label evidence.

## Artifact layers

There are three related artifacts with different responsibilities:

1. The source archive contains all authorized recordings already uploaded or prepared locally.
2. Recording metadata contains durable facts and analysis for one complete two-track conversation.
3. Parquet shards contain many materialized training windows and reference the source archive.

JSON is appropriate for the first two manifests because it is human-readable and easy to inspect.
Parquet is appropriate for the window table because it stores thousands of typed rows and fixed-size
label arrays compactly and can be read a shard at a time.

## Canonical layout

Preserve the existing top-level dataset identifiers so uploaded audio does not need to be moved or
duplicated:

```text
voice-light-audio/
  corpus.json
  dataset_1/
    dataset.json
    samples/
      <recording-id>/
        speaker_1.flac
        speaker_2.flac
        metadata.json
  dataset_2/
    dataset.json
    samples/
      <recording-id>/
        speaker_1.flac
        speaker_2.flac
        metadata.json
  dataset_3/
    dataset.json
    samples/
      <recording-id>/
        speaker_1.flac
        speaker_2.flac
        metadata.json
        source_annotations.json
  dataset_4/
    dataset.json
    <source-relative-recording-path>/
      <existing-speaker-tracks>.flac
      metadata.json
  training/
    train/shard-00000.parquet
    validation/shard-00000.parquet
    test/shard-00000.parquet
```

The local staging tree must use the same relative paths as Hugging Face. Reuse existing files by
reference or hard link when possible. A private build manifest maps each database-addressed source
path and source-file hash to its staged Hub path and FLAC-file hash. A WAV and its lossless FLAC have
different encoded hashes even when their decoded PCM is identical; those identities must never be
substituted for one another. Copying the same audio into several local trees is unnecessary.

## Dataset metadata

Each `dataset.json` records:

- stable dataset ID and display name;
- source name, source revision, acquisition date, and authorized source URL;
- license, attribution, privacy, and redistribution constraints;
- preparation version and audio normalization policy;
- recording count, duration, language, and known limitations.

Do not put credentials, private source mappings, or restricted participant identifiers in the Hub
metadata. Keep those in the existing restricted local provenance records.

## Recording metadata

Each `metadata.json` is the authoritative record for one complete conversation and contains:

- schema version, dataset ID, recording ID, and original external ID where permitted;
- both FLAC paths, SHA-256 hashes, duration, sample rate, channel count, and synchronization facts;
- quality score, component scores, flags, metric version, and analysis status;
- the canonical two-speaker annotation and its version;
- timestamped transcripts and model/source provenance for each transcript;
- per-speaker VAD intervals;
- usable and unusable conversation regions with reasons;
- source-annotation references and manual-review state when available.

The full annotations belong here rather than being copied into every 20-second row. Training rows
contain only the derived frame targets needed for efficient training.

## Parquet training row

One Parquet row represents one oriented 20-second example. It is an actual training sample, but its
audio is a reference rather than embedded bytes. Required fields are:

- schema version and deterministic window ID;
- dataset ID, recording ID, split, target speaker side, and partner speaker side;
- target FLAC path and optional partner FLAC reference for auditing;
- crop start and end seconds;
- recording quality score and sampling/event category;
- 250 values at 80 ms for the known assistant-speaking input;
- 250 HOLD/YIELD targets and validity/reliability masks;
- five independent interaction-event targets and masks;
- four future-user-activity targets and masks.

Separating shards by split makes accidental test loading difficult. The root `corpus.json` records
the shard paths, row counts, hashes, source and analysis versions, split policy, seed, and hours per
dataset and split.

This format needs a custom PyTorch dataset because a generic Parquet iterator cannot infer that it
must download a FLAC, seek to the crop, decode it, align 250 target frames, and construct tensors.
The current Hugging Face dataset class already implements most of that contract. Its audio reader
currently decodes the complete recording for every crop; replace that with accurate random seeking
and persistent file caching before a real training run.

## Step 1: Build the local corpus

1. Inventory the private Hub repository without deleting, moving, or overwriting audio blindly.
2. Produce a canonical mapping from each database recording and track hash to its existing or planned
   Hub-relative FLAC path.
3. Create the local staging layout using the same relative paths.
4. Reuse the existing `dataset_1` and `dataset_4` Hub FLAC files. Do not re-encode or re-upload them
   when their decoded audio and hashes are already valid.
5. Convert the 16 MagicHub mono WAV tracks to lossless FLAC and retain both the source WAV hashes
   and derived FLAC hashes in restricted local provenance. Validate filenames and audio metadata for
   the complete conversion, but perform the expensive exact decoded-PCM comparison on only one or
   two deterministic representative tracks.
6. Reuse TurnBench's 76 source FLAC tracks and preserve all three human annotation tracks.
7. Run the standard language, real Parakeet, real Canary, VAD, conversation-annotation, region, and
   quality pipeline for all MagicHub and TurnBench recordings.
8. Inspect representative and suspicious samples, including the lowest-scoring recordings, in the
   dataset dashboard and training sample lab.
9. Select recordings using the current quality threshold only after reviewing the score distribution
   and failure reasons. Record the exact threshold and metric version in `corpus.json`.
10. Generate rich recording metadata and deterministic training windows for every accepted
    recording.
11. Assign conversation-disjoint train, validation, and test splits, stratified by source, duration,
    and event coverage. Keep the test assignment locked after inspection is complete.

## Step 2: Validate before upload

### Structural validation

- Every manifest entry resolves to exactly one metadata file and two FLAC tracks.
- Every Parquet row resolves to a selected recording and stays within its duration.
- IDs and Hub-relative paths are unique and platform-independent.
- No conversation appears in more than one split.

### Audio validation

- Every file is valid lossless FLAC, mono, and decodable at the declared sample rate.
- Paired tracks share the intended timeline after any declared padding or synchronization repair.
- Declared durations and SHA-256 hashes match the staged files.
- One or two deterministic WAV-to-FLAC samples have matching decoded PCM; the complete conversion
  has valid filenames, FLAC headers, hashes, durations, sample rates, and channel counts.

### Annotation and target validation

- Transcript words, VAD intervals, source events, and conversation regions stay within bounds.
- Annotation, quality, and region-analysis versions are current and complete.
- Every supervised frame has a valid HOLD/YIELD target or an explicit mask.
- Auxiliary and future-activity masks match their available evidence.
- The partner track is used to derive labels but is never passed to the acoustic backbone.

### Corpus-level validation

- Report source hours, accepted hours, effective supervised hours, windows, masks, and unique event
  counts by dataset and split.
- Report score distributions and all rejection reasons, rather than only the final threshold count.
- Detect duplicate audio hashes and overlapping windows that cross split boundaries.
- Cap concentration from any one conversation and report source balance.

### Manual validation

- Listen to a deterministic stratified sample from every dataset and split.
- Inspect examples from every sampling category and every major rejection reason.
- Compare TurnBench-derived targets with its independent human annotations.
- Freeze a review report listing accepted problems, required fixes, and final approval.

For the first plumbing pilot, the accepted manual gate is narrower than the full target above: 12
dataset-stratified recordings were reviewed and all final review items passed after excluding
TurnBench `sample_020`, meetings `meeting-112`, and the unusable interval in Dataset 1
`sample_246`. The pilot did not independently sample every split, training category, or rejection
reason. Preserve that limitation in the publication metadata and do not describe this run as the
full manual-validation protocol.

### Upload validation

- Generate an upload plan showing new, unchanged, and changed paths and total bytes.
- Upload metadata, manifests, shards, MagicHub FLAC, and TurnBench FLAC only after the plan passes.
- Do not delete existing Hub audio as part of corpus publication.
- After upload, instantiate the loader against the pinned Hub revision, sample rows from every split,
  verify hashes, and perform one model forward pass.

## Completion criteria

Steps 1 and 2 are complete when the corpus can be rebuilt deterministically, its validation report
has no unexplained failures, every accepted row loads from the local mirror and pinned Hub revision,
and the exact train/validation/test inventory is immutable for the first experiment.
