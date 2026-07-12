from __future__ import annotations

from statistics import mean, median

from app.analyses.asr.metric_models import (
    ClassMetric,
    DisfluencyMetrics,
    FileMetrics,
    TimestampMetric,
    TimestampMetrics,
)
from app.asr.alignment import AlignedWord, AlignmentOperation, align_words, word_error_counts
from app.asr.normalization import is_partial_token, normalized_tokens
from app.asr.transcript import (
    JsonScalar,
    ReferenceTranscript,
    RuntimeStats,
    TranscriptionResult,
    Word,
)

FILLER_CLASSES: dict[str, set[str]] = {
    "um": {"um", "umm", "uhm"},
    "uh": {"uh", "er", "eh"},
    "mhm_mm": {"mhm", "mm", "mm-hmm", "mmhm"},
    "hmm": {"hmm", "hm"},
    "uh_huh": {"uh-huh", "uhuh"},
}


def evaluate_file(reference: ReferenceTranscript, prediction: TranscriptionResult) -> FileMetrics:
    alignment = align_words(reference.words, prediction.words)
    runtime = RuntimeStats(
        audio_duration_seconds=prediction.audio_duration_seconds,
        processing_time_seconds=prediction.processing_time_seconds,
        model_loading_time_seconds=prediction.model_loading_time_seconds,
        inference_time_seconds=prediction.inference_time_seconds,
        real_time_factor=prediction.real_time_factor,
        peak_gpu_memory_mb=prediction.peak_gpu_memory_mb,
    )
    return FileMetrics(
        model_name=prediction.model_name,
        audio_path=prediction.audio_path,
        word_error_counts=word_error_counts(alignment),
        filler_metrics=filler_metrics(alignment),
        disfluency_metrics=disfluency_metrics(reference.words, prediction.words),
        timestamp_metrics=timestamp_metrics(alignment),
        runtime=runtime,
        largest_timestamp_errors=largest_timestamp_errors(alignment, limit=25),
    )


def filler_metrics(alignment: list[AlignedWord]) -> dict[str, ClassMetric]:
    metrics: dict[str, ClassMetric] = {}
    for class_name, members in FILLER_CLASSES.items():
        metrics[class_name] = class_metric_for_members(alignment, members)
    all_members = set().union(*FILLER_CLASSES.values())
    metrics["filler_any"] = class_metric_for_members(alignment, all_members)
    return metrics


def class_metric_for_members(alignment: list[AlignedWord], members: set[str]) -> ClassMetric:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for item in alignment:
        reference_is_member = item.reference_token in members
        prediction_is_member = item.prediction_token in members
        if reference_is_member and prediction_is_member:
            true_positives += 1
        elif reference_is_member:
            false_negatives += 1
        elif prediction_is_member:
            false_positives += 1
    precision = ratio(true_positives, true_positives + false_positives)
    recall = ratio(true_positives, true_positives + false_negatives)
    f1 = ratio(2.0 * precision * recall, precision + recall)
    return ClassMetric(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def disfluency_metrics(
    reference_words: tuple[Word, ...],
    prediction_words: tuple[Word, ...],
) -> DisfluencyMetrics:
    reference_tokens = normalized_tokens(reference_words)
    prediction_tokens = normalized_tokens(prediction_words)
    reference_repetitions = immediate_repetition_events(reference_tokens)
    predicted_repetitions = immediate_repetition_events(prediction_tokens)
    reference_partials = partial_events(reference_tokens)
    predicted_partials = partial_events(prediction_tokens)
    recalled_repetitions = count_recalled_events(reference_repetitions, predicted_repetitions)
    recalled_partials = count_recalled_events(reference_partials, predicted_partials)
    return DisfluencyMetrics(
        repetition_reference_events=len(reference_repetitions),
        repetition_predicted_events=len(predicted_repetitions),
        repetition_recalled_events=recalled_repetitions,
        repetition_recall=ratio(recalled_repetitions, len(reference_repetitions)),
        partial_reference_events=len(reference_partials),
        partial_predicted_events=len(predicted_partials),
        partial_recalled_events=recalled_partials,
        partial_recall=ratio(recalled_partials, len(reference_partials)),
    )


def immediate_repetition_events(tokens: list[str]) -> list[str]:
    events: list[str] = []
    for token_index in range(1, len(tokens)):
        if tokens[token_index] == tokens[token_index - 1]:
            events.append(tokens[token_index])
    return events


def partial_events(tokens: list[str]) -> list[str]:
    return [token for token in tokens if is_partial_token(token)]


def count_recalled_events(reference_events: list[str], predicted_events: list[str]) -> int:
    remaining_events = list(predicted_events)
    recalled_events = 0
    for event in reference_events:
        if event in remaining_events:
            remaining_events.remove(event)
            recalled_events += 1
    return recalled_events


def timestamp_metrics(alignment: list[AlignedWord]) -> TimestampMetrics:
    start_differences: list[float] = []
    end_differences: list[float] = []
    for item in alignment:
        if item.operation != AlignmentOperation.EQUAL:
            continue
        assert item.reference is not None
        assert item.prediction is not None
        if item.reference.start_seconds is not None and item.prediction.start_seconds is not None:
            start_differences.append(item.prediction.start_seconds - item.reference.start_seconds)
        if item.reference.end_seconds is not None and item.prediction.end_seconds is not None:
            end_differences.append(item.prediction.end_seconds - item.reference.end_seconds)
    return TimestampMetrics(
        start=timestamp_metric_from_differences(start_differences),
        end=timestamp_metric_from_differences(end_differences),
    )


def timestamp_metric_from_differences(differences: list[float]) -> TimestampMetric:
    if not differences:
        return TimestampMetric(
            count=0,
            median_absolute_error=None,
            mean_absolute_error=None,
            p90_absolute_error=None,
            within_50ms=None,
            within_100ms=None,
            within_200ms=None,
            greater_than_500ms=None,
            median_bias=None,
        )
    absolute_errors = [abs(difference) for difference in differences]
    return TimestampMetric(
        count=len(differences),
        median_absolute_error=median(absolute_errors),
        mean_absolute_error=mean(absolute_errors),
        p90_absolute_error=percentile(absolute_errors, 0.90),
        within_50ms=ratio(count_within(absolute_errors, 0.050), len(absolute_errors)),
        within_100ms=ratio(count_within(absolute_errors, 0.100), len(absolute_errors)),
        within_200ms=ratio(count_within(absolute_errors, 0.200), len(absolute_errors)),
        greater_than_500ms=ratio(
            sum(1 for error in absolute_errors if error > 0.500), len(absolute_errors)
        ),
        median_bias=median(differences),
    )


def largest_timestamp_errors(
    alignment: list[AlignedWord], limit: int
) -> list[dict[str, JsonScalar]]:
    rows: list[dict[str, JsonScalar]] = []
    for item in alignment:
        if item.operation != AlignmentOperation.EQUAL:
            continue
        assert item.reference is not None
        assert item.prediction is not None
        start_error = optional_difference(
            item.prediction.start_seconds, item.reference.start_seconds
        )
        end_error = optional_difference(item.prediction.end_seconds, item.reference.end_seconds)
        if start_error is None and end_error is None:
            max_error = 0.0
        else:
            max_error = max(abs(error) for error in (start_error, end_error) if error is not None)
        rows.append(
            {
                "reference": item.reference.text,
                "prediction": item.prediction.text,
                "ref_start": item.reference.start_seconds,
                "pred_start": item.prediction.start_seconds,
                "start_error": start_error,
                "ref_end": item.reference.end_seconds,
                "pred_end": item.prediction.end_seconds,
                "end_error": end_error,
                "max_abs_error": max_error,
            }
        )
    return sorted(rows, key=lambda row: numeric_json(row["max_abs_error"]), reverse=True)[:limit]


def optional_difference(predicted: float | None, reference: float | None) -> float | None:
    if predicted is None or reference is None:
        return None
    return predicted - reference


def percentile(values: list[float], quantile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def count_within(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value <= threshold)


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def numeric_json(value: JsonScalar) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
