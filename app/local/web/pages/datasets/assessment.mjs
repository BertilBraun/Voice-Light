export function sampleLanguageStatus(assessments) {
  const statuses = new Set(
    (assessments || []).map((assessment) => assessment.status),
  );
  if (statuses.has("non_english")) {
    return "Non-English — full analysis skipped";
  }
  if (statuses.has("failed") || statuses.has("inconclusive")) {
    return "Inconclusive — full analysis allowed";
  }
  if (statuses.size > 0) {
    return "English";
  }
  return "Not assessed";
}

export function silenceMaskSegments(
  conversation,
  durationSeconds,
  minimumSilenceSeconds,
) {
  if (!(durationSeconds > 0)) {
    return [];
  }
  const activity = [
    ...(conversation.speaker1?.speech_segments || []),
    ...(conversation.speaker2?.speech_segments || []),
  ]
    .map((segment) => ({
      start_seconds: Math.max(0, Number(segment.start_seconds)),
      end_seconds: Math.min(durationSeconds, Number(segment.end_seconds)),
    }))
    .filter((segment) => segment.end_seconds > segment.start_seconds)
    .sort((left, right) => left.start_seconds - right.start_seconds);

  const merged = [];
  for (const segment of activity) {
    const previous = merged.at(-1);
    if (previous && segment.start_seconds <= previous.end_seconds) {
      previous.end_seconds = Math.max(
        previous.end_seconds,
        segment.end_seconds,
      );
    } else {
      merged.push({ ...segment });
    }
  }

  const masks = [];
  let previousEndSeconds = 0;
  for (const segment of [
    ...merged,
    { start_seconds: durationSeconds, end_seconds: durationSeconds },
  ]) {
    const silenceSeconds = segment.start_seconds - previousEndSeconds;
    if (silenceSeconds >= minimumSilenceSeconds) {
      masks.push({
        start_seconds: previousEndSeconds,
        end_seconds: segment.start_seconds,
      });
    }
    previousEndSeconds = Math.max(
      previousEndSeconds,
      segment.end_seconds,
    );
  }
  return masks;
}
