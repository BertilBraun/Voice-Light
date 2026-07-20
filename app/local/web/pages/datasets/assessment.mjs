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

const gamingTerms = new Set([
  "ammo",
  "boss",
  "damage",
  "enemy",
  "game",
  "gg",
  "gun",
  "heal",
  "hp",
  "kill",
  "level",
  "loot",
  "mana",
  "map",
  "match",
  "quest",
  "rank",
  "respawn",
  "round",
  "score",
  "spawn",
  "team",
]);

export function conversationStructureRegions(
  conversation,
  durationSeconds,
  regionSeconds = 30,
) {
  if (!(durationSeconds > 0) || !(regionSeconds > 0)) {
    return [];
  }
  const speaker1Segments = conversation.speaker1?.speech_segments || [];
  const speaker2Segments = conversation.speaker2?.speech_segments || [];
  const speaker1Targets = conversation.speaker1?.segment_targets || [];
  const speaker2Targets = conversation.speaker2?.segment_targets || [];
  const regions = [];
  for (
    let startSeconds = 0;
    startSeconds < durationSeconds;
    startSeconds += regionSeconds
  ) {
    const endSeconds = Math.min(
      durationSeconds,
      startSeconds + regionSeconds,
    );
    const speaker1SpeechSeconds = occupiedSeconds(
      speaker1Segments,
      startSeconds,
      endSeconds,
    );
    const speaker2SpeechSeconds = occupiedSeconds(
      speaker2Segments,
      startSeconds,
      endSeconds,
    );
    const targets = [
      ...speaker1Targets.map((target) => ({
        ...target,
        speaker: "speaker1",
      })),
      ...speaker2Targets.map((target) => ({
        ...target,
        speaker: "speaker2",
      })),
    ]
      .filter(
        (target) =>
          Number(target.end_seconds) > startSeconds &&
          Number(target.start_seconds) < endSeconds,
      )
      .sort(
        (left, right) =>
          Number(left.start_seconds) - Number(right.start_seconds),
      );
    const speakerTransitions = targets
      .slice(1)
      .filter((target, index) => target.speaker !== targets[index].speaker)
      .length;
    const words = targets
      .flatMap((target) => tokenize(target.text || ""));
    const matchedGamingTerms = [
      ...new Set(words.filter((word) => gamingTerms.has(word))),
    ].sort();
    const category = structureCategory({
      speaker1SpeechSeconds,
      speaker2SpeechSeconds,
      speakerTransitions,
      wordCount: words.length,
      gamingTermCount: matchedGamingTerms.length,
    });
    regions.push({
      start_seconds: startSeconds,
      end_seconds: endSeconds,
      category,
      speaker1_speech_seconds: speaker1SpeechSeconds,
      speaker2_speech_seconds: speaker2SpeechSeconds,
      speaker_transitions: speakerTransitions,
      gaming_terms: matchedGamingTerms,
    });
  }
  return regions;
}

function structureCategory({
  speaker1SpeechSeconds,
  speaker2SpeechSeconds,
  speakerTransitions,
  wordCount,
  gamingTermCount,
}) {
  const totalSpeechSeconds =
    speaker1SpeechSeconds + speaker2SpeechSeconds;
  if (totalSpeechSeconds < 1) {
    return "silence";
  }
  if (
    gamingTermCount >= 2 ||
    (gamingTermCount >= 1 && wordCount > 0 && gamingTermCount / wordCount >= 0.1)
  ) {
    return "gaming_like";
  }
  if (speaker1SpeechSeconds < 1 || speaker2SpeechSeconds < 1) {
    return "single_speaker";
  }
  if (speakerTransitions > 0) {
    return "alternating_dialogue";
  }
  return "two_speaker_unstructured";
}

function occupiedSeconds(segments, startSeconds, endSeconds) {
  const clipped = segments
    .map((segment) => ({
      start: Math.max(startSeconds, Number(segment.start_seconds)),
      end: Math.min(endSeconds, Number(segment.end_seconds)),
    }))
    .filter((segment) => segment.end > segment.start)
    .sort((left, right) => left.start - right.start);
  let occupied = 0;
  let mergedEnd = startSeconds;
  for (const segment of clipped) {
    if (segment.end <= mergedEnd) {
      continue;
    }
    occupied += segment.end - Math.max(segment.start, mergedEnd);
    mergedEnd = segment.end;
  }
  return occupied;
}

function tokenize(text) {
  return String(text)
    .toLowerCase()
    .match(/[a-z0-9]+/g) || [];
}
