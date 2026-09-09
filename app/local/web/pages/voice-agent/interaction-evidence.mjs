export const INTERACTION_TIMELINE_DURATION_MS = 20000;

export function updateInteractionEvidence(evidence, message) {
  const previous = evidence.get(message.observed_audio_time_ms);
  const isModelObservation = message.turn_completion_probability !== null;
  evidence.set(message.observed_audio_time_ms, {
    audioTimeMs: message.observed_audio_time_ms,
    sileroSpeech: message.silero_speech,
    assistantAudible: message.assistant_audible,
    isModelObservation: isModelObservation || previous?.isModelObservation || false,
    predictionDisposition: isModelObservation
      ? message.prediction_disposition
      : previous?.predictionDisposition ?? null,
    turnCompletion: isModelObservation
      ? message.turn_completion_probability
      : previous?.turnCompletion ?? null,
    floorTake: isModelObservation
      ? message.floor_take_probability
      : previous?.floorTake ?? null,
    nonFloorFeedback: isModelObservation
      ? message.non_floor_feedback_probability
      : previous?.nonFloorFeedback ?? null,
  });

  const newestAudioTimeMs = Math.max(...evidence.keys());
  for (const audioTimeMs of evidence.keys()) {
    if (audioTimeMs < newestAudioTimeMs - INTERACTION_TIMELINE_DURATION_MS) {
      evidence.delete(audioTimeMs);
    }
  }
  return newestAudioTimeMs;
}

export function modelObservationSamples(points, field) {
  return points
    .filter((point) => point.isModelObservation && point[field] !== null)
    .map((point) => ({
      audioTimeMs: point.audioTimeMs,
      probability: point[field],
      disposition: point.predictionDisposition,
    }));
}

export function summarizeModelObservationCadence(points) {
  const latestAudioTimeMs = points.at(-1)?.audioTimeMs;
  const observations = points.filter((point) => point.isModelObservation);
  const latestObservationAudioTimeMs = observations.at(-1)?.audioTimeMs;
  if (latestAudioTimeMs === undefined || latestObservationAudioTimeMs === undefined) {
    return null;
  }

  const intervals = [];
  for (let index = 1; index < observations.length; index += 1) {
    intervals.push(observations[index].audioTimeMs - observations[index - 1].audioTimeMs);
  }
  intervals.sort((left, right) => left - right);
  const medianIntervalMs = intervals.length === 0
    ? null
    : intervals[Math.floor(intervals.length / 2)];
  return {
    ageMs: latestAudioTimeMs - latestObservationAudioTimeMs,
    medianIntervalMs,
    observationCount: observations.length,
  };
}
