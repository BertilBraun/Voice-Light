import assert from "node:assert/strict";
import test from "node:test";

import {
  modelObservationSamples,
  summarizeModelObservationCadence,
  updateInteractionEvidence,
} from "../../app/local/web/pages/voice-agent/interaction-evidence.mjs";

function debugEvent(audioTimeMs, turnCompletion = null) {
  return {
    observed_audio_time_ms: audioTimeMs,
    silero_speech: false,
    assistant_audible: true,
    turn_completion_probability: turnCompletion,
    floor_take_probability: turnCompletion === null ? null : 0.2,
    non_floor_feedback_probability: turnCompletion === null ? null : 0.3,
  };
}

test("keeps heartbeat frames distinct from real model observations", () => {
  const evidence = new Map();
  updateInteractionEvidence(evidence, debugEvent(80));
  updateInteractionEvidence(evidence, debugEvent(160, 0.7));
  updateInteractionEvidence(evidence, debugEvent(240));

  const points = [...evidence.values()];
  assert.deepEqual(modelObservationSamples(points, "turnCompletion"), [
    { audioTimeMs: 160, probability: 0.7 },
  ]);
  assert.equal(points[2].turnCompletion, null);
  assert.equal(points[2].isModelObservation, false);
});

test("reports model cadence and the age of the latest observation", () => {
  const evidence = new Map();
  updateInteractionEvidence(evidence, debugEvent(80, 0.4));
  updateInteractionEvidence(evidence, debugEvent(240, 0.6));
  updateInteractionEvidence(evidence, debugEvent(400, 0.8));
  updateInteractionEvidence(evidence, debugEvent(480));

  assert.deepEqual(summarizeModelObservationCadence([...evidence.values()]), {
    ageMs: 80,
    medianIntervalMs: 160,
    observationCount: 3,
  });
});

test("preserves a real observation when a heartbeat shares its timestamp", () => {
  const evidence = new Map();
  updateInteractionEvidence(evidence, debugEvent(160, 0.7));
  updateInteractionEvidence(evidence, debugEvent(160));

  const point = evidence.get(160);
  assert.equal(point.isModelObservation, true);
  assert.equal(point.turnCompletion, 0.7);
});
