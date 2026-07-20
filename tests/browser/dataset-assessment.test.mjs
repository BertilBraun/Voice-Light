import assert from "node:assert/strict";
import test from "node:test";

import {
  sampleLanguageStatus,
  silenceMaskSegments,
} from "../../app/local/web/pages/datasets/assessment.mjs";

test("masks only silence at or above the selected threshold", () => {
  const conversation = {
    speaker1: {
      speech_segments: [
        { start_seconds: 2, end_seconds: 8 },
        { start_seconds: 11, end_seconds: 15 },
      ],
    },
    speaker2: {
      speech_segments: [{ start_seconds: 20, end_seconds: 25 }],
    },
  };

  assert.deepEqual(silenceMaskSegments(conversation, 30, 5), [
    { start_seconds: 15, end_seconds: 20 },
    { start_seconds: 25, end_seconds: 30 },
  ]);
});

test("merges overlapping speaker activity before finding silence", () => {
  const conversation = {
    speaker1: {
      speech_segments: [{ start_seconds: 0, end_seconds: 10 }],
    },
    speaker2: {
      speech_segments: [{ start_seconds: 8, end_seconds: 12 }],
    },
  };

  assert.deepEqual(silenceMaskSegments(conversation, 20, 5), [
    { start_seconds: 12, end_seconds: 20 },
  ]);
});

test("reports a non-English track as an excluded sample", () => {
  assert.equal(
    sampleLanguageStatus([
      { status: "english" },
      { status: "non_english" },
    ]),
    "Non-English — full analysis skipped",
  );
});

test("keeps inconclusive language visible without calling it excluded", () => {
  assert.equal(
    sampleLanguageStatus([
      { status: "english" },
      { status: "inconclusive" },
    ]),
    "Inconclusive — full analysis allowed",
  );
});
