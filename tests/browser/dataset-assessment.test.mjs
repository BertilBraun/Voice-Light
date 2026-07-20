import assert from "node:assert/strict";
import test from "node:test";

import {
  conversationStructureRegions,
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

test("classifies alternating dialogue from ordered speaker targets", () => {
  const conversation = {
    speaker1: {
      speech_segments: [{ start_seconds: 1, end_seconds: 8 }],
      segment_targets: [
        { start_seconds: 1, end_seconds: 4, text: "hello there" },
        { start_seconds: 7, end_seconds: 8, text: "right" },
      ],
    },
    speaker2: {
      speech_segments: [{ start_seconds: 4, end_seconds: 7 }],
      segment_targets: [
        { start_seconds: 4, end_seconds: 7, text: "how are you" },
      ],
    },
  };

  assert.equal(
    conversationStructureRegions(conversation, 30)[0].category,
    "alternating_dialogue",
  );
});

test("marks gaming language as a review candidate without excluding it", () => {
  const conversation = {
    speaker1: {
      speech_segments: [{ start_seconds: 1, end_seconds: 8 }],
      segment_targets: [
        {
          start_seconds: 1,
          end_seconds: 8,
          text: "enemy respawn on the map",
        },
      ],
    },
    speaker2: {
      speech_segments: [{ start_seconds: 8, end_seconds: 12 }],
      segment_targets: [
        { start_seconds: 8, end_seconds: 12, text: "I see them" },
      ],
    },
  };

  const region = conversationStructureRegions(conversation, 30)[0];

  assert.equal(region.category, "gaming_like");
  assert.deepEqual(region.gaming_terms, ["enemy", "map", "respawn"]);
});
