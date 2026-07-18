import assert from "node:assert/strict";
import test from "node:test";

import { SpokenTextProgress } from "../../app/local/web/pages/voice-agent/spoken-text-progress.mjs";

test("retains a word-start offset that arrives before its text delta", () => {
  const progress = new SpokenTextProgress();
  progress.appendText("Let me check");

  progress.markSpoken(36);

  assert.equal(progress.spokenText(), "Let me check");
  assert.equal(progress.unspokenText(), "");

  progress.appendText(" the latest information.");

  assert.equal(progress.spokenText(), "Let me check the latest information.");
  assert.equal(progress.unspokenText(), "");
});

test("keeps later unspoken text separate from a retained boundary", () => {
  const progress = new SpokenTextProgress();
  progress.markSpoken(11);
  progress.appendText("First word. Final answer.");

  assert.equal(progress.spokenText(), "First word.");
  assert.equal(progress.unspokenText(), " Final answer.");
});

test("settles interrupted text at the acknowledged boundary", () => {
  const progress = new SpokenTextProgress();
  progress.appendText("One two three");
  progress.markSpoken(13);
  progress.acknowledge(7);

  progress.settleInterruptedText();

  assert.equal(progress.spokenText(), "One two");
  assert.equal(progress.unspokenText(), " three");
});
