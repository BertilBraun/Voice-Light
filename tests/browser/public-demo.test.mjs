import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { PRODUCTION_VOICE_WEBSOCKET_URL } from "../../app/local/web/pages/voice-agent/public-config.mjs";

const pagePath = new URL("../../app/local/web/pages/voice-agent/index.html", import.meta.url);

test("public demo uses the production secure WebSocket", () => {
  assert.equal(
    PRODUCTION_VOICE_WEBSOCKET_URL,
    "wss://bertil-braun-private--voicelightagent-voice-light.eu-west.modal.run/v1/voice",
  );
});

test("public demo hides endpoint configuration and generated text by default", async () => {
  const page = await readFile(pagePath, "utf8");

  assert.doesNotMatch(page, /endpoint-url/);
  assert.match(page, /id="generated-text-toggle" type="checkbox"/);
  assert.match(page, /href="\.\/styles\.css"/);
  assert.match(page, /src="\.\/app\.js\?v=20"/);
});
