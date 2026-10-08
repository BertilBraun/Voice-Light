import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { PRODUCTION_VOICE_WEBSOCKET_URL } from "../../app/local/web/pages/voice-agent/public-config.mjs";
import {
  HIDDEN_SESSION_STOP_DELAY_MS,
  shouldScheduleHiddenSessionStop,
} from "../../app/local/web/pages/voice-agent/session-lifetime.mjs";

const pagePath = new URL("../../app/local/web/pages/voice-agent/index.html", import.meta.url);
const iconPath = new URL("../../app/local/web/pages/voice-agent/icon.svg", import.meta.url);

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
  assert.match(page, /src="\.\/app\.js\?v=25"/);
});

test("public demo bounds abandoned background sessions", async () => {
  const applicationPath = new URL(
    "../../app/local/web/pages/voice-agent/app.js",
    import.meta.url,
  );
  const application = await readFile(applicationPath, "utf8");

  assert.equal(HIDDEN_SESSION_STOP_DELAY_MS, 60_000);
  assert.equal(shouldScheduleHiddenSessionStop(true, 0), true);
  assert.equal(shouldScheduleHiddenSessionStop(true, 1), true);
  assert.equal(shouldScheduleHiddenSessionStop(false, 1), false);
  assert.equal(shouldScheduleHiddenSessionStop(true, 3), false);
  assert.match(application, /visibilitychange/);
  assert.match(application, /pagehide/);
});

test("public demo publishes branded browser metadata", async () => {
  const [page, icon] = await Promise.all([readFile(pagePath, "utf8"), readFile(iconPath, "utf8")]);

  assert.match(page, /name="theme-color" content="#16615a"/);
  assert.match(page, /rel="icon" href="\.\/icon\.svg" type="image\/svg\+xml"/);
  assert.match(icon, /viewBox="0 0 64 64"/);
  assert.match(icon, /fill="#16615a"/);
});
