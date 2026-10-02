import assert from "node:assert/strict";
import test from "node:test";

import {
  computeHealthUrl,
  provisioningStatus,
  rejectIfModalGpuBudgetIsExhausted,
} from "../../app/local/web/pages/voice-agent/compute-availability.mjs";

const WEBSOCKET_URL = "wss://example--voice.modal.run/v1/voice";

test("compute health URL uses HTTPS on the Modal origin", () => {
  assert.equal(computeHealthUrl(WEBSOCKET_URL), "https://example--voice.modal.run/health/live");
});

test("provisioning status explains each cold-start phase", () => {
  assert.deepEqual(provisioningStatus(0), {
    title: "Requesting GPU…",
    detail: "Connecting through the European endpoint and requesting a two-GPU worker.",
  });
  assert.deepEqual(provisioningStatus(12_900), {
    title: "Provisioning GPU…",
    detail: "Waiting for GPU capacity and loading the voice models (12s).",
  });
  assert.deepEqual(provisioningStatus(52_100), {
    title: "Still provisioning…",
    detail: "No worker is ready yet (52s). Modal is searching its global GPU pool; you can keep waiting or stop and retry later.",
  });
});

test("disabled Modal workspace explains the exhausted monthly GPU budget", async () => {
  const fetchRequest = async () =>
    new Response("modal-http: workspace ac-example is disabled\n", { status: 404 });

  await assert.rejects(
    rejectIfModalGpuBudgetIsExhausted(WEBSOCKET_URL, fetchRequest),
    new Error(
      "The monthly GPU allowance has been used up. Sorry—the voice demo will be available again when the €30 monthly compute credits reset.",
    ),
  );
});

test("ordinary missing endpoints do not masquerade as exhausted GPU budget", async () => {
  const fetchRequest = async () => new Response("Not found", { status: 404 });

  await rejectIfModalGpuBudgetIsExhausted(WEBSOCKET_URL, fetchRequest);
});

test("preflight network errors fall through to the WebSocket connection", async () => {
  const fetchRequest = async () => {
    throw new TypeError("Failed to fetch");
  };

  await rejectIfModalGpuBudgetIsExhausted(WEBSOCKET_URL, fetchRequest);
});
