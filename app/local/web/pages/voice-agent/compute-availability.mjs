const GPU_BUDGET_EXHAUSTED_MESSAGE =
  "The monthly GPU allowance has been used up. Sorry—the voice demo will be available again when the €30 monthly compute credits reset.";
const MODAL_DISABLED_WORKSPACE_RESPONSE = /^modal-http: workspace \S+ is disabled\s*$/;

export function provisioningStatus(elapsedMilliseconds) {
  const elapsedSeconds = Math.max(0, Math.floor(elapsedMilliseconds / 1000));
  if (elapsedSeconds < 10) {
    return {
      title: "Requesting GPU…",
      detail: "Connecting through the European endpoint and requesting a two-GPU worker.",
    };
  }
  if (elapsedSeconds < 45) {
    return {
      title: "Provisioning GPU…",
      detail: `Waiting for GPU capacity and loading the voice models (${elapsedSeconds}s).`,
    };
  }
  return {
    title: "Still provisioning…",
    detail: `No worker is ready yet (${elapsedSeconds}s). Modal is searching its global GPU pool; you can keep waiting or stop and retry later.`,
  };
}

export function computeHealthUrl(websocketUrl) {
  const healthUrl = new URL(websocketUrl);
  healthUrl.protocol = healthUrl.protocol === "wss:" ? "https:" : "http:";
  healthUrl.pathname = "/health/live";
  healthUrl.search = "";
  healthUrl.hash = "";
  return healthUrl.toString();
}

export async function rejectIfModalGpuBudgetIsExhausted(websocketUrl, fetchRequest = fetch) {
  let response;
  try {
    response = await fetchRequest(computeHealthUrl(websocketUrl), { cache: "no-store" });
  } catch {
    return;
  }
  if (response.status !== 404) return;

  let responseBody;
  try {
    responseBody = await response.text();
  } catch {
    return;
  }
  if (MODAL_DISABLED_WORKSPACE_RESPONSE.test(responseBody)) {
    throw new Error(GPU_BUDGET_EXHAUSTED_MESSAGE);
  }
}
