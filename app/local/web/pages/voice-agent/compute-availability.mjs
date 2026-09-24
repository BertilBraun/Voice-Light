const GPU_BUDGET_EXHAUSTED_MESSAGE =
  "The monthly GPU allowance has been used up. Sorry—the voice demo will be available again when the €30 monthly compute credits reset.";
const MODAL_DISABLED_WORKSPACE_RESPONSE = /^modal-http: workspace \S+ is disabled\s*$/;

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
