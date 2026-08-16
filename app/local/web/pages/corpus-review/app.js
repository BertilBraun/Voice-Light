const pageParameters = new URLSearchParams(window.location.search);
const reviewSetName = pageParameters.get("name") ?? "prepublish-v1";
const statusElement = document.querySelector("#status");
const summaryElement = document.querySelector("#summary");
const itemsElement = document.querySelector("#review-items");
const gateIssuesElement = document.querySelector("#gate-issues");

function statusBadge(value) {
  const badge = document.createElement("span");
  badge.className = `status ${value}`;
  badge.textContent = value.toUpperCase();
  return badge;
}

function reviewLink(item) {
  const parameters = new URLSearchParams({
    review_set: reviewSetName,
    review_item_id: item.id,
    dataset_id: item.dataset_id,
    sample_id: item.sample_id,
    user_side: item.user_side,
    start_seconds: String(item.start_seconds),
  });
  const link = document.createElement("a");
  link.className = "review-link";
  link.href = `/training/sample-lab?${parameters}`;
  link.textContent = item.overall_status === "pending" ? "Review →" : "Open →";
  return link;
}

function renderSummary(items) {
  const passed = items.filter((item) => item.overall_status === "pass").length;
  const failed = items.filter((item) => item.overall_status === "fail").length;
  const pending = items.length - passed - failed;
  const datasets = new Set(items.map((item) => item.dataset_id)).size;
  summaryElement.replaceChildren(
    summaryCard("Datasets", datasets),
    summaryCard("Passed", passed),
    summaryCard("Failed", failed),
    summaryCard("Pending", pending),
  );
}

function summaryCard(label, value) {
  const card = document.createElement("article");
  const strong = document.createElement("strong");
  strong.textContent = String(value);
  const span = document.createElement("span");
  span.textContent = label;
  card.append(strong, span);
  return card;
}

function renderItems(items) {
  itemsElement.replaceChildren(
    ...items.map((item) => {
      const row = document.createElement("tr");
      const values = [
        item.dataset_name,
        item.external_id,
        `${item.start_seconds.toFixed(2)} s · ${item.user_side}`,
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      for (const value of [
        item.audio_status,
        item.annotation_status,
        item.label_status,
        item.overall_status,
      ]) {
        const cell = document.createElement("td");
        cell.append(statusBadge(value));
        row.append(cell);
      }
      const action = document.createElement("td");
      action.append(reviewLink(item));
      row.append(action);
      return row;
    }),
  );
}

async function loadReview() {
  const encodedName = encodeURIComponent(reviewSetName);
  const [reviewResponse, readinessResponse] = await Promise.all([
    fetch(`/api/corpus-review/sets/${encodedName}`, { cache: "no-store" }),
    fetch(`/api/corpus-review/sets/${encodedName}/readiness`, { cache: "no-store" }),
  ]);
  const payload = await reviewResponse.json();
  const readiness = await readinessResponse.json();
  if (!reviewResponse.ok) {
    throw new Error(payload.detail ?? `Review request failed (${reviewResponse.status})`);
  }
  if (!readinessResponse.ok) {
    throw new Error(
      readiness.detail ?? `Readiness request failed (${readinessResponse.status})`,
    );
  }
  renderSummary(payload.items);
  renderItems(payload.items);
  gateIssuesElement.replaceChildren(
    ...readiness.issues.map((issue) => {
      const item = document.createElement("li");
      item.textContent = issue.message;
      return item;
    }),
  );
  if (readiness.ready_to_publish) {
    const item = document.createElement("li");
    item.textContent = "All review evidence is current and every fixed crop passed.";
    gateIssuesElement.append(item);
  }
  const gate = readiness.ready_to_publish ? "READY TO PUBLISH" : "BLOCKED";
  statusElement.textContent =
    `${gate} · ${payload.items.length} fixed crops · seed ${payload.review_set.seed} · ` +
    payload.review_set.config.selection_algorithm;
}

try {
  await loadReview();
} catch (error) {
  statusElement.textContent = error instanceof Error ? error.message : String(error);
}
