const datasetOptions = document.querySelector("#dataset-options");
const minimumQualityInput = document.querySelector("#minimum-quality");
const maximumMaskedRatioInput = document.querySelector("#maximum-masked-ratio");
const minimumPrimaryRatioInput = document.querySelector("#minimum-primary-ratio");
const minimumFutureRatioInput = document.querySelector("#minimum-future-ratio");
const runButton = document.querySelector("#run-button");
const status = document.querySelector("#status");
const summary = document.querySelector("#summary");
const pilotReadiness = document.querySelector("#pilot-readiness");
const datasetTable = document.querySelector("#dataset-table");
const categoryTable = document.querySelector("#category-table");
const rejectionTable = document.querySelector("#rejection-table");
const conversationTable = document.querySelector("#conversation-table");

async function loadDatasets() {
  const response = await fetch("/api/dataset-dashboard/datasets", { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(errorMessage(payload, response.status));
  }
  datasetOptions.replaceChildren(
    ...payload.datasets.map((dataset) => {
      const label = document.createElement("label");
      label.className = "dataset-option";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = dataset.id;
      input.checked = dataset.name === "meetings-s3";
      const name = document.createElement("span");
      name.textContent = dataset.name;
      label.append(input, name);
      return label;
    }),
  );
  if (payload.datasets.length === 0) {
    throw new Error("No datasets are available.");
  }
}

async function runAudit() {
  const selectedDatasetIds = Array.from(
    datasetOptions.querySelectorAll('input[type="checkbox"]:checked'),
    (input) => input.value,
  );
  if (selectedDatasetIds.length === 0) {
    setStatus("Select at least one dataset.", true);
    return;
  }
  if (!controlsAreValid()) {
    setStatus("Audit thresholds must be between zero and one.", true);
    return;
  }
  runButton.disabled = true;
  setStatus("Enumerating both speaker orientations…", false);
  try {
    const response = await fetch("/api/corpus-audit", {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_ids: selectedDatasetIds,
        minimum_quality: minimumQualityInput.valueAsNumber,
        maximum_masked_ratio: maximumMaskedRatioInput.valueAsNumber,
        minimum_primary_supervision_ratio: minimumPrimaryRatioInput.valueAsNumber,
        minimum_future_supervision_ratio: minimumFutureRatioInput.valueAsNumber,
      }),
    });
    const report = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(errorMessage(report, response.status));
    }
    renderReport(report);
    setStatus(
      `${report.accepted_window_count.toLocaleString()} windows accepted · ${formatHours(report.effective_supervised_duration_seconds)} effective supervision`,
      false,
    );
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  } finally {
    runButton.disabled = false;
  }
}

function controlsAreValid() {
  return [
    minimumQualityInput,
    maximumMaskedRatioInput,
    minimumPrimaryRatioInput,
    minimumFutureRatioInput,
  ].every((input) => input.checkValidity());
}

function renderReport(report) {
  const acceptanceRatio = ratio(report.accepted_window_count, report.candidate_window_count);
  const effectiveRatio = ratio(
    report.effective_supervised_duration_seconds,
    report.supervised_duration_seconds,
  );
  summary.replaceChildren(
    summaryCard("Conversations", report.accepted_conversation_count, `${report.conversation_count} eligible`),
    summaryCard("Accepted windows", report.accepted_window_count.toLocaleString(), percentage(acceptanceRatio)),
    summaryCard("Input audio", formatHours(report.input_duration_seconds), "20-second contexts"),
    summaryCard("Supervised", formatHours(report.supervised_duration_seconds), "before masks"),
    summaryCard("Effective supervision", formatHours(report.effective_supervised_duration_seconds), percentage(effectiveRatio)),
    summaryCard("Masked in accepted", formatHours(report.masked_duration_seconds), "permissive regions"),
  );
  renderPilotMetrics(report.pilot_metrics);
  renderDatasetTable(report.dataset_summaries);
  renderCategoryTable(report.categories, report.accepted_window_count);
  renderRejectionTable(report.rejections, report.candidate_window_count);
  renderConversationTable(report.conversations);
}

function renderPilotMetrics(metrics) {
  pilotReadiness.classList.remove("empty");
  pilotReadiness.replaceChildren(
    ...metrics.map((metric) => {
      const item = document.createElement("div");
      item.className = `readiness-item ${metric.ready ? "ready" : "not-ready"}`;
      const label = document.createElement("strong");
      label.textContent = metric.label;
      const value = document.createElement("span");
      value.textContent = `${formatMetric(metric.current, metric.unit)} / ${formatMetric(metric.target, metric.unit)}`;
      item.append(label, value);
      return item;
    }),
  );
}

function renderDatasetTable(datasets) {
  renderTable(
    datasetTable,
    ["Dataset", "Conversations", "Source", "Usable", "Windows", "Effective", "Mask"],
    datasets.map((dataset) => [
      dataset.dataset_name,
      `${dataset.accepted_conversation_count} / ${dataset.conversation_count}`,
      formatHours(dataset.source_duration_seconds),
      formatHours(dataset.usable_source_duration_seconds),
      dataset.accepted_window_count.toLocaleString(),
      formatHours(dataset.effective_supervised_duration_seconds),
      formatHours(dataset.masked_duration_seconds),
    ]),
  );
}

function renderCategoryTable(categories, acceptedWindowCount) {
  renderTable(
    categoryTable,
    ["Bucket", "Windows", "Share", "Events available", "Events covered"],
    categories.map((category) => [
      prettyName(category.kind),
      category.accepted_window_count.toLocaleString(),
      percentage(ratio(category.accepted_window_count, acceptedWindowCount)),
      category.available_oriented_event_count.toLocaleString(),
      category.covered_oriented_event_count.toLocaleString(),
    ]),
  );
}

function renderRejectionTable(rejections, candidateWindowCount) {
  renderTable(
    rejectionTable,
    ["Reason", "Windows", "Candidate share"],
    rejections.map((rejection) => [
      prettyName(rejection.reason),
      rejection.window_count.toLocaleString(),
      percentage(ratio(rejection.window_count, candidateWindowCount)),
    ]),
  );
}

function renderConversationTable(conversations) {
  const ordered = [...conversations].sort((first, second) => {
    const firstRatio = ratio(first.accepted_window_count, first.candidate_window_count);
    const secondRatio = ratio(second.accepted_window_count, second.candidate_window_count);
    return firstRatio - secondRatio || first.external_id.localeCompare(second.external_id);
  });
  renderTable(
    conversationTable,
    ["Dataset", "Conversation", "Quality", "Source", "Usable", "Windows", "Acceptance", "Effective", "Mask"],
    ordered.map((conversation) => [
      conversation.dataset_name,
      conversation.external_id,
      conversation.quality_score.toFixed(3),
      formatHours(conversation.source_duration_seconds),
      formatHours(conversation.usable_source_duration_seconds),
      `${conversation.accepted_window_count} / ${conversation.candidate_window_count}`,
      percentage(ratio(conversation.accepted_window_count, conversation.candidate_window_count)),
      formatHours(conversation.effective_supervised_duration_seconds),
      formatHours(conversation.masked_duration_seconds),
    ]),
  );
}

function renderTable(table, headings, rows) {
  const head = document.createElement("thead");
  const headingRow = document.createElement("tr");
  headings.forEach((heading, index) => {
    const cell = document.createElement("th");
    cell.textContent = heading;
    if (index > 0) cell.className = "number";
    headingRow.append(cell);
  });
  head.append(headingRow);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    row.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index > 0) cell.className = "number";
      tableRow.append(cell);
    });
    body.append(tableRow);
  });
  table.replaceChildren(head, body);
}

function summaryCard(labelText, valueText, noteText) {
  const card = document.createElement("article");
  card.className = "summary-card";
  const label = document.createElement("span");
  label.textContent = labelText;
  const value = document.createElement("strong");
  value.textContent = valueText;
  const note = document.createElement("small");
  note.textContent = noteText;
  card.append(label, value, note);
  return card;
}

function formatHours(seconds) {
  return `${(seconds / 3600).toFixed(2)} h`;
}

function formatMetric(value, unit) {
  return unit === "hours" ? `${value.toFixed(1)} h` : Math.round(value).toLocaleString();
}

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : 0;
}

function percentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function prettyName(value) {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

async function readJsonResponse(response) {
  const responseText = await response.text();
  try {
    return JSON.parse(responseText);
  } catch {
    throw new Error(`Request failed (${response.status}): ${responseText.trim() || response.statusText}`);
  }
}

function errorMessage(payload, statusCode) {
  return payload !== null && typeof payload === "object" && typeof payload.detail === "string"
    ? payload.detail
    : `Request failed (${statusCode})`;
}

function setStatus(message, isError) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

runButton.addEventListener("click", runAudit);

try {
  await loadDatasets();
  await runAudit();
} catch (error) {
  setStatus(error instanceof Error ? error.message : String(error), true);
}
