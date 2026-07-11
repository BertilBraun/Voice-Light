const state = {
  datasets: [],
  samples: [],
};

const elements = {
  status: document.querySelector("#status"),
  refreshButton: document.querySelector("#refresh-button"),
  datasetFilter: document.querySelector("#dataset-filter"),
  qualityMin: document.querySelector("#quality-min"),
  overlapMax: document.querySelector("#overlap-max"),
  flagFilter: document.querySelector("#flag-filter"),
  sampleList: document.querySelector("#sample-list"),
  sampleDetail: document.querySelector("#sample-detail"),
};

elements.refreshButton.addEventListener("click", loadDashboard);
for (const element of [
  elements.datasetFilter,
  elements.qualityMin,
  elements.overlapMax,
  elements.flagFilter,
]) {
  element.addEventListener("change", loadSamples);
}

loadDashboard();

async function loadDashboard() {
  elements.status.textContent = "Loading database records";
  const datasetsPayload = await fetchJson("/api/dataset-dashboard/datasets");
  state.datasets = datasetsPayload.datasets;
  renderDatasetOptions();
  await loadSamples();
}

async function loadSamples() {
  const parameters = new URLSearchParams();
  const datasetId = elements.datasetFilter.value;
  if (datasetId) {
    parameters.set("dataset_id", datasetId);
  }
  if (elements.qualityMin.value) {
    parameters.set("quality_min", elements.qualityMin.value);
  }
  if (elements.overlapMax.value) {
    parameters.set("overlap_ratio_max", elements.overlapMax.value);
  }
  if (elements.flagFilter.value.trim()) {
    parameters.set("flag", elements.flagFilter.value.trim());
  }
  const samplesPayload = await fetchJson(`/api/dataset-dashboard/samples?${parameters}`);
  state.samples = samplesPayload.samples;
  elements.status.textContent = `${state.datasets.length} datasets, ${state.samples.length} samples`;
  renderSamples();
}

function renderDatasetOptions() {
  elements.datasetFilter.replaceChildren();
  elements.datasetFilter.appendChild(new Option("All datasets", ""));
  for (const dataset of state.datasets) {
    elements.datasetFilter.appendChild(new Option(dataset.name, dataset.id));
  }
}

function renderSamples() {
  elements.sampleList.replaceChildren();
  for (const sample of state.samples) {
    const button = document.createElement("button");
    button.className = "sample-row";
    button.type = "button";
    button.addEventListener("click", () => renderSampleDetail(sample));

    const text = document.createElement("div");
    const title = document.createElement("div");
    title.className = "sample-id";
    title.textContent = sample.sample.external_id;
    const metrics = document.createElement("div");
    metrics.className = "metric-line";
    metrics.textContent = [
      formatSeconds(sample.sample.duration_seconds),
      `speech ${formatRatio(sample.latest_quality?.speech_ratio)}`,
      `overlap ${formatRatio(sample.latest_quality?.overlap_ratio)}`,
    ].join(" - ");
    text.append(title, metrics);

    const score = document.createElement("div");
    score.className = "score";
    score.textContent = formatRatio(sample.sample.quality_score);
    button.append(text, score);
    elements.sampleList.appendChild(button);
  }
}

function renderSampleDetail(sample) {
  const container = document.createElement("div");
  container.className = "detail-grid";

  const title = document.createElement("h2");
  title.textContent = sample.sample.external_id;
  container.appendChild(title);

  const audioGrid = document.createElement("div");
  audioGrid.className = "audio-grid";
  for (const track of sample.tracks) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = `/api/dataset-dashboard/audio/${sample.sample.id}/${track.side}`;
    const label = document.createElement("div");
    label.textContent = `${track.side} - ${formatSeconds(track.duration_seconds)} - ${track.sample_rate || "?"} Hz`;
    audioGrid.append(label, audio);
  }
  container.appendChild(audioGrid);

  const chips = document.createElement("div");
  chips.className = "chips";
  for (const flag of sample.sample.quality_flags) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = flag;
    chips.appendChild(chip);
  }
  container.appendChild(chips);

  const qualityPayload = document.createElement("pre");
  qualityPayload.textContent = JSON.stringify(sample.latest_quality?.payload || {}, null, 2);
  container.appendChild(qualityPayload);
  elements.sampleDetail.replaceChildren(container);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message);
  }
  return response.json();
}

function formatSeconds(value) {
  if (value === null || value === undefined) {
    return "duration ?";
  }
  return `${Number(value).toFixed(1)}s`;
}

function formatRatio(value) {
  if (value === null || value === undefined) {
    return "?";
  }
  return Number(value).toFixed(2);
}
