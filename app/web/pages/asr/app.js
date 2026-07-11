const sessionSelect = document.querySelector("#session-select");
const speakerSelect = document.querySelector("#speaker-select");
const runButton = document.querySelector("#run-button");
const loadingIndicator = document.querySelector("#loading-indicator");
const summary = document.querySelector("#summary");
const modelSummary = document.querySelector("#model-summary");
const modelOptions = document.querySelector("#model-options");
const referenceInput = document.querySelector("#reference-input");
const audioPlayer = document.querySelector("#audio-player");
const metricsSummary = document.querySelector("#metrics-summary");
const metricsList = document.querySelector("#metrics-list");
const transcriptSummary = document.querySelector("#transcript-summary");
const transcriptList = document.querySelector("#transcript-list");

let sessions = [];
let models = [];

runButton.addEventListener("click", runAnalysis);

await loadInitialOptions();

async function loadInitialOptions() {
  const [sessionsPayload, modelsPayload] = await Promise.all([
    fetchJson("/api/sessions"),
    fetchJson("/api/asr/models"),
  ]);
  sessions = sessionsPayload.sessions;
  models = modelsPayload.models;
  sessionSelect.replaceChildren(
    ...sessions.map((session) => {
      const option = document.createElement("option");
      option.value = session.identifier;
      option.textContent = `${session.identifier} - ${formatSeconds(session.duration_seconds)}`;
      return option;
    }),
  );
  renderModelOptions();
  runButton.disabled = sessions.length === 0 || models.length === 0;
  summary.textContent =
    sessions.length === 0 ? "No sessions available." : "Select an audio track and model set.";
}

function renderModelOptions() {
  modelOptions.replaceChildren(
    ...models.map((model) => {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = model.mode;
      checkbox.checked = true;

      const title = document.createElement("strong");
      title.textContent = model.label;
      const description = document.createElement("span");
      description.textContent = model.description;
      const text = document.createElement("div");
      text.append(title, description);

      const option = document.createElement("label");
      option.className = "model-option";
      option.append(checkbox, text);
      return option;
    }),
  );
  modelSummary.textContent = `${models.length} ASR model adapters available`;
}

async function runAnalysis() {
  const selectedModels = [...modelOptions.querySelectorAll("input[type='checkbox']")]
    .filter((checkbox) => checkbox.checked)
    .map((checkbox) => checkbox.value);
  if (selectedModels.length === 0) {
    summary.textContent = "Select at least one model.";
    return;
  }
  setLoading(true);
  metricsList.replaceChildren();
  transcriptList.replaceChildren();
  try {
    const referenceWords = parseReferenceWords();
    const payload = await postJson("/api/asr/analyze", {
      session_id: sessionSelect.value,
      speaker_track: speakerSelect.value,
      models: selectedModels,
      reference_words: referenceWords,
    });
    applyAnalysisPayload(payload);
  } catch (error) {
    summary.textContent = error.message;
  } finally {
    setLoading(false);
  }
}

function parseReferenceWords() {
  const rawValue = referenceInput.value.trim();
  if (!rawValue) {
    return [];
  }
  const parsedValue = JSON.parse(rawValue);
  if (!Array.isArray(parsedValue)) {
    throw new Error("Reference words must be a JSON array.");
  }
  return parsedValue;
}

async function applyAnalysisPayload(payload) {
  audioPlayer.src = payload.audio_url;
  audioPlayer.load();
  summary.textContent = `${payload.session_id} ${payload.speaker_track} analyzed for ${formatSeconds(payload.analyzed_duration_seconds)}.`;
  metricsSummary.textContent =
    payload.reference_word_count > 0
      ? `${payload.reference_word_count} reference words`
      : "No reference words supplied";
  transcriptSummary.textContent = `${payload.runs.length} model runs`;
  renderMetrics(payload.runs);
  renderTranscripts(payload.runs);
}

function renderMetrics(runs) {
  metricsList.replaceChildren(
    ...runs.map((run) => {
      const card = document.createElement("div");
      card.className = "metric-card";
      const title = document.createElement("strong");
      title.textContent = run.model.label;
      const subtitle = document.createElement("span");
      subtitle.textContent = run.transcription.error
        ? run.transcription.error
        : `${run.transcription.words.length} words - RTF ${formatNullable(run.transcription.real_time_factor)}`;
      if (run.transcription.error) {
        subtitle.className = "error-text";
      }
      card.append(title, subtitle);

      const grid = document.createElement("div");
      grid.className = "metric-grid";
      if (run.metrics) {
        grid.append(
          metricValue("WER", formatNullable(run.metrics.word_error_counts.wer)),
          metricValue("Sub", run.metrics.word_error_counts.substitutions),
          metricValue("Ins", run.metrics.word_error_counts.insertions),
          metricValue("Del", run.metrics.word_error_counts.deletions),
          metricValue("Start p90", formatSeconds(run.metrics.timestamp_metrics.start.p90_absolute_error)),
          metricValue("End p90", formatSeconds(run.metrics.timestamp_metrics.end.p90_absolute_error)),
        );
      } else {
        grid.append(
          metricValue("Runtime", formatSeconds(run.transcription.processing_time_seconds)),
          metricValue("Load", formatSeconds(run.transcription.model_loading_time_seconds)),
          metricValue("Infer", formatSeconds(run.transcription.inference_time_seconds)),
          metricValue("Words", run.transcription.words.length),
        );
      }
      card.appendChild(grid);
      return card;
    }),
  );
}

function renderTranscripts(runs) {
  transcriptList.replaceChildren(
    ...runs.map((run) => {
      const card = document.createElement("div");
      card.className = "transcript-card";
      const title = document.createElement("strong");
      title.textContent = run.model.label;
      card.appendChild(title);
      if (run.transcription.error) {
        const error = document.createElement("p");
        error.className = "error-text";
        error.textContent = run.transcription.error;
        card.appendChild(error);
        return card;
      }
      card.appendChild(wordTable(run.transcription.words.slice(0, 200)));
      return card;
    }),
  );
}

function wordTable(words) {
  const table = document.createElement("table");
  table.className = "word-table";
  const head = document.createElement("thead");
  head.innerHTML = "<tr><th>#</th><th>Word</th><th>Start</th><th>End</th><th>Conf</th></tr>";
  const body = document.createElement("tbody");
  for (const [index, word] of words.entries()) {
    const row = document.createElement("tr");
    row.replaceChildren(
      cell(index + 1),
      cell(word.text),
      cell(formatSeconds(word.start_seconds)),
      cell(formatSeconds(word.end_seconds)),
      cell(formatNullable(word.confidence)),
    );
    body.appendChild(row);
  }
  table.append(head, body);
  return table;
}

function metricValue(label, value) {
  const container = document.createElement("div");
  container.className = "metric-value";
  const labelElement = document.createElement("span");
  labelElement.textContent = label;
  const valueElement = document.createElement("b");
  valueElement.textContent = String(value);
  container.append(labelElement, valueElement);
  return container;
}

function cell(value) {
  const element = document.createElement("td");
  element.textContent = String(value);
  return element;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function setLoading(isLoading) {
  runButton.disabled = isLoading;
  sessionSelect.disabled = isLoading;
  speakerSelect.disabled = isLoading;
  loadingIndicator.hidden = !isLoading;
  if (isLoading) {
    summary.textContent = "Running ASR analysis...";
  }
}

function formatSeconds(value) {
  if (value === null || value === undefined) {
    return "?";
  }
  return `${Number(value).toFixed(2)}s`;
}

function formatNullable(value) {
  if (value === null || value === undefined) {
    return "?";
  }
  return Number(value).toFixed(3);
}
