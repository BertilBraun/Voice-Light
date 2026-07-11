const sessionSelect = document.querySelector("#session-select");
const speakerSelect = document.querySelector("#speaker-select");
const runButton = document.querySelector("#run-button");
const loadingIndicator = document.querySelector("#loading-indicator");
const summary = document.querySelector("#summary");
const modelSummary = document.querySelector("#model-summary");
const modelOptions = document.querySelector("#model-options");
const referenceInput = document.querySelector("#reference-input");
const playToggle = document.querySelector("#play-toggle");
const playbackTime = document.querySelector("#playback-time");
const audioPlayer = document.querySelector("#audio-player");
const metricsSummary = document.querySelector("#metrics-summary");
const metricsList = document.querySelector("#metrics-list");
const transcriptSummary = document.querySelector("#transcript-summary");
const transcriptList = document.querySelector("#transcript-list");

let sessions = [];
let models = [];
let currentPayload = null;
let activeWordKeys = new Map();

runButton.addEventListener("click", runAnalysis);
playToggle.addEventListener("click", togglePlayback);
audioPlayer.addEventListener("timeupdate", updateActiveTranscriptWords);
audioPlayer.addEventListener("timeupdate", updatePlaybackState);
audioPlayer.addEventListener("durationchange", updatePlaybackState);
audioPlayer.addEventListener("loadedmetadata", updatePlaybackState);
audioPlayer.addEventListener("canplay", updatePlaybackState);
audioPlayer.addEventListener("seeked", updateActiveTranscriptWords);
audioPlayer.addEventListener("play", updateActiveTranscriptWords);
audioPlayer.addEventListener("play", updatePlaybackState);
audioPlayer.addEventListener("pause", updatePlaybackState);
audioPlayer.addEventListener("ended", updatePlaybackState);

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
  currentPayload = payload;
  activeWordKeys = new Map();
  playToggle.disabled = true;
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
  updateActiveTranscriptWords();
  updatePlaybackState();
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
    ...runs.map((run, runIndex) => {
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
      const wordRail = document.createElement("div");
      wordRail.className = "word-rail";
      wordRail.dataset.runIndex = String(runIndex);
      wordRail.replaceChildren(
        ...run.transcription.words.map((word, wordIndex) => transcriptWord(word, runIndex, wordIndex)),
      );
      card.appendChild(wordRail);
      return card;
    }),
  );
}

function transcriptWord(word, runIndex, wordIndex) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "transcript-word";
  element.textContent = word.text;
  element.dataset.runIndex = String(runIndex);
  element.dataset.wordIndex = String(wordIndex);
  element.title = `${formatSeconds(word.start_seconds)} - ${formatSeconds(word.end_seconds)}`;
  element.addEventListener("click", () => {
    if (word.start_seconds !== null && word.start_seconds !== undefined) {
      audioPlayer.currentTime = Number(word.start_seconds);
      audioPlayer.play();
    }
  });
  return element;
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

function updateActiveTranscriptWords() {
  if (currentPayload === null) {
    return;
  }
  const currentSeconds = audioPlayer.currentTime;
  currentPayload.runs.forEach((run, runIndex) => {
    const activeWordIndex = activeWordIndexForTime(run.transcription.words, currentSeconds);
    updateActiveTranscriptWord(runIndex, activeWordIndex);
  });
}

function activeWordIndexForTime(words, currentSeconds) {
  return words.findIndex((word) => {
    if (word.start_seconds === null || word.start_seconds === undefined) {
      return false;
    }
    if (word.end_seconds === null || word.end_seconds === undefined) {
      return currentSeconds >= Number(word.start_seconds);
    }
    return currentSeconds >= Number(word.start_seconds) && currentSeconds <= Number(word.end_seconds);
  });
}

function updateActiveTranscriptWord(runIndex, activeWordIndex) {
  const previousWordKey = activeWordKeys.get(runIndex);
  const nextWordKey = activeWordIndex === -1 ? null : `${runIndex}:${activeWordIndex}`;
  if (previousWordKey === nextWordKey) {
    return;
  }
  if (previousWordKey !== undefined && previousWordKey !== null) {
    const previousWord = transcriptList.querySelector(wordSelectorFromKey(previousWordKey));
    if (previousWord !== null) {
      previousWord.classList.remove("transcript-word-active");
    }
  }
  activeWordKeys.set(runIndex, nextWordKey);
  if (nextWordKey === null) {
    return;
  }
  const activeWord = transcriptList.querySelector(wordSelectorFromKey(nextWordKey));
  if (activeWord === null) {
    return;
  }
  activeWord.classList.add("transcript-word-active");
  centerTranscriptWord(activeWord);
}

function wordSelectorFromKey(wordKey) {
  const [runIndex, wordIndex] = wordKey.split(":");
  return `[data-run-index='${runIndex}'][data-word-index='${wordIndex}']`;
}

function centerTranscriptWord(wordElement) {
  const wordRail = wordElement.closest(".word-rail");
  if (wordRail === null) {
    return;
  }
  const railRect = wordRail.getBoundingClientRect();
  const wordRect = wordElement.getBoundingClientRect();
  const wordCenterOffset = wordRect.top - railRect.top + wordRail.scrollTop + wordRect.height / 2;
  const targetScrollTop = wordCenterOffset - wordRail.clientHeight / 2;
  const maximumScrollTop = wordRail.scrollHeight - wordRail.clientHeight;
  const nextScrollTop = Math.max(0, Math.min(targetScrollTop, maximumScrollTop));
  if (Math.abs(wordRail.scrollTop - nextScrollTop) > 1) {
    wordRail.scrollTop = nextScrollTop;
  }
}

async function togglePlayback() {
  if (audioPlayer.paused) {
    await audioPlayer.play();
    return;
  }
  audioPlayer.pause();
}

function updatePlaybackState() {
  playToggle.disabled = audioPlayer.currentSrc === "" || !Number.isFinite(audioPlayer.duration);
  playToggle.textContent = audioPlayer.paused ? "Play" : "Pause";
  playbackTime.textContent = `${formatSeconds(audioPlayer.currentTime)} / ${formatSeconds(audioPlayer.duration)}`;
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
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
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
