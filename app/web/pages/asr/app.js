const sessionSelect = document.querySelector("#session-select");
const speakerSelect = document.querySelector("#speaker-select");
const runButton = document.querySelector("#run-button");
const loadingIndicator = document.querySelector("#loading-indicator");
const summary = document.querySelector("#summary");
const modelSettingsButton = document.querySelector("#model-settings-button");
const modelSettingsModal = document.querySelector("#model-settings-modal");
const closeModelSettingsButton = document.querySelector("#close-model-settings-button");
const cancelModelSettingsButton = document.querySelector("#cancel-model-settings-button");
const saveModelSettingsButton = document.querySelector("#save-model-settings-button");
const selectAllModelsButton = document.querySelector("#select-all-models-button");
const clearModelsButton = document.querySelector("#clear-models-button");
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
const timingSummary = document.querySelector("#timing-summary");
const timingWaveforms = document.querySelector("#timing-waveforms");

let sessions = [];
let models = [];
let selectedModelModes = new Set();
let stagedModelModes = new Set();
let currentPayload = null;
let activeWordKeys = new Map();
let playbackObjectUrl = null;
let waveformPeaks = [];
let timelineDurationSeconds = 0;
const sentenceGapSeconds = 0.15;

runButton.addEventListener("click", runAnalysis);
modelSettingsButton.addEventListener("click", openModelSettingsModal);
closeModelSettingsButton.addEventListener("click", closeModelSettingsModal);
cancelModelSettingsButton.addEventListener("click", closeModelSettingsModal);
saveModelSettingsButton.addEventListener("click", saveModelSettings);
selectAllModelsButton.addEventListener("click", selectAllModels);
clearModelsButton.addEventListener("click", clearModels);
playToggle.addEventListener("click", togglePlayback);
audioPlayer.addEventListener("timeupdate", updateActiveTranscriptWords);
audioPlayer.addEventListener("timeupdate", updatePlaybackState);
audioPlayer.addEventListener("timeupdate", drawTimingWaveforms);
audioPlayer.addEventListener("durationchange", updatePlaybackState);
audioPlayer.addEventListener("loadedmetadata", updatePlaybackState);
audioPlayer.addEventListener("canplay", updatePlaybackState);
audioPlayer.addEventListener("seeked", updateActiveTranscriptWords);
audioPlayer.addEventListener("seeked", drawTimingWaveforms);
audioPlayer.addEventListener("play", updateActiveTranscriptWords);
audioPlayer.addEventListener("play", updatePlaybackState);
audioPlayer.addEventListener("pause", updatePlaybackState);
audioPlayer.addEventListener("ended", updatePlaybackState);
window.addEventListener("resize", drawTimingWaveforms);

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
  selectedModelModes = new Set(models.map((model) => model.mode));
  stagedModelModes = new Set(selectedModelModes);
  updateModelSettingsSummary();
  setModelSettingsButtonsEnabled(models.length > 0);
  runButton.disabled = sessions.length === 0 || selectedModelModes.size === 0;
  summary.textContent =
    sessions.length === 0 ? "No sessions available." : "Select an audio track and model set.";
  drawTimingWaveforms();
}

function renderModelOptions() {
  modelOptions.replaceChildren(
    ...models.map((model) => {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = model.mode;
      checkbox.checked = stagedModelModes.has(model.mode);
      checkbox.addEventListener("change", updateStagedModelModesFromInputs);

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
  updateModelSettingsSummary();
  updateSaveModelSettingsButton();
}

async function runAnalysis() {
  const selectedModels = models
    .filter((model) => selectedModelModes.has(model.mode))
    .map((model) => model.mode);
  if (selectedModels.length === 0) {
    summary.textContent = "Select at least one model.";
    return;
  }
  setLoading(true);
  metricsList.replaceChildren();
  transcriptList.replaceChildren();
  timingWaveforms.replaceChildren();
  try {
    const referenceWords = parseReferenceWords();
    const payload = await postJson("/api/asr/analyze", {
      session_id: sessionSelect.value,
      speaker_track: speakerSelect.value,
      models: selectedModels,
      reference_words: referenceWords,
    });
    await applyAnalysisPayload(payload);
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
  timelineDurationSeconds = Number(payload.analyzed_duration_seconds);
  playToggle.disabled = true;
  await loadPlaybackAudio(payload.audio_url);
  summary.textContent = `${payload.session_id} ${payload.speaker_track} analyzed for ${formatSeconds(payload.analyzed_duration_seconds)}.`;
  metricsSummary.textContent =
    payload.reference_word_count > 0
      ? `${payload.reference_word_count} reference words`
      : "No reference words supplied";
  transcriptSummary.textContent = `${payload.runs.length} model runs`;
  renderMetrics(payload.runs);
  renderTranscripts(payload.runs);
  renderTimingWaveforms(payload.runs);
  updateActiveTranscriptWords();
  updatePlaybackState();
  drawTimingWaveforms();
}

function openModelSettingsModal() {
  stagedModelModes = new Set(selectedModelModes);
  renderModelOptions();
  modelSettingsModal.hidden = false;
}

function closeModelSettingsModal() {
  modelSettingsModal.hidden = true;
  stagedModelModes = new Set(selectedModelModes);
  updateModelSettingsSummary();
}

function updateStagedModelModesFromInputs() {
  stagedModelModes = new Set(
    [...modelOptions.querySelectorAll("input[type='checkbox']")]
      .filter((checkbox) => checkbox.checked)
      .map((checkbox) => checkbox.value),
  );
  updateModelSettingsSummary();
  updateSaveModelSettingsButton();
}

function selectAllModels() {
  stagedModelModes = new Set(models.map((model) => model.mode));
  renderModelOptions();
}

function clearModels() {
  stagedModelModes = new Set();
  renderModelOptions();
}

function saveModelSettings() {
  if (modelModeSetsEqual(selectedModelModes, stagedModelModes)) {
    closeModelSettingsModal();
    return;
  }
  selectedModelModes = new Set(stagedModelModes);
  updateModelSettingsSummary();
  closeModelSettingsModal();
  runButton.disabled = sessions.length === 0 || selectedModelModes.size === 0;
}

function updateModelSettingsSummary() {
  const displayedModelModes = modelSettingsModal.hidden ? selectedModelModes : stagedModelModes;
  modelSummary.textContent = `${displayedModelModes.size}/${models.length} models selected`;
}

function updateSaveModelSettingsButton() {
  saveModelSettingsButton.disabled = modelModeSetsEqual(selectedModelModes, stagedModelModes);
}

function setModelSettingsButtonsEnabled(areEnabled) {
  modelSettingsButton.disabled = !areEnabled;
  selectAllModelsButton.disabled = !areEnabled;
  clearModelsButton.disabled = !areEnabled;
}

function modelModeSetsEqual(firstModelModes, secondModelModes) {
  if (firstModelModes.size !== secondModelModes.size) {
    return false;
  }
  return [...firstModelModes].every((modelMode) => secondModelModes.has(modelMode));
}

async function loadPlaybackAudio(audioUrl) {
  waveformPeaks = [];
  drawTimingWaveforms();
  const response = await fetch(audioUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Audio download failed with HTTP ${response.status}`);
  }
  const audioBlob = await response.blob();
  const audioBuffer = await audioBlob.arrayBuffer();
  if (playbackObjectUrl !== null) {
    URL.revokeObjectURL(playbackObjectUrl);
  }
  playbackObjectUrl = URL.createObjectURL(audioBlob);
  audioPlayer.src = playbackObjectUrl;
  audioPlayer.load();
  await decodeWaveform(audioBuffer.slice(0));
}

async function decodeWaveform(audioBuffer) {
  const audioContext = new AudioContext();
  try {
    const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
    waveformPeaks = waveformPeaksFromChannel(decodedAudio.getChannelData(0), decodedAudio.sampleRate);
  } finally {
    await audioContext.close();
  }
}

function waveformPeaksFromChannel(samples, sampleRate) {
  const durationSeconds = samples.length / sampleRate;
  const peakCount = Math.max(1, Math.min(2400, Math.ceil(durationSeconds * 20)));
  const samplesPerPeak = Math.max(1, Math.floor(samples.length / peakCount));
  const peaks = [];
  for (let peakIndex = 0; peakIndex < peakCount; peakIndex += 1) {
    const startIndex = peakIndex * samplesPerPeak;
    const endIndex = Math.min(samples.length, startIndex + samplesPerPeak);
    let peak = 0;
    for (let sampleIndex = startIndex; sampleIndex < endIndex; sampleIndex += 1) {
      peak = Math.max(peak, Math.abs(samples[sampleIndex]));
    }
    peaks.push(peak);
  }
  return peaks;
}

function drawWaveform(canvas, speechSpans, color) {
  const waveformContext = canvas.getContext("2d");
  if (waveformContext === null) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * scale);
  canvas.height = Math.floor(rect.height * scale);
  waveformContext.setTransform(scale, 0, 0, scale, 0, 0);
  waveformContext.clearRect(0, 0, rect.width, rect.height);
  waveformContext.fillStyle = "#f7f8f6";
  waveformContext.fillRect(0, 0, rect.width, rect.height);

  if (waveformPeaks.length === 0) {
    waveformContext.fillStyle = "#5a666b";
    waveformContext.font = "14px sans-serif";
    waveformContext.fillText("Waveform unavailable", 14, 28);
    return;
  }

  drawSpeechSpans(waveformContext, rect, speechSpans, color);
  const middle = rect.height / 2;
  const amplitude = rect.height / 2 - 12;
  const columns = Math.max(1, Math.floor(rect.width));
  waveformContext.strokeStyle = "#425057";
  waveformContext.lineWidth = 1;
  for (let column = 0; column < columns; column += 1) {
    const peakStart = Math.floor((column / columns) * waveformPeaks.length);
    const peakEnd = Math.max(
      peakStart + 1,
      Math.ceil(((column + 1) / columns) * waveformPeaks.length),
    );
    const peak = Math.max(...waveformPeaks.slice(peakStart, peakEnd));
    const x = column + 0.5;
    waveformContext.beginPath();
    waveformContext.moveTo(x, middle - peak * amplitude);
    waveformContext.lineTo(x, middle + peak * amplitude);
    waveformContext.stroke();
  }

  if (timelineDurationSeconds > 0) {
    const playheadX = Math.min(audioPlayer.currentTime / timelineDurationSeconds, 1) * rect.width;
    waveformContext.strokeStyle = "#11181b";
    waveformContext.lineWidth = 2;
    waveformContext.beginPath();
    waveformContext.moveTo(playheadX, 10);
    waveformContext.lineTo(playheadX, rect.height - 10);
    waveformContext.stroke();
  }
}

function drawSpeechSpans(waveformContext, rect, speechSpans, color) {
  if (!Number.isFinite(timelineDurationSeconds) || timelineDurationSeconds <= 0) {
    return;
  }
  waveformContext.fillStyle = color;
  for (const speechSpan of speechSpans) {
    const startX = (speechSpan.startSeconds / timelineDurationSeconds) * rect.width;
    const endX = (speechSpan.endSeconds / timelineDurationSeconds) * rect.width;
    waveformContext.fillRect(startX, 0, Math.max(1, endX - startX), rect.height);
  }
}

function renderTimingWaveforms(runs) {
  timingWaveforms.replaceChildren(
    ...runs.map((run, runIndex) => timingWaveform(run, runIndex)),
  );
  timingSummary.textContent = `${runs.length} model timing tracks; words within 150ms are grouped.`;
  drawTimingWaveforms();
}

function timingWaveform(run, runIndex) {
  const track = document.createElement("section");
  track.className = "timing-track";
  const header = document.createElement("div");
  header.className = "timing-track-header";
  const label = document.createElement("strong");
  label.textContent = run.model.label;
  const detail = document.createElement("span");
  const spans = sentenceSpansFromWords(run.transcription.words);
  detail.textContent = run.transcription.error
    ? run.transcription.error
    : `${spans.length} speech blocks`;
  if (run.transcription.error) {
    detail.className = "error-text";
  }
  header.append(label, detail);
  const canvas = document.createElement("canvas");
  canvas.className = "timing-waveform";
  canvas.width = 1400;
  canvas.height = 112;
  canvas.dataset.runIndex = String(runIndex);
  track.append(header, canvas);
  return track;
}

function sentenceSpansFromWords(words) {
  const timestampedWords = words.filter(
    (word) => word.start_seconds !== null && word.end_seconds !== null,
  );
  const spans = [];
  for (const word of timestampedWords) {
    const startSeconds = Number(word.start_seconds);
    const endSeconds = Number(word.end_seconds);
    const previousSpan = spans.at(-1);
    if (previousSpan !== undefined && startSeconds - previousSpan.endSeconds < sentenceGapSeconds) {
      previousSpan.endSeconds = Math.max(previousSpan.endSeconds, endSeconds);
      continue;
    }
    spans.push({ startSeconds, endSeconds });
  }
  return spans;
}

function drawTimingWaveforms() {
  if (currentPayload === null) {
    return;
  }
  currentPayload.runs.forEach((run, runIndex) => {
    const canvas = timingWaveforms.querySelector(`[data-run-index='${runIndex}']`);
    if (canvas === null) {
      return;
    }
    drawWaveform(canvas, sentenceSpansFromWords(run.transcription.words), timingColor(runIndex));
  });
}

function timingColor(runIndex) {
  const colors = ["#bfe4d8", "#c9def4", "#f6d5a8", "#e3c9ef", "#f1c7cd"];
  return colors[runIndex % colors.length];
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
  runButton.disabled = isLoading || sessions.length === 0 || selectedModelModes.size === 0;
  modelSettingsButton.disabled = isLoading || models.length === 0;
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
