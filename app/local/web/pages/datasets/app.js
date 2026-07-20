import { drawAnnotationTimelineRow } from "/pages/shared/annotation-timeline.js";
import {
  conversationStructureRegions,
  sampleLanguageStatus,
  silenceMaskSegments,
} from "/pages/datasets/assessment.mjs";

const state = {
  datasets: [],
  samples: [],
  selectedSample: null,
  playbackController: null,
};

const metricDescriptions = new Map([
  ["Analyzed samples", "Samples with a current conversation-quality result, including results marked invalid."],
  ["Invalid samples", "Samples rejected before scoring, currently because the two track durations differ by more than 1%."],
  ["Analyzed audio", "Total audio duration actually inspected by ASR, capped at the first 180 seconds of each sample."],
  ["Represented audio", "Total full recording duration represented by the estimated conversation counts."],
  ["Speech segments", "ASR- and activity-derived stretches of speech found within the first 180 seconds of both tracks."],
  ["Interactions", "Turn-takings, backchannels, and interruptions detected within the first 180 seconds."],
  ["Turns", "Detected points where one speaker's contribution appears complete within the first 180 seconds."],
  ["Turn takings", "Transitions where the next transcript segment comes from the other speaker within the first 180 seconds."],
  ["Pauses", "Meaningful silences inside one speaker's ongoing contribution within the first 180 seconds."],
  ["Backchannels", "Short acknowledgments such as 'yeah,' 'right,' or 'mhm' that support the other speaker without taking the floor."],
  ["Interruptions", "Events where one speaker begins taking the floor before the other speaker has finished."],
  ["Useful events", "Turn completions, pauses, backchannels, and interruptions usable as annotation or training signals."],
  ["Overall weighted", "Weighted score: 15% full-file interaction, 10% full-file timing, 25% full-file audio, and 50% first-three-minute conversation annotation."],
  ["Interaction", "Full-recording score combining speech coverage, turn-completion rate, useful-candidate rate, and overlap."],
  ["Timing", "Full-recording average of plausible speech-segment durations and balance of detected events between speakers."],
  ["Audio", "Full-recording composite of both track scores, duration agreement, correlation, envelope correlation, and leakage."],
  ["Conversation", "First-three-minute average of speech coverage, useful-event density, and speaker balance."],
  ["Analyzed duration", "Duration covered by the ASR conversation annotation, capped at 180 seconds per sample."],
  ["Events / hour", "Useful first-three-minute annotation events normalized to an hourly rate."],
  ["Speaker balance", "Balance of annotated speech time between speakers: 1 is equal speaking time and 0 means only one speaker spoke."],
  ["Speech", "Fraction of the full recording where at least one track is classified as speech."],
  ["Silence", "Fraction of the full recording where neither track is classified as speech."],
  ["Overlap", "Fraction of the full recording where both tracks are simultaneously classified as speech."],
  ["Candidates / hour", "Full-recording rate of turn completions, pauses, responses, interruptions, and backchannels detected by energy VAD."],
  ["Turns / hour", "Full-recording rate of cross-speaker transitions separated by 0.05 to 2 seconds."],
  ["Responses / hour", "Full-recording rate at which the next speaker starts within 0.70 seconds of the previous speaker."],
  ["Interruptions / hour", "Full-recording rate of substantive overlapping starts classified as interruptions."],
  ["Backchannels / hour", "Full-recording rate of short overlapping acknowledgments classified as backchannels."],
  ["Median segment", "Median duration of all full-recording VAD speech segments across both tracks."],
  ["Median turn gap", "Median silence between full-recording cross-speaker turn and response candidates."],
  ["Median pause", "Median full-recording silence classified as an internal pause for one speaker."],
  ["Median overlap", "Median duration of full-recording overlap candidates."],
  ["Tiny fragments", "Fraction of full-recording VAD speech segments shorter than 0.20 seconds."],
  ["Long segments", "Fraction of full-recording VAD speech segments longer than 30 seconds."],
  ["Duration gap", "Absolute difference between the full lengths of the two speaker tracks."],
  ["Duration mismatch", "Track duration gap divided by the longer track duration."],
  ["Track correlation", "Full-recording waveform correlation between tracks; a high absolute value can indicate duplicated audio or bleed."],
  ["Envelope correlation", "Correlation between full-recording energy envelopes; a high positive value can indicate shared audio, reverb, or bleed."],
  ["Speaker 1 leakage", "Speaker 1 track level during speaker-2-only activity relative to speaker-1-only activity; more negative is better."],
  ["Speaker 2 leakage", "Speaker 2 track level during speaker-1-only activity relative to speaker-2-only activity; more negative is better."],
]);

const estimatedMetricDescriptions = new Map([
  ["Speech segments", "Estimated full-conversation speech segments, scaled from this sample's observed ASR annotation window."],
  ["Interactions", "Estimated full-conversation turn-takings, backchannels, and interruptions, scaled per sample."],
  ["Turns", "Estimated full-conversation completed speaker contributions, scaled per sample."],
  ["Turn takings", "Estimated full-conversation transitions to the other speaker, scaled per sample."],
  ["Pauses", "Estimated full-conversation meaningful internal silences, scaled per sample."],
  ["Backchannels", "Estimated full-conversation short acknowledgments, scaled per sample."],
  ["Interruptions", "Estimated full-conversation interruptions, scaled per sample."],
  ["Useful events", "Estimated full-conversation turn completions, pauses, backchannels, and interruptions available as training signals."],
  ["Represented duration", "The full duration represented by this sample's extrapolated conversation counts."],
  ["Scale factor", "Full sample duration divided by the ASR annotation duration; observed counts are multiplied by this value."],
]);

const audioTrackMetricDescriptions = new Map([
  ["Score", "Full-track audio score combining speech coverage, RMS level, clipping, near-zero samples, peak level, and speech/silence balance."],
  ["RMS", "Root-mean-square signal level over the full track in decibels relative to full scale."],
  ["Peak", "Largest absolute sample amplitude in the full track, where 1.0 is digital full scale."],
  ["Clipping", "Fraction of full-track samples at or above 99.9% of digital full scale."],
  ["Near zero", "Fraction of full-track samples whose absolute amplitude is at most 0.0001."],
  ["Speech", "Fraction of this full track classified as speech by energy VAD."],
]);

const eventTypeDescriptions = new Map([
  ["turn_completion", "A cross-speaker transition with 0.05 to 2 seconds between speakers."],
  ["pause", "At least 0.75 seconds of silence between one speaker's segments with no activity from the other speaker."],
  ["start_response", "The next speaker starts within 0.70 seconds after the previous speaker stops."],
  ["interruption", "A substantive speaker starts during the other speaker and overlaps for at least 0.20 seconds."],
  ["backchannel", "A short segment of at most 0.85 seconds occurs inside the other speaker's longer contribution."],
  ["overlap", "Both speaker tracks contain speech for at least 0.12 seconds at the same time."],
]);

let nextMetricTooltipId = 1;

const elements = {
  status: document.querySelector("#status"),
  refreshButton: document.querySelector("#refresh-button"),
  datasetFilter: document.querySelector("#dataset-filter"),
  languageFilter: document.querySelector("#language-filter"),
  qualityMin: document.querySelector("#quality-min"),
  overlapMax: document.querySelector("#overlap-max"),
  flagFilter: document.querySelector("#flag-filter"),
  sampleList: document.querySelector("#sample-list"),
  sampleDetail: document.querySelector("#sample-detail"),
  conversationSummary: document.querySelector("#conversation-summary"),
  completenessSummary: document.querySelector("#completeness-summary"),
};

elements.refreshButton.addEventListener("click", () => {
  void loadDashboard();
});
for (const element of [
  elements.datasetFilter,
  elements.languageFilter,
  elements.qualityMin,
  elements.overlapMax,
  elements.flagFilter,
]) {
  element.addEventListener("change", () => {
    void loadSamples().catch(renderDashboardError);
  });
}

void loadDashboard();

async function loadDashboard() {
  elements.status.textContent = "Loading database records";
  try {
    const datasetsPayload = await fetchJson("/api/dataset-dashboard/datasets");
    state.datasets = datasetsPayload.datasets;
    renderDatasetOptions();
    await loadSamples();
  } catch (error) {
    renderDashboardError(error);
  }
}

async function loadSamples() {
  elements.status.textContent = "Loading samples and statistics";
  const parameters = new URLSearchParams();
  const datasetId = elements.datasetFilter.value;
  if (datasetId) {
    parameters.set("dataset_id", datasetId);
  }
  if (elements.languageFilter.value) {
    parameters.set("language_status", elements.languageFilter.value);
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
  const completenessParameters = new URLSearchParams();
  if (datasetId) {
    completenessParameters.set("dataset_id", datasetId);
  }
  const samplesPayload = await fetchJson(
    `/api/dataset-dashboard/sample-summaries?${parameters}`,
  );
  const [summary, completeness] = await Promise.all([
    fetchJson(`/api/dataset-dashboard/conversation-summary?${parameters}`),
    fetchJson(`/api/dataset-dashboard/completeness?${completenessParameters}`),
  ]);
  state.samples = samplesPayload.samples;
  elements.status.textContent = `${state.datasets.length} datasets, ${state.samples.length} samples`;
  renderSamples();
  renderConversationSummary(summary);
  renderCompletenessSummary(completeness);
}

function renderDashboardError(error) {
  const reason = error instanceof Error ? error.message : "Unknown dashboard error";
  const message = `Could not load datasets: ${reason}. Use Refresh to retry.`;
  elements.status.textContent = message;
  elements.completenessSummary.replaceChildren(createEmptyMessage(message));
  elements.conversationSummary.replaceChildren(createEmptyMessage(message));
  elements.sampleList.replaceChildren(createEmptyMessage(message));
}

function renderCompletenessSummary(summary) {
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h2");
  title.textContent = "Ingestion completeness";
  const expected = document.createElement("span");
  expected.className = "muted";
  expected.textContent =
    `${summary.expected_metric_version} · ${summary.expected_annotation_version}`;
  heading.append(title, expected);

  const overview = document.createElement("div");
  overview.className = "completeness-overview";
  for (const [label, value, className] of [
    ["All samples", summary.sample_count, ""],
    ["Current accepted", summary.current_quality_sample_count, "complete"],
    ["Not current", summary.not_current_sample_count, "warning"],
    ["Duration-excluded", summary.duration_excluded_sample_count, ""],
    ["Reviewed offsets", summary.reviewed_current_quality_sample_count, "reviewed"],
    ["Unreviewed offsets", summary.unreviewed_current_quality_sample_count, ""],
  ]) {
    const card = document.createElement("div");
    card.className = `completeness-card ${className}`;
    const count = document.createElement("strong");
    count.textContent = String(value);
    const caption = document.createElement("span");
    caption.textContent = label;
    card.append(count, caption);
    overview.appendChild(card);
  }

  const grids = document.createElement("div");
  grids.className = "completeness-tables";
  grids.append(
    createCompletenessTable(
      "Latest quality / annotation versions",
      ["Quality version", "Annotation version", "Status", "Samples"],
      summary.quality_versions.map((row) => [
        row.metric_version,
        row.annotation_version || "none",
        row.status,
        row.sample_count,
      ]),
    ),
    createCompletenessTable(
      "Latest full-recording transcripts",
      ["Cohort", "Model", "Track", "Success", "Failed"],
      summary.full_asr_coverage.map((row) => [
        row.cohort,
        row.model_id,
        formatSpeaker(row.side),
        row.successful_transcript_count,
        row.failed_transcript_count,
      ]),
    ),
  );

  const note = document.createElement("p");
  note.className = "summary-note";
  note.textContent =
    "Current accepted samples have the expected quality and annotation versions. " +
    "Duration-excluded samples retain their older real results and are not counted as current.";
  elements.completenessSummary.replaceChildren(heading, overview, grids, note);
}

function createCompletenessTable(title, columnLabels, rows) {
  const section = document.createElement("section");
  const heading = document.createElement("h3");
  heading.textContent = title;
  const table = document.createElement("table");
  table.className = "completeness-table";
  const tableHead = document.createElement("thead");
  const headingRow = document.createElement("tr");
  for (const label of columnLabels) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    headingRow.appendChild(cell);
  }
  tableHead.appendChild(headingRow);
  const tableBody = document.createElement("tbody");
  for (const row of rows) {
    const tableRow = document.createElement("tr");
    for (const value of row) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      tableRow.appendChild(cell);
    }
    tableBody.appendChild(tableRow);
  }
  table.append(tableHead, tableBody);
  section.append(heading, table);
  return section;
}

function renderConversationSummary(summary) {
  const estimatedSection = createMetricSection("Estimated full-conversation totals", [
    ["Analyzed samples", summary.analyzed_sample_count, "integer"],
    ["Invalid samples", summary.invalid_sample_count, "integer"],
    ["Represented audio", summary.represented_duration_seconds / 3600, "hours"],
    ["Speech segments", summary.estimated_speech_segment_count, "estimatedInteger"],
    ["Interactions", summary.estimated_interaction_count, "estimatedInteger"],
    ["Turns", summary.estimated_turn_count, "estimatedInteger"],
    ["Turn takings", summary.estimated_turn_taking_count, "estimatedInteger"],
    ["Pauses", summary.estimated_pause_count, "estimatedInteger"],
    ["Backchannels", summary.estimated_backchannel_count, "estimatedInteger"],
    ["Interruptions", summary.estimated_interruption_count, "estimatedInteger"],
    ["Useful events", summary.estimated_usable_event_count, "estimatedInteger"],
  ]);
  const observedSection = createMetricSection("Observed ASR annotations", [
    ["Analyzed audio", summary.analyzed_duration_seconds / 3600, "hours"],
    ["Speech segments", summary.speech_segment_count, "integer"],
    ["Interactions", summary.interaction_count, "integer"],
    ["Turns", summary.turn_count, "integer"],
    ["Turn takings", summary.turn_taking_count, "integer"],
    ["Pauses", summary.pause_count, "integer"],
    ["Backchannels", summary.backchannel_count, "integer"],
    ["Interruptions", summary.interruption_count, "integer"],
    ["Useful events", summary.usable_event_count, "integer"],
  ]);
  const note = document.createElement("p");
  note.className = "summary-note";
  note.textContent = "Estimates scale each sample's first three minutes of ASR annotations to that sample's full duration.";
  elements.conversationSummary.replaceChildren(estimatedSection, observedSection, note);
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
    button.addEventListener("click", () => {
      void loadSampleDetail(sample);
    });

    const text = document.createElement("div");
    const title = document.createElement("div");
    title.className = "sample-id";
    title.textContent = sample.sample.external_id;
    const metrics = document.createElement("div");
    metrics.className = "metric-line";
    metrics.textContent = [
      formatSeconds(sample.sample.duration_seconds),
      `language ${formatLanguageStatus(sample.language_status)}`,
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

async function loadSampleDetail(sampleSummary) {
  state.playbackController?.destroy();
  elements.sampleDetail.className = "detail-empty";
  elements.sampleDetail.textContent = `Loading ${sampleSummary.sample.external_id}`;
  try {
    const sample = await fetchJson(
      `/api/dataset-dashboard/samples/${sampleSummary.sample.id}`,
    );
    state.selectedSample = sample;
    renderSampleDetail(sample);
  } catch (error) {
    elements.sampleDetail.className = "detail-empty";
    elements.sampleDetail.textContent =
      error instanceof Error ? error.message : "Could not load sample details";
  }
}

function renderSampleDetail(sample) {
  state.playbackController?.destroy();

  const container = document.createElement("div");
  container.className = "detail-grid";

  const payload = sample.latest_quality?.payload || {};
  const heading = document.createElement("div");
  heading.className = "detail-heading";
  const titleGroup = document.createElement("div");
  const title = document.createElement("h2");
  title.textContent = sample.sample.external_id;
  const version = document.createElement("div");
  version.className = "muted";
  version.textContent = payload.metric_version || "No completed quality result";
  titleGroup.append(title, version);
  heading.append(titleGroup, createScoreBadge(payload.total_quality_score));
  container.appendChild(heading);
  container.appendChild(createLanguageSection(sample));

  container.appendChild(
    createMetricSection("Quality scores", [
      ["Overall weighted", payload.total_quality_score, "score"],
      ["Interaction", payload.interaction_density?.quality_score, "score"],
      ["Timing", payload.timing_reliability?.quality_score, "score"],
      ["Audio", payload.audio_quality?.quality_score, "score"],
      ["Conversation", payload.conversation_annotation?.quality_score, "score"],
    ]),
  );

  const interaction = payload.interaction_density || {};
  const conversation = payload.conversation_annotation || {};
  const conversationEstimate = payload.conversation_count_estimate || {};
  const estimatedConversation = conversationEstimate.estimated || {};
  const annotationPlaybackController = createSynchronizedPlayback(sample, {
    title: payload.conversation_annotation
      ? "First three minutes with ASR annotations"
      : "First three minutes",
    trimmed: true,
    conversationAnnotation: payload.conversation_annotation,
  });
  container.appendChild(annotationPlaybackController.element);
  container.appendChild(
    createMetricSection("Observed ASR conversation annotation", [
      ["Analyzed duration", conversation.analyzed_duration_seconds, "seconds"],
      ["Speech segments", conversation.speech_segment_count, "integer"],
      ["Interactions", conversation.interaction_count, "integer"],
      ["Turns", conversation.turn_count, "integer"],
      ["Turn takings", conversation.turn_taking_count, "integer"],
      ["Pauses", conversation.pause_count, "integer"],
      ["Backchannels", conversation.backchannel_count, "integer"],
      ["Interruptions", conversation.interruption_count, "integer"],
      ["Useful events", conversation.usable_event_count, "integer"],
      ["Events / hour", conversation.events_per_hour, "number"],
      ["Speaker balance", conversation.speaker_balance_score, "score"],
    ]),
  );

  container.appendChild(
    createMetricSection("Estimated full conversation", [
      ["Represented duration", conversationEstimate.represented_duration_seconds, "seconds"],
      ["Scale factor", conversationEstimate.scale_factor, "multiplier"],
      ["Speech segments", estimatedConversation.speech_segment_count, "estimatedInteger"],
      ["Interactions", estimatedConversation.interaction_count, "estimatedInteger"],
      ["Turns", estimatedConversation.turn_count, "estimatedInteger"],
      ["Turn takings", estimatedConversation.turn_taking_count, "estimatedInteger"],
      ["Pauses", estimatedConversation.pause_count, "estimatedInteger"],
      ["Backchannels", estimatedConversation.backchannel_count, "estimatedInteger"],
      ["Interruptions", estimatedConversation.interruption_count, "estimatedInteger"],
      ["Useful events", estimatedConversation.usable_event_count, "estimatedInteger"],
    ]),
  );

  container.appendChild(
    createMetricSection("Interaction", [
      ["Speech", interaction.speech_ratio, "percent"],
      ["Silence", interaction.silence_ratio, "percent"],
      ["Overlap", interaction.overlap_ratio, "percent"],
      ["Candidates / hour", interaction.usable_candidate_windows_per_hour, "number"],
      ["Turns / hour", interaction.turn_completions_per_hour, "number"],
      ["Responses / hour", interaction.start_responses_per_hour, "number"],
      ["Interruptions / hour", interaction.interruptions_per_hour, "number"],
      ["Backchannels / hour", interaction.backchannels_per_hour, "number"],
    ]),
  );
  container.appendChild(
    createSilenceMaskPreview(
      payload.conversation_annotation,
      sample.sample.duration_seconds,
    ),
  );
  container.appendChild(
    createConversationStructurePreview(
      payload.conversation_annotation,
      sample.sample.duration_seconds,
    ),
  );

  const timing = payload.timing_reliability || {};
  container.appendChild(
    createMetricSection("Timing reliability", [
      ["Median segment", timing.median_segment_duration_seconds, "seconds"],
      ["Median turn gap", timing.median_turn_gap_seconds, "seconds"],
      ["Median pause", timing.median_pause_duration_seconds, "seconds"],
      ["Median overlap", timing.median_overlap_duration_seconds, "seconds"],
      ["Tiny fragments", timing.tiny_fragment_ratio, "percent"],
      ["Long segments", timing.long_segment_ratio, "percent"],
    ]),
  );

  container.appendChild(createAudioQualitySection(payload.audio_quality));

  const chips = document.createElement("div");
  chips.className = "chips";
  const flags = [...new Set([...(sample.sample.quality_flags || []), ...(payload.audio_quality?.flags || [])])];
  for (const flag of flags) {
    const chip = document.createElement("span");
    chip.className = "chip warning-chip";
    chip.textContent = flag;
    chips.appendChild(chip);
  }
  if (flags.length > 0) {
    container.appendChild(chips);
  }

  container.appendChild(
    createEventSection(payload.event_candidates || [], annotationPlaybackController),
  );

  const fullPlaybackController = createSynchronizedPlayback(sample, {
    title: "Full recordings",
    trimmed: false,
    conversationAnnotation: null,
  });
  container.appendChild(fullPlaybackController.element);
  state.playbackController = {
    destroy() {
      annotationPlaybackController.destroy();
      fullPlaybackController.destroy();
    },
  };

  const rawDetails = document.createElement("details");
  rawDetails.className = "raw-details";
  const rawSummary = document.createElement("summary");
  rawSummary.textContent = "Raw quality JSON";
  const qualityPayload = document.createElement("pre");
  qualityPayload.textContent = JSON.stringify(payload, null, 2);
  rawDetails.append(rawSummary, qualityPayload);
  container.appendChild(rawDetails);
  elements.sampleDetail.replaceChildren(container);
}

function createSynchronizedPlayback(sample, playbackOptions) {
  const section = document.createElement("section");
  section.className = "quality-section playback-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = playbackOptions.title;
  const status = document.createElement("span");
  status.className = "muted";
  status.textContent = "Loading audio";
  heading.append(title, status);

  const controls = document.createElement("div");
  controls.className = "playback-controls";
  const playButton = document.createElement("button");
  playButton.type = "button";
  playButton.className = "play-button";
  playButton.textContent = "Play both";
  const seek = document.createElement("input");
  seek.type = "range";
  seek.className = "seek";
  seek.min = "0";
  seek.max = "1000";
  seek.value = "0";
  seek.setAttribute("aria-label", "Playback position");
  const clock = document.createElement("span");
  clock.className = "playback-clock";
  clock.textContent = "0:00 / --:--";
  controls.append(playButton, seek, clock);

  const tracks = document.createElement("div");
  tracks.className = "track-grid";
  const trackPlayers = sample.tracks.map((track) => {
    const card = document.createElement("div");
    card.className = `track-card ${track.side}`;
    const trackTitle = document.createElement("strong");
    trackTitle.textContent = formatSpeaker(track.side);
    const metadata = document.createElement("span");
    metadata.className = "muted";
    const playbackDurationSeconds = playbackOptions.trimmed
      ? Math.min(track.duration_seconds, 180)
      : track.duration_seconds;
    metadata.textContent = `${formatSeconds(playbackDurationSeconds)} · ${track.sample_rate || "?"} Hz · ${track.channels || "?"} ch`;
    const waveformBox = document.createElement("div");
    waveformBox.className = "waveform-box";
    const canvas = document.createElement("canvas");
    canvas.className = playbackOptions.conversationAnnotation
      ? "waveform-canvas annotated-waveform-canvas"
      : "waveform-canvas";
    canvas.width = 1200;
    canvas.height = playbackOptions.conversationAnnotation ? 150 : 100;
    canvas.setAttribute(
      "aria-label",
      `${formatSpeaker(track.side)} ${playbackOptions.trimmed ? "three-minute annotated" : "full recording"} waveform`,
    );
    const waveformStatus = document.createElement("span");
    waveformStatus.className = "waveform-status";
    waveformStatus.textContent = playbackOptions.trimmed
      ? "Building three-minute waveform…"
      : "Building full waveform…";
    waveformBox.append(canvas, waveformStatus);
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.src = datasetAudioUrl(sample.sample.id, track.side, playbackOptions.trimmed);
    card.append(trackTitle, metadata, waveformBox, audio);
    tracks.appendChild(card);
    const player = { audio, canvas, waveformStatus, baseWaveform: null };
    canvas.addEventListener("click", (event) => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1);
      seekTo(ratio * duration());
    });
    void loadTrackWaveform(
      sample.sample.id,
      track.side,
      player,
      playbackOptions,
    ).then(update);
    return player;
  });
  const audioElements = trackPlayers.map((player) => player.audio);
  section.append(heading, controls, tracks);

  const master = audioElements[0];
  let destroyed = false;
  let seeking = false;

  function duration() {
    const durations = audioElements.map((audio) => audio.duration).filter(Number.isFinite);
    return durations.length === 0 ? 0 : Math.min(...durations);
  }

  function update() {
    if (destroyed || !master) {
      return;
    }
    const sharedDuration = duration();
    const currentTime = master.currentTime;
    if (!seeking && sharedDuration > 0) {
      seek.value = String(Math.round((currentTime / sharedDuration) * 1000));
    }
    clock.textContent = `${formatClock(currentTime)} / ${sharedDuration > 0 ? formatClock(sharedDuration) : "--:--"}`;
    playButton.textContent = master.paused ? "Play both" : "Pause both";
    status.textContent = audioElements.length === 2 ? "Speaker 1 + Speaker 2" : `${audioElements.length} track(s)`;
    for (const player of trackPlayers) {
      const trackDuration = player.audio.duration;
      const progress = Number.isFinite(trackDuration) && trackDuration > 0 ? currentTime / trackDuration : 0;
      renderWaveformProgress(player, progress);
    }
    if (!seeking && !master.paused) {
      for (const audio of audioElements.slice(1)) {
        if (Math.abs(audio.currentTime - currentTime) > 0.08) {
          audio.currentTime = currentTime;
        }
      }
    }
  }

  async function play() {
    if (!master) {
      return;
    }
    const startTime = master.currentTime;
    for (const audio of audioElements) {
      audio.currentTime = startTime;
    }
    const results = await Promise.allSettled(audioElements.map((audio) => audio.play()));
    if (results.some((result) => result.status === "rejected")) {
      audioElements.forEach((audio) => audio.pause());
      status.textContent = "Playback could not start";
    }
    update();
  }

  function pause() {
    audioElements.forEach((audio) => audio.pause());
    update();
  }

  function seekTo(seconds) {
    const sharedDuration = duration();
    const nextTime = sharedDuration > 0 ? Math.min(Math.max(seconds, 0), sharedDuration) : Math.max(seconds, 0);
    for (const audio of audioElements) {
      audio.currentTime = nextTime;
    }
    update();
  }

  playButton.addEventListener("click", () => {
    if (master?.paused !== false) {
      void play();
    } else {
      pause();
    }
  });
  seek.addEventListener("input", () => {
    seeking = true;
    seekTo((Number(seek.value) / 1000) * duration());
  });
  seek.addEventListener("change", () => {
    seeking = false;
    seekTo((Number(seek.value) / 1000) * duration());
  });
  for (const audio of audioElements) {
    audio.addEventListener("loadedmetadata", update);
    audio.addEventListener("ended", pause);
  }
  master?.addEventListener("timeupdate", update);

  return {
    element: section,
    seekTo,
    destroy() {
      destroyed = true;
      audioElements.forEach((audio) => {
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
      });
    },
  };
}

async function loadTrackWaveform(sampleId, side, player, playbackOptions) {
  try {
    const waveform = await fetchJson(
      datasetWaveformUrl(sampleId, side, playbackOptions.trimmed),
    );
    player.baseWaveform = drawWaveformEnvelope(
      player.canvas,
      waveform.points,
      waveform.duration_seconds,
      annotationForSide(playbackOptions.conversationAnnotation, side),
    );
    const scope = playbackOptions.trimmed ? "First three minutes" : "Full recording";
    player.waveformStatus.textContent = `${scope} · ${formatClock(waveform.duration_seconds)}`;
  } catch (error) {
    player.waveformStatus.textContent = error instanceof Error ? error.message : "Waveform unavailable";
  }
}

function datasetAudioUrl(sampleId, side, trimmed) {
  const query = trimmed ? "?trimmed=true" : "";
  return `/api/dataset-dashboard/audio/${sampleId}/${side}${query}`;
}

function datasetWaveformUrl(sampleId, side, trimmed) {
  const trimmedQuery = trimmed ? "&trimmed=true" : "";
  return `/api/dataset-dashboard/waveform/${sampleId}/${side}?points=1200${trimmedQuery}`;
}

function annotationForSide(conversationAnnotation, side) {
  if (!conversationAnnotation) {
    return null;
  }
  return side === "speaker1" ? conversationAnnotation.speaker1 : conversationAnnotation.speaker2;
}

function drawWaveformEnvelope(canvas, points, durationSeconds, annotation) {
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }
  const width = canvas.width;
  const height = annotation ? 82 : canvas.height;
  const midpoint = height / 2;
  context.fillStyle = "#f8fafc";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#d7dde5";
  context.beginPath();
  context.moveTo(0, midpoint + 0.5);
  context.lineTo(width, midpoint + 0.5);
  context.stroke();
  context.strokeStyle = "#386d9d";
  context.lineWidth = 1;
  context.beginPath();
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const x = (index / Math.max(points.length - 1, 1)) * width;
    context.moveTo(x, midpoint - point.maximum_amplitude * midpoint * 0.9);
    context.lineTo(x, midpoint - point.minimum_amplitude * midpoint * 0.9);
  }
  context.stroke();
  if (annotation) {
    drawAnnotationTimelineRow({
      context,
      annotation,
      left: 0,
      top: 92,
      width,
      viewportStartSeconds: 0,
      viewportEndSeconds: durationSeconds,
    });
  }
  return context.getImageData(0, 0, width, canvas.height);
}

function renderWaveformProgress(player, progress) {
  if (!player.baseWaveform) {
    return;
  }
  const context = player.canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.putImageData(player.baseWaveform, 0, 0);
  const position = Math.min(Math.max(progress, 0), 1) * player.canvas.width;
  context.fillStyle = "rgba(29, 79, 122, 0.14)";
  context.fillRect(0, 0, position, player.canvas.height);
  context.strokeStyle = "#b45309";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(position, 0);
  context.lineTo(position, player.canvas.height);
  context.stroke();
}

function createMetricSection(title, metrics) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  const descriptions = title.startsWith("Estimated")
    ? estimatedMetricDescriptions
    : metricDescriptions;
  for (const [label, value, format] of metrics) {
    const metric = document.createElement("div");
    metric.className = "metric-card";
    const metricValue = document.createElement("strong");
    metricValue.textContent = formatMetric(value, format);
    metric.append(createMetricLabel(label, descriptions), metricValue);
    grid.appendChild(metric);
  }
  section.append(heading, grid);
  return section;
}

function createMetricLabel(label, descriptions) {
  const labelRow = document.createElement("div");
  labelRow.className = "metric-label-row";
  const labelText = document.createElement("span");
  labelText.className = "metric-label-text";
  labelText.textContent = label;
  labelRow.appendChild(labelText);

  const description = descriptions.get(label);
  if (!description) {
    return labelRow;
  }
  appendTooltip(labelRow, label, description);
  return labelRow;
}

function appendTooltip(container, label, description) {
  const tooltipId = `metric-tooltip-${nextMetricTooltipId}`;
  nextMetricTooltipId += 1;
  const help = document.createElement("button");
  help.type = "button";
  help.className = "metric-help";
  help.textContent = "?";
  help.setAttribute("aria-label", `Explain ${label}`);
  help.setAttribute("aria-describedby", tooltipId);
  help.title = description;
  const tooltip = document.createElement("span");
  tooltip.id = tooltipId;
  tooltip.className = "metric-tooltip";
  tooltip.role = "tooltip";
  tooltip.textContent = description;
  container.append(help, tooltip);
}

function createAudioQualitySection(audioQuality) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("h3");
  heading.textContent = "Audio quality";
  section.appendChild(heading);
  if (!audioQuality) {
    section.appendChild(createEmptyMessage("No audio quality result"));
    return section;
  }

  const overview = createMetricSection("", [
    ["Duration gap", audioQuality.duration_gap_seconds, "seconds"],
    ["Duration mismatch", audioQuality.duration_gap_ratio, "percent"],
    ["Track correlation", audioQuality.track_correlation, "number"],
    ["Envelope correlation", audioQuality.energy_envelope_correlation, "number"],
    ["Speaker 1 leakage", audioQuality.speaker1_leakage_db, "db"],
    ["Speaker 2 leakage", audioQuality.speaker2_leakage_db, "db"],
  ]);
  overview.className = "audio-overview";
  section.appendChild(overview.querySelector(".metric-grid"));

  const trackGrid = document.createElement("div");
  trackGrid.className = "audio-quality-tracks";
  for (const [side, track] of [["Speaker 1", audioQuality.speaker1], ["Speaker 2", audioQuality.speaker2]]) {
    if (!track) {
      continue;
    }
    const card = document.createElement("div");
    card.className = "audio-quality-card";
    const title = document.createElement("strong");
    title.textContent = side;
    const details = document.createElement("dl");
    for (const [label, value] of [
      ["Score", formatMetric(track.quality_score, "score")],
      ["RMS", formatMetric(track.rms_dbfs, "db")],
      ["Peak", formatMetric(track.peak_amplitude, "number")],
      ["Clipping", formatMetric(track.clipping_ratio, "percent")],
      ["Near zero", formatMetric(track.near_zero_ratio, "percent")],
      ["Speech", formatMetric(track.speech_ratio, "percent")],
    ]) {
      const term = document.createElement("dt");
      term.textContent = label;
      const metricDescription = audioTrackMetricDescriptions.get(label);
      if (metricDescription) {
        term.className = "definition-with-tooltip";
        appendTooltip(term, `${side} ${label}`, metricDescription);
      }
      const description = document.createElement("dd");
      description.textContent = value;
      details.append(term, description);
    }
    card.append(title, details);
    trackGrid.appendChild(card);
  }
  section.appendChild(trackGrid);
  return section;
}

function createLanguageSection(sample) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = "Language preflight";
  const status = document.createElement("span");
  status.className = "muted";
  status.textContent = sampleLanguageStatus(sample.language_assessments);
  heading.append(title, status);
  section.appendChild(heading);

  if (!sample.language_assessments?.length) {
    section.appendChild(createEmptyMessage("No language preflight result"));
    return section;
  }

  const grid = document.createElement("div");
  grid.className = "language-track-grid";
  for (const assessment of sample.language_assessments) {
    const track = sample.tracks.find(
      (candidate) => candidate.id === assessment.sample_track_id,
    );
    const card = document.createElement("article");
    card.className = "language-track";
    const cardHeading = document.createElement("div");
    cardHeading.className = "section-heading";
    const trackTitle = document.createElement("strong");
    trackTitle.textContent = formatSpeaker(track?.side);
    const trackStatus = document.createElement("span");
    trackStatus.className = `language-status language-${assessment.status}`;
    trackStatus.textContent = formatLanguageStatus(assessment.status);
    cardHeading.append(trackTitle, trackStatus);

    const evidence = document.createElement("dl");
    for (const [label, value] of [
      ["Detected", assessment.language_code || "—"],
      ["Confidence", formatPercent(assessment.confidence)],
      ["Probe words", String(assessment.transcript_word_count)],
      ["Windows", formatProbeWindows(assessment.probe_windows)],
    ]) {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = value;
      evidence.append(term, description);
    }

    const transcript = document.createElement("p");
    transcript.className = "language-transcript";
    transcript.textContent =
      assessment.transcript_text || assessment.error || "No probe transcript";
    card.append(cardHeading, evidence, transcript);
    grid.appendChild(card);
  }
  section.appendChild(grid);
  return section;
}

function createSilenceMaskPreview(conversation, representedDurationSeconds) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = "Silence masking preview";
  const summary = document.createElement("span");
  summary.className = "muted";
  heading.append(title, summary);
  section.appendChild(heading);

  if (!conversation) {
    section.appendChild(createEmptyMessage("Available after conversation analysis"));
    return section;
  }

  const durationSeconds = Number(
    conversation.analyzed_duration_seconds || representedDurationSeconds || 0,
  );
  const controls = document.createElement("label");
  controls.className = "silence-control";
  const controlText = document.createElement("span");
  const threshold = document.createElement("input");
  threshold.type = "range";
  threshold.min = "1";
  threshold.max = "30";
  threshold.step = "1";
  threshold.value = "5";
  threshold.setAttribute("aria-label", "Minimum silence to mask in seconds");
  controls.append(controlText, threshold);

  const timeline = document.createElement("div");
  timeline.className = "silence-timeline";
  timeline.setAttribute("role", "img");
  const legend = document.createElement("div");
  legend.className = "silence-legend";
  legend.textContent =
    "Blue is retained context; gray is proposed masking. This preview does not alter ingestion.";

  function updatePreview() {
    const minimumSilenceSeconds = Number(threshold.value);
    controlText.textContent =
      `Mask silence lasting at least ${minimumSilenceSeconds} s`;
    const masks = silenceMaskSegments(
      conversation,
      durationSeconds,
      minimumSilenceSeconds,
    );
    timeline.replaceChildren();
    let maskedSeconds = 0;
    for (const mask of masks) {
      const span = document.createElement("span");
      span.className = "silence-mask";
      span.style.left = `${(mask.start_seconds / durationSeconds) * 100}%`;
      span.style.width =
        `${((mask.end_seconds - mask.start_seconds) / durationSeconds) * 100}%`;
      timeline.appendChild(span);
      maskedSeconds += mask.end_seconds - mask.start_seconds;
    }
    const retainedSeconds = Math.max(0, durationSeconds - maskedSeconds);
    summary.textContent =
      `${formatClock(retainedSeconds)} retained · ${formatClock(maskedSeconds)} masked`;
    timeline.setAttribute(
      "aria-label",
      `${masks.length} silence regions would mask ${formatClock(maskedSeconds)} of ${formatClock(durationSeconds)}`,
    );
  }

  threshold.addEventListener("input", updatePreview);
  updatePreview();
  section.append(controls, timeline, legend);
  return section;
}

function createConversationStructurePreview(
  conversation,
  representedDurationSeconds,
) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = "Conversation structure";
  const status = document.createElement("span");
  status.className = "muted";
  heading.append(title, status);
  section.appendChild(heading);

  if (!conversation) {
    section.appendChild(createEmptyMessage("Available after conversation analysis"));
    return section;
  }
  const durationSeconds = Number(
    conversation.analyzed_duration_seconds || representedDurationSeconds || 0,
  );
  const regions = conversationStructureRegions(
    conversation,
    durationSeconds,
    30,
  );
  const grid = document.createElement("div");
  grid.className = "structure-grid";
  grid.setAttribute("role", "img");
  grid.setAttribute(
    "aria-label",
    `${regions.length} thirty-second conversation structure regions`,
  );
  const counts = new Map();
  for (const region of regions) {
    counts.set(region.category, (counts.get(region.category) || 0) + 1);
    const cell = document.createElement("span");
    cell.className = `structure-region structure-${region.category}`;
    const gamingEvidence = region.gaming_terms.length
      ? `; terms ${region.gaming_terms.join(", ")}`
      : "";
    cell.title =
      `${formatClock(region.start_seconds)}–${formatClock(region.end_seconds)}; ` +
      `${formatStructureCategory(region.category)}; ` +
      `${region.speaker_transitions} speaker transitions${gamingEvidence}`;
    cell.setAttribute("aria-label", cell.title);
    grid.appendChild(cell);
  }

  const legend = document.createElement("div");
  legend.className = "structure-legend";
  for (const category of [
    "alternating_dialogue",
    "gaming_like",
    "single_speaker",
    "two_speaker_unstructured",
    "silence",
  ]) {
    const item = document.createElement("span");
    item.className = `structure-legend-item structure-${category}`;
    item.textContent =
      `${formatStructureCategory(category)} ${counts.get(category) || 0}`;
    legend.appendChild(item);
  }
  const alternatingCount = counts.get("alternating_dialogue") || 0;
  status.textContent =
    `${alternatingCount} of ${regions.length} regions show alternating dialogue`;
  const note = document.createElement("p");
  note.className = "summary-note";
  note.textContent =
    "Gaming-like regions are transcript-keyword review candidates, not automatic exclusions.";
  section.append(grid, legend, note);
  return section;
}

function formatStructureCategory(category) {
  switch (category) {
    case "alternating_dialogue":
      return "Alternating dialogue";
    case "gaming_like":
      return "Gaming-like";
    case "single_speaker":
      return "Single speaker";
    case "two_speaker_unstructured":
      return "Two-speaker, no alternation";
    default:
      return "Silence";
  }
}

function createEventSection(events, playbackController) {
  const section = document.createElement("section");
  section.className = "quality-section event-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = "Event candidates";
  const total = document.createElement("span");
  total.className = "muted";
  total.textContent = `${events.length} stored candidates`;
  heading.append(title, total);
  section.appendChild(heading);
  if (events.length === 0) {
    section.appendChild(createEmptyMessage("No event candidates detected"));
    return section;
  }

  const counts = new Map();
  for (const event of events) {
    counts.set(event.event_type, (counts.get(event.event_type) || 0) + 1);
  }
  const chips = document.createElement("div");
  chips.className = "chips event-counts";
  for (const [eventType, count] of [...counts.entries()].sort()) {
    const chip = document.createElement("span");
    chip.className = `chip event-${eventType}`;
    chip.textContent = `${formatEventType(eventType)} ${count}`;
    const eventDescription = eventTypeDescriptions.get(eventType);
    if (eventDescription) {
      chip.title = eventDescription;
      chip.tabIndex = 0;
      chip.setAttribute("aria-label", `${chip.textContent}: ${eventDescription}`);
    }
    chips.appendChild(chip);
  }
  section.appendChild(chips);

  const eventDetails = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Browse and seek to candidates";
  const list = document.createElement("div");
  list.className = "event-list";
  for (const event of events) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "event-row";
    row.addEventListener("click", () => playbackController.seekTo(event.start_seconds));
    const type = document.createElement("strong");
    type.textContent = formatEventType(event.event_type);
    const eventDescription = eventTypeDescriptions.get(event.event_type);
    if (eventDescription) {
      type.title = eventDescription;
    }
    const speakers = document.createElement("span");
    speakers.textContent = event.secondary_speaker
      ? `${formatSpeaker(event.primary_speaker)} → ${formatSpeaker(event.secondary_speaker)}`
      : formatSpeaker(event.primary_speaker);
    const timing = document.createElement("span");
    timing.textContent = `${formatClock(event.start_seconds)} · ${formatEventDuration(event)}`;
    row.append(type, speakers, timing);
    list.appendChild(row);
  }
  eventDetails.append(summary, list);
  section.appendChild(eventDetails);
  return section;
}

function createScoreBadge(value) {
  const badge = document.createElement("div");
  badge.className = `score-badge ${scoreClass(value)}`;
  badge.textContent = formatMetric(value, "score");
  const description = metricDescriptions.get("Overall weighted");
  badge.title = description;
  badge.setAttribute("aria-label", `Overall ${badge.textContent}: ${description}`);
  return badge;
}

function createEmptyMessage(message) {
  const empty = document.createElement("div");
  empty.className = "detail-empty";
  empty.textContent = message;
  return empty;
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

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatLanguageStatus(value) {
  if (!value) {
    return "not assessed";
  }
  return String(value).replaceAll("_", " ");
}

function formatProbeWindows(windows) {
  if (!windows?.length) {
    return "—";
  }
  return windows
    .map(
      (window) =>
        `${formatClock(window.start_seconds)} (${Number(window.rms_dbfs).toFixed(1)} dBFS)`,
    )
    .join(", ");
}

function formatMetric(value, format) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "—";
  }
  const number = Number(value);
  switch (format) {
    case "score":
      return number.toFixed(2);
    case "percent":
      return `${(number * 100).toFixed(1)}%`;
    case "seconds":
      return `${number.toFixed(2)} s`;
    case "db":
      return `${number.toFixed(1)} dB`;
    case "integer":
      return String(Math.round(number));
    case "estimatedInteger":
      return `≈${Math.round(number).toLocaleString()}`;
    case "multiplier":
      return `${number.toFixed(2)}×`;
    case "hours":
      return `${number.toFixed(2)} h`;
    default:
      return number.toFixed(2);
  }
}

function formatClock(value) {
  if (!Number.isFinite(Number(value))) {
    return "--:--";
  }
  const totalSeconds = Math.max(0, Math.floor(Number(value)));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatSpeaker(side) {
  if (side === "speaker1") {
    return "Speaker 1";
  }
  if (side === "speaker2") {
    return "Speaker 2";
  }
  return side || "Unknown speaker";
}

function formatEventType(eventType) {
  return String(eventType)
    .split("_")
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function formatEventDuration(event) {
  const duration = event.gap_seconds ?? event.overlap_seconds ?? event.end_seconds - event.start_seconds;
  return `${Number(duration).toFixed(2)} s`;
}

function scoreClass(value) {
  if (!Number.isFinite(Number(value))) {
    return "score-unknown";
  }
  if (Number(value) >= 0.75) {
    return "score-good";
  }
  if (Number(value) >= 0.45) {
    return "score-medium";
  }
  return "score-bad";
}
