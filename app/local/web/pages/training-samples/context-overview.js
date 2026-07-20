import { commonWaveformDisplayScale } from "/pages/shared/waveform-rendering.js";

const CONTEXT_DURATION_SECONDS = 180;
const CONTEXT_WAVEFORM_POINTS = 1200;

export function createConversationContextOverview(options) {
  const {
    canvas,
    label,
    getPreview,
    getSelectedStartSeconds,
    setSelectedStartSeconds,
    commitSelection,
  } = options;
  let waveforms = null;
  let requestGeneration = 0;
  let dragging = false;

  async function load(selectedStartSeconds) {
    const preview = getPreview();
    if (preview === null) {
      return;
    }
    const bounds = contextBounds(preview, selectedStartSeconds);
    const generation = ++requestGeneration;
    label.textContent =
      `Loading ${formatDuration(bounds.startSeconds)}–${formatDuration(bounds.endSeconds)}…`;
    const parameters = new URLSearchParams({
      points: String(CONTEXT_WAVEFORM_POINTS),
      start_seconds: String(bounds.startSeconds),
      duration_seconds: String(bounds.endSeconds - bounds.startSeconds),
    });
    const [userResponse, assistantResponse] = await Promise.all([
      fetch(
        `/api/dataset-dashboard/waveform/${preview.sample_id}/${preview.user_side}?${parameters}`,
        { cache: "no-store" },
      ),
      fetch(
        `/api/dataset-dashboard/waveform/${preview.sample_id}/${preview.assistant_side}?${parameters}`,
        { cache: "no-store" },
      ),
    ]);
    const [userWaveform, assistantWaveform] = await Promise.all([
      userResponse.json(),
      assistantResponse.json(),
    ]);
    if (!userResponse.ok) {
      throw new Error(errorMessage(userWaveform, userResponse.status));
    }
    if (!assistantResponse.ok) {
      throw new Error(errorMessage(assistantWaveform, assistantResponse.status));
    }
    if (generation !== requestGeneration) {
      return;
    }
    waveforms = {
      startSeconds: bounds.startSeconds,
      endSeconds: bounds.endSeconds,
      user: userWaveform.points,
      assistant: assistantWaveform.points,
    };
    restoreLabel();
    draw();
  }

  function reset() {
    waveforms = null;
    requestGeneration += 1;
    draw();
  }

  function draw() {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const displayWidth = Math.max(980, canvas.clientWidth);
    const displayHeight = 220;
    canvas.width = Math.round(displayWidth * devicePixelRatio);
    canvas.height = Math.round(displayHeight * devicePixelRatio);
    const context = canvas.getContext("2d");
    context.scale(devicePixelRatio, devicePixelRatio);
    context.clearRect(0, 0, displayWidth, displayHeight);
    const preview = getPreview();
    if (preview === null || waveforms === null) {
      return;
    }
    const left = 112;
    const right = 18;
    const plotWidth = displayWidth - left - right;
    drawUnusableRegionOverlay(
      context,
      preview.conversation_regions,
      left,
      18,
      plotWidth,
      150,
      waveforms.startSeconds,
      waveforms.endSeconds,
    );
    const waveformRows = [
      {
        label: `USER · ${prettySide(preview.user_side)}`,
        points: waveforms.user,
        color: "#0057d9",
        gain: preview.user_gain.default_gain,
        spans: recordingSpeechSpans(preview, "user"),
      },
      {
        label: `ASSISTANT · ${prettySide(preview.assistant_side)}`,
        points: waveforms.assistant,
        color: "#7a1fa2",
        gain: preview.assistant_gain.default_gain,
        spans: recordingSpeechSpans(preview, "assistant"),
      },
    ];
    const displayScale = commonWaveformDisplayScale(waveformRows);
    waveformRows.forEach((row, rowIndex) => {
      const waveformTop = 24 + rowIndex * 80;
      context.fillStyle = "#48575c";
      context.font = "bold 11px sans-serif";
      context.fillText(row.label, 8, waveformTop + 26);
      drawWaveformPoints(
        context,
        row.points,
        left,
        waveformTop,
        plotWidth,
        42,
        row.color,
        row.gain * displayScale,
      );
      drawSpeechSpans(
        context,
        row.spans,
        left,
        waveformTop + 46,
        plotWidth,
        waveforms.startSeconds,
        waveforms.endSeconds,
      );
    });
    drawSelection(
      context,
      preview,
      getSelectedStartSeconds(),
      left,
      18,
      plotWidth,
      150,
      waveforms.startSeconds,
      waveforms.endSeconds,
    );
    drawAxis(
      context,
      left,
      plotWidth,
      188,
      waveforms.startSeconds,
      waveforms.endSeconds,
    );
  }

  function positionSelection(event) {
    const preview = getPreview();
    if (preview === null || waveforms === null) {
      return;
    }
    const rectangle = canvas.getBoundingClientRect();
    const left = 112;
    const right = 18;
    const plotWidth = rectangle.width - left - right;
    const x = Math.min(plotWidth, Math.max(0, event.clientX - rectangle.left - left));
    const timeSeconds =
      waveforms.startSeconds +
      (x / plotWidth) * (waveforms.endSeconds - waveforms.startSeconds);
    const maximumStartSeconds = Math.max(
      0,
      preview.eligible_duration_seconds - preview.input_duration_seconds,
    );
    const startSeconds = Math.min(
      maximumStartSeconds,
      Math.max(0, timeSeconds - preview.input_duration_seconds / 2),
    );
    setSelectedStartSeconds(startSeconds);
    draw();
  }

  function showInspection(event) {
    const preview = getPreview();
    if (preview === null || preview.conversation_regions === null) {
      return;
    }
    const timeSeconds = timeAtEvent(canvas, waveforms, event);
    if (timeSeconds === null) {
      return;
    }
    const region = preview.conversation_regions.unusable_regions.find(
      (candidate) =>
        candidate.start_seconds <= timeSeconds &&
        timeSeconds < candidate.end_seconds,
    );
    label.textContent =
      region === undefined
        ? `${formatDuration(timeSeconds)} · no permissive exclusion`
        : `${formatDuration(timeSeconds)} · ${region.reasons.map(formatRegionReason).join(", ")}`;
  }

  function restoreLabel() {
    if (waveforms === null) {
      return;
    }
    label.textContent =
      `${formatDuration(waveforms.startSeconds)}–` +
      `${formatDuration(waveforms.endSeconds)} · drag to reposition the 20-second crop`;
  }

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    canvas.setPointerCapture(event.pointerId);
    positionSelection(event);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging) {
      positionSelection(event);
    } else {
      showInspection(event);
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!dragging) {
      return;
    }
    dragging = false;
    canvas.releasePointerCapture(event.pointerId);
    positionSelection(event);
    commitSelection();
  });
  canvas.addEventListener("pointercancel", () => {
    dragging = false;
  });
  canvas.addEventListener("pointerleave", restoreLabel);

  return { load, reset, draw };
}

export function drawUnusableRegionOverlay(
  context,
  analysis,
  left,
  top,
  width,
  height,
  viewportStartSeconds,
  viewportEndSeconds,
) {
  if (analysis === null) {
    return;
  }
  const patternCanvas = document.createElement("canvas");
  patternCanvas.width = 8;
  patternCanvas.height = 8;
  const patternContext = patternCanvas.getContext("2d");
  patternContext.fillStyle = "rgba(242, 153, 74, 0.12)";
  patternContext.fillRect(0, 0, 8, 8);
  patternContext.strokeStyle = "rgba(202, 80, 16, 0.42)";
  patternContext.lineWidth = 2;
  patternContext.beginPath();
  patternContext.moveTo(-2, 8);
  patternContext.lineTo(8, -2);
  patternContext.moveTo(4, 10);
  patternContext.lineTo(10, 4);
  patternContext.stroke();
  context.fillStyle = context.createPattern(patternCanvas, "repeat");
  for (const region of analysis.unusable_regions) {
    const startSeconds = Math.max(viewportStartSeconds, region.start_seconds);
    const endSeconds = Math.min(viewportEndSeconds, region.end_seconds);
    if (endSeconds <= startSeconds) {
      continue;
    }
    const x =
      left +
      ((startSeconds - viewportStartSeconds) /
        (viewportEndSeconds - viewportStartSeconds)) *
        width;
    const regionWidth =
      ((endSeconds - startSeconds) /
        (viewportEndSeconds - viewportStartSeconds)) *
      width;
    context.fillRect(x, top, Math.max(1, regionWidth), height);
  }
}

function contextBounds(preview, selectedStartSeconds) {
  const durationSeconds = Math.min(
    CONTEXT_DURATION_SECONDS,
    preview.eligible_duration_seconds,
  );
  const selectionCenterSeconds =
    selectedStartSeconds + preview.input_duration_seconds / 2;
  const maximumStartSeconds = Math.max(
    0,
    preview.eligible_duration_seconds - durationSeconds,
  );
  const startSeconds = Math.min(
    maximumStartSeconds,
    Math.max(0, selectionCenterSeconds - durationSeconds / 2),
  );
  return {
    startSeconds,
    endSeconds: startSeconds + durationSeconds,
  };
}

function recordingSpeechSpans(preview, role) {
  return preview[`recording_${role}_spans`].filter(
    (span) => span.event_type === `${role}_speech`,
  );
}

function drawWaveformPoints(context, points, left, top, width, height, color, gain) {
  const middle = top + height / 2;
  const amplitudeScale = height * 0.43;
  context.fillStyle = "#f7f9f8";
  context.fillRect(left, top, width, height);
  context.strokeStyle = color;
  context.lineWidth = 2;
  points.forEach((point, index) => {
    const x = left + (index / Math.max(1, points.length - 1)) * width;
    context.beginPath();
    const maximumAmplitude = Math.min(1, point.maximum_amplitude * gain);
    const minimumAmplitude = Math.max(-1, point.minimum_amplitude * gain);
    context.moveTo(x, middle - maximumAmplitude * amplitudeScale);
    context.lineTo(x, middle - minimumAmplitude * amplitudeScale);
    context.stroke();
  });
}

function drawSpeechSpans(
  context,
  spans,
  left,
  top,
  width,
  viewportStartSeconds,
  viewportEndSeconds,
) {
  context.fillStyle = "rgba(20, 107, 99, 0.34)";
  spans.forEach((span) => {
    const startSeconds = Math.max(viewportStartSeconds, span.start_seconds);
    const endSeconds = Math.min(viewportEndSeconds, span.end_seconds);
    if (endSeconds <= startSeconds) {
      return;
    }
    const x =
      left +
      ((startSeconds - viewportStartSeconds) /
        (viewportEndSeconds - viewportStartSeconds)) *
        width;
    const spanWidth =
      ((endSeconds - startSeconds) /
        (viewportEndSeconds - viewportStartSeconds)) *
      width;
    context.fillRect(x, top, Math.max(1, spanWidth), 5);
  });
}

function drawSelection(
  context,
  preview,
  startSeconds,
  left,
  top,
  width,
  height,
  viewportStartSeconds,
  viewportEndSeconds,
) {
  const endSeconds = startSeconds + preview.input_duration_seconds;
  const x =
    left +
    ((startSeconds - viewportStartSeconds) /
      (viewportEndSeconds - viewportStartSeconds)) *
      width;
  const selectionWidth =
    (preview.input_duration_seconds /
      (viewportEndSeconds - viewportStartSeconds)) *
    width;
  context.fillStyle = "rgba(20, 107, 99, 0.12)";
  context.fillRect(x, top, selectionWidth, height);
  context.strokeStyle = "#146b63";
  context.lineWidth = 2;
  context.strokeRect(x, top, selectionWidth, height);
  context.fillStyle = "#146b63";
  context.font = "bold 11px sans-serif";
  context.fillText(
    `${formatDuration(startSeconds)}–${formatDuration(endSeconds)}`,
    Math.max(left + 4, x + 5),
    top + 15,
  );
  context.lineWidth = 1;
}

function drawAxis(
  context,
  left,
  width,
  top,
  viewportStartSeconds,
  viewportEndSeconds,
) {
  context.strokeStyle = "#cbd3d1";
  context.fillStyle = "#657378";
  context.font = "11px sans-serif";
  const durationSeconds = viewportEndSeconds - viewportStartSeconds;
  for (let offsetSeconds = 0; offsetSeconds <= durationSeconds; offsetSeconds += 15) {
    const x = left + (offsetSeconds / durationSeconds) * width;
    context.beginPath();
    context.moveTo(x, 174);
    context.lineTo(x, top - 2);
    context.stroke();
    context.fillText(formatDuration(viewportStartSeconds + offsetSeconds), x + 3, top + 12);
  }
}

function timeAtEvent(canvas, waveforms, event) {
  if (waveforms === null) {
    return null;
  }
  const rectangle = canvas.getBoundingClientRect();
  const left = 112;
  const right = 18;
  const plotWidth = rectangle.width - left - right;
  const x = event.clientX - rectangle.left - left;
  if (x < 0 || x > plotWidth) {
    return null;
  }
  return (
    waveforms.startSeconds +
    (x / plotWidth) * (waveforms.endSeconds - waveforms.startSeconds)
  );
}

function formatRegionReason(reason) {
  const labels = {
    dual_silence: "dual silence",
    one_sided_activity: "long one-sided activity",
    slow_turn_exchange: "slow turn exchange",
  };
  return labels[reason] ?? reason.replaceAll("_", " ");
}

function prettySide(side) {
  return side === "speaker1" ? "Speaker 1" : "Speaker 2";
}

function formatDuration(totalSeconds) {
  const bounded = Math.max(0, totalSeconds);
  const minutes = Math.floor(bounded / 60);
  const seconds = bounded - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}

function errorMessage(payload, statusCode) {
  return typeof payload.detail === "string" ? payload.detail : `Request failed (${statusCode})`;
}
