export function drawAnnotationTimelineRow(options) {
  const { context, annotation, left, top, width, viewportStartSeconds, viewportEndSeconds } =
    options;
  const segmentTargets = annotation.segment_hypotheses ?? annotation.segment_targets ?? [];
  const connectionTargets =
    annotation.connection_hypotheses ?? annotation.connection_targets ?? [];
  const hasConfidenceTargets = segmentTargets.length > 0;

  context.fillStyle = "#f3f5f6";
  context.fillRect(left, top, width, 42);
  if (hasConfidenceTargets) {
    drawConnectionTargets(context, connectionTargets, options);
  }
  drawSpans(
    context,
    annotation.speech_segments ?? [],
    top + 8,
    26,
    "rgba(20, 107, 99, 0.18)",
    "rgba(20, 107, 99, 0.35)",
    options,
  );
  if (hasConfidenceTargets) {
    drawSegmentTargets(context, segmentTargets, options);
  } else {
    drawSpans(
      context,
      annotation.pause_spans ?? annotation.pauses ?? [],
      top + 11,
      20,
      "rgba(224, 173, 42, 0.5)",
      "rgba(156, 110, 0, 0.7)",
      options,
    );
    drawSpans(
      context,
      annotation.backchannel_spans ?? annotation.backchannels ?? [],
      top + 6,
      30,
      "rgba(126, 87, 194, 0.5)",
      "rgba(90, 54, 153, 0.75)",
      options,
    );
    drawInterruptions(
      context,
      annotation.interruption_events ?? annotation.interruptions ?? [],
      options,
    );
  }
  drawTurns(context, annotation.end_of_turn_events ?? annotation.turns ?? [], options);
}

function drawSegmentTargets(context, targets, options) {
  const { left, top, width, viewportStartSeconds, viewportEndSeconds } = options;
  targets.forEach((target) => {
    if (!overlaps(target.start_seconds, target.end_seconds, options)) {
      return;
    }
    const startSeconds = Math.max(target.start_seconds, viewportStartSeconds);
    const endSeconds = Math.min(target.end_seconds, viewportEndSeconds);
    const x = left + ratio(startSeconds, options) * width;
    const targetWidth = Math.max(
      1,
      ((endSeconds - startSeconds) / (viewportEndSeconds - viewportStartSeconds)) * width,
    );
    const keepPlayingDominates = target.keep_playing_confidence >= target.turn_confidence;
    context.fillStyle = keepPlayingDominates ? "#d8c9f0" : "#b9dcd8";
    context.strokeStyle = keepPlayingDominates ? "#6c3eb4" : "#146b63";
    context.lineWidth = target.evidence_source === "audio_activity" ? 2 : 1;
    context.fillRect(x, top + 3, targetWidth, 36);
    context.strokeRect(x, top + 3, targetWidth, 36);
    context.lineWidth = 1;
    if (target.evidence_source === "audio_activity") {
      context.strokeStyle = "#27343a";
      context.beginPath();
      context.moveTo(x, top + 3);
      context.lineTo(Math.min(x + 10, x + targetWidth), top + 13);
      context.moveTo(Math.max(x, x + targetWidth - 10), top + 29);
      context.lineTo(x + targetWidth, top + 39);
      context.stroke();
    }
    if (target.interruption_confidence > 0.02) {
      context.strokeStyle = "#ca5010";
      context.lineWidth = 1 + 3 * target.interruption_confidence;
      context.beginPath();
      context.moveTo(x, top + 1);
      context.lineTo(x, top + 41);
      context.stroke();
      context.lineWidth = 1;
    }
    if (targetWidth >= 62) {
      context.fillStyle = "#1e2528";
      context.font = "10px sans-serif";
      context.fillText(
        `${keepPlayingDominates ? "KEEP" : "TURN"} ${Math.max(target.keep_playing_confidence, target.turn_confidence).toFixed(2)}${target.evidence_source === "audio_activity" ? " · AUDIO" : ""}`,
        x + 4,
        top + 17,
      );
    }
    if (targetWidth >= 120) {
      context.fillText(
        `K ${target.keep_playing_confidence.toFixed(2)}  T ${target.turn_confidence.toFixed(2)}  I ${target.interruption_confidence.toFixed(2)}`,
        x + 4,
        top + 31,
      );
    }
  });
}

function drawConnectionTargets(context, targets, options) {
  const { left, top, width, viewportStartSeconds, viewportEndSeconds } = options;
  targets.forEach((target) => {
    if (!overlaps(target.earlier_end_seconds, target.later_start_seconds, options)) {
      return;
    }
    const startSeconds = Math.max(target.earlier_end_seconds, viewportStartSeconds);
    const endSeconds = Math.min(target.later_start_seconds, viewportEndSeconds);
    const x = left + ratio(startSeconds, options) * width;
    const targetWidth = Math.max(
      1,
      ((endSeconds - startSeconds) / (viewportEndSeconds - viewportStartSeconds)) * width,
    );
    context.fillStyle = `rgba(224, 173, 42, ${0.55 * target.pause_confidence})`;
    context.fillRect(x, top + 11, targetWidth, 20);
    context.strokeStyle = `rgba(20, 107, 99, ${0.15 + 0.8 * target.merge_confidence})`;
    context.lineWidth = 1 + 3 * target.merge_confidence;
    context.beginPath();
    context.moveTo(x, top + 35);
    context.lineTo(x + targetWidth, top + 35);
    context.stroke();
    context.lineWidth = 1;
    if (targetWidth >= 85) {
      context.fillStyle = "#5f4b0b";
      context.font = "10px sans-serif";
      context.fillText(
        `P ${target.pause_confidence.toFixed(2)}  M ${target.merge_confidence.toFixed(2)}`,
        x + 4,
        top + 23,
      );
    }
  });
}

function drawSpans(context, spans, top, height, fillStyle, strokeStyle, options) {
  const { left, width, viewportStartSeconds, viewportEndSeconds } = options;
  context.fillStyle = fillStyle;
  context.strokeStyle = strokeStyle;
  spans.forEach((span) => {
    if (!overlaps(span.start_seconds, span.end_seconds, options)) {
      return;
    }
    const startSeconds = Math.max(span.start_seconds, viewportStartSeconds);
    const endSeconds = Math.min(span.end_seconds, viewportEndSeconds);
    const x = left + ratio(startSeconds, options) * width;
    const spanWidth =
      ((endSeconds - startSeconds) / (viewportEndSeconds - viewportStartSeconds)) * width;
    context.fillRect(x, top, Math.max(1, spanWidth), height);
    context.strokeRect(x, top, Math.max(1, spanWidth), height);
  });
}

function drawInterruptions(context, interruptions, options) {
  const { left, top, width, viewportStartSeconds, viewportEndSeconds } = options;
  context.strokeStyle = "rgba(146, 50, 8, 0.95)";
  context.fillStyle = "rgba(202, 80, 16, 0.9)";
  context.lineWidth = 2;
  interruptions.forEach((interruption) => {
    if (interruption.time_seconds < viewportStartSeconds || interruption.time_seconds > viewportEndSeconds) {
      return;
    }
    const x = left + ratio(interruption.time_seconds, options) * width;
    context.beginPath();
    context.moveTo(x, top + 2);
    context.lineTo(x, top + 40);
    context.stroke();
    context.beginPath();
    context.moveTo(x, top + 2);
    context.lineTo(x - 4, top + 9);
    context.lineTo(x + 4, top + 9);
    context.closePath();
    context.fill();
  });
  context.lineWidth = 1;
}

function drawTurns(context, turns, options) {
  const { left, top, width, viewportStartSeconds, viewportEndSeconds } = options;
  context.strokeStyle = "#c2322d";
  turns.forEach((turn) => {
    if (turn.time_seconds < viewportStartSeconds || turn.time_seconds > viewportEndSeconds) {
      return;
    }
    const x = left + ratio(turn.time_seconds, options) * width;
    context.beginPath();
    context.moveTo(x, top - 2);
    context.lineTo(x, top + 48);
    context.stroke();
  });
}

function overlaps(startSeconds, endSeconds, options) {
  return endSeconds >= options.viewportStartSeconds && startSeconds <= options.viewportEndSeconds;
}

function ratio(seconds, options) {
  return (
    (seconds - options.viewportStartSeconds) /
    (options.viewportEndSeconds - options.viewportStartSeconds)
  );
}
