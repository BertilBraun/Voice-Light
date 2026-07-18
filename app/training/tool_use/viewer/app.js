"use strict";

const recordSelect = document.querySelector("#record-select");
const previousButton = document.querySelector("#previous-record");
const nextButton = document.querySelector("#next-record");
const chooseFileButton = document.querySelector("#choose-file");
const emptyChooseFileButton = document.querySelector("#empty-choose-file");
const fileInput = document.querySelector("#file-input");
const emptyState = document.querySelector("#empty-state");
const emptyMessage = document.querySelector("#empty-message");
const recordView = document.querySelector("#record-view");
const recordPosition = document.querySelector("#record-position");
const recordTitle = document.querySelector("#record-title");
const recordBadges = document.querySelector("#record-badges");
const reviewRejection = document.querySelector("#review-rejection");
const systemContent = document.querySelector("#system-content");
const conversation = document.querySelector("#conversation");
const rawRecord = document.querySelector("#raw-record");

let records = [];
let selectedIndex = 0;

function parseJsonLines(text) {
  return text
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, lineIndex) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Invalid JSON on line ${lineIndex + 1}: ${error.message}`);
      }
    });
}

function badge(text) {
  const element = document.createElement("span");
  element.className = "badge";
  element.textContent = text;
  return element;
}

function canonicalValueToText(value) {
  if (!value) {
    return "";
  }
  switch (value.kind) {
    case "string":
    case "number":
    case "integer":
    case "boolean":
      return String(value.value);
    case "null":
      return "null";
    case "array":
      return `[${value.items.map(canonicalValueToText).join(", ")}]`;
    case "object":
      return value.fields
        .map((field) => `${field.name}=${canonicalValueToText(field.value)}`)
        .join(", ");
    default:
      return JSON.stringify(value);
  }
}

function renderToolCall(call) {
  const element = document.createElement("div");
  element.className = "tool-call";

  const label = document.createElement("span");
  label.className = "tool-label";
  label.textContent = `Tool call · ${call.call_id}`;

  const body = document.createElement("span");
  const argumentsText = canonicalValueToText(call.arguments);
  body.textContent = `${call.tool_name}(${argumentsText})`;

  element.append(label, body);
  return element;
}

function renderAssistantMessage(message) {
  const element = document.createElement("div");
  element.className = "message assistant";

  const role = document.createElement("p");
  role.className = "role";
  role.textContent = "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.audible_text || "No audible text";
  for (const call of message.tool_calls) {
    bubble.append(renderToolCall(call));
  }

  element.append(role, bubble);
  return element;
}

function renderUserMessage(message) {
  const element = document.createElement("div");
  element.className = "message user";

  const role = document.createElement("p");
  role.className = "role";
  role.textContent = "User";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.text;

  element.append(role, bubble);
  return element;
}

function renderToolResult(message) {
  const element = document.createElement("div");
  element.className = `tool-result ${message.outcome.status}`;

  const label = document.createElement("span");
  label.className = "tool-label";
  label.textContent = `Tool result · ${message.outcome.status} · ${message.call_id}`;

  const body = document.createElement("span");
  body.textContent = message.outcome.content || message.outcome.message;

  element.append(label, body);
  return element;
}

function renderSystem(record) {
  systemContent.replaceChildren();
  const systemMessage = record.messages.find((message) => message.kind === "system");
  const instruction = document.createElement("div");
  instruction.textContent = systemMessage?.text || "No system instruction.";
  systemContent.append(instruction);

  for (const tool of record.tools) {
    const toolElement = document.createElement("div");
    toolElement.className = "tool-definition";
    toolElement.textContent = `${tool.name} — ${tool.description}`;
    systemContent.append(toolElement);
  }
}

function renderConversation(record) {
  conversation.replaceChildren();
  for (const message of record.messages) {
    switch (message.kind) {
      case "user":
        conversation.append(renderUserMessage(message));
        break;
      case "assistant":
        conversation.append(renderAssistantMessage(message));
        break;
      case "tool_result":
        conversation.append(renderToolResult(message));
        break;
      default:
        break;
    }
  }
}

function renderSelectedRecord() {
  if (!records.length) {
    emptyState.hidden = false;
    recordView.hidden = true;
    previousButton.disabled = true;
    nextButton.disabled = true;
    return;
  }

  const record = records[selectedIndex];
  const scenario = record.metadata.scenario;
  emptyState.hidden = true;
  recordView.hidden = false;
  recordPosition.textContent = `${selectedIndex + 1} of ${records.length}`;
  recordTitle.textContent = record.record_id;
  recordBadges.replaceChildren(
    badge(scenario.family.replaceAll("_", " ")),
    badge(scenario.tool_need.replaceAll("_", " ")),
    badge(`${scenario.user_turn_count} user turn${scenario.user_turn_count === 1 ? "" : "s"}`),
    badge(record.metadata.split.name),
  );
  if (record.review?.status === "rejected") {
    reviewRejection.textContent = `Rejected: ${record.review.reason}`;
    reviewRejection.hidden = false;
  } else {
    reviewRejection.textContent = "";
    reviewRejection.hidden = true;
  }

  renderSystem(record);
  renderConversation(record);
  rawRecord.textContent = JSON.stringify(record, null, 2);
  recordSelect.value = String(selectedIndex);
  previousButton.disabled = selectedIndex === 0;
  nextButton.disabled = selectedIndex === records.length - 1;
  window.location.hash = encodeURIComponent(record.record_id);
  window.scrollTo({ top: 0, behavior: "instant" });
}

function selectRecord(index) {
  selectedIndex = Math.max(0, Math.min(index, records.length - 1));
  renderSelectedRecord();
}

function setRecords(loadedRecords) {
  records = loadedRecords;
  recordSelect.replaceChildren();

  for (const [index, record] of records.entries()) {
    const option = document.createElement("option");
    const family = record.metadata.scenario.family.replaceAll("_", " ");
    option.value = String(index);
    option.textContent = `${String(index + 1).padStart(2, "0")} · ${record.record_id} · ${family}`;
    recordSelect.append(option);
  }

  const hashIdentifier = decodeURIComponent(window.location.hash.slice(1));
  const hashIndex = records.findIndex((record) => record.record_id === hashIdentifier);
  selectRecord(hashIndex >= 0 ? hashIndex : 0);
}

async function loadDefaultRecords() {
  try {
    const configuredPath = new URLSearchParams(window.location.search).get("data");
    const response = await fetch(configuredPath || "../records.jsonl", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    setRecords(parseJsonLines(await response.text()));
  } catch (error) {
    emptyMessage.textContent = `Could not load records.jsonl automatically (${error.message}).`;
    renderSelectedRecord();
  }
}

async function loadSelectedFile(file) {
  try {
    setRecords(parseJsonLines(await file.text()));
  } catch (error) {
    records = [];
    emptyMessage.textContent = error.message;
    renderSelectedRecord();
  }
}

previousButton.addEventListener("click", () => selectRecord(selectedIndex - 1));
nextButton.addEventListener("click", () => selectRecord(selectedIndex + 1));
recordSelect.addEventListener("change", () => selectRecord(Number(recordSelect.value)));
chooseFileButton.addEventListener("click", () => fileInput.click());
emptyChooseFileButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) {
    void loadSelectedFile(file);
  }
});
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
    return;
  }
  if (event.key === "ArrowLeft") {
    selectRecord(selectedIndex - 1);
  } else if (event.key === "ArrowRight") {
    selectRecord(selectedIndex + 1);
  }
});

void loadDefaultRecords();
