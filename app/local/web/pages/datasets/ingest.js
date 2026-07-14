const form = document.querySelector("#ingestion-form");
const statusElement = document.querySelector("#status");
const errorOutput = document.querySelector("#error-output");
const jobsList = document.querySelector("#jobs-list");
const refreshJobsButton = document.querySelector("#refresh-jobs");

form.addEventListener("submit", submitIngestion);
refreshJobsButton.addEventListener("click", loadJobs);

loadJobs();
setInterval(loadJobs, 5000);

async function submitIngestion(event) {
  event.preventDefault();
  setError("");
  statusElement.textContent = "Queueing ingestion";
  const payload = {
    dataset_name: document.querySelector("#dataset-name").value.trim(),
    root_path: document.querySelector("#root-path").value.trim(),
    layout: document.querySelector("#dataset-layout").value,
    max_workers: Number(document.querySelector("#max-workers").value),
    max_duration_hours: optionalNumber("#max-duration-hours"),
  };
  try {
    await postJson("/api/dataset-dashboard/ingest/local", payload);
    statusElement.textContent = "Ingestion queued";
    await loadJobs();
  } catch (error) {
    statusElement.textContent = "Ingestion request failed";
    setError(error.message);
  }
}

function optionalNumber(selector) {
  const value = document.querySelector(selector).value.trim();
  return value ? Number(value) : null;
}

async function loadJobs() {
  try {
    const payload = await fetchJson("/api/dataset-dashboard/jobs");
    renderJobs(payload.jobs);
  } catch (error) {
    setError(error.message);
  }
}

function renderJobs(jobs) {
  jobsList.replaceChildren();
  if (jobs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "job-meta";
    empty.textContent = "No ingestion jobs yet";
    jobsList.appendChild(empty);
    return;
  }
  for (const job of jobs) {
    const row = document.createElement("div");
    row.className = "job-row";

    const status = document.createElement("div");
    status.className = "job-status";
    status.textContent = job.status;

    const main = document.createElement("div");
    main.className = "job-main";
    const message = document.createElement("div");
    message.className = "job-message";
    message.textContent = job.message || job.source_uri;
    const meta = document.createElement("div");
    meta.className = "job-meta";
    meta.textContent = `${job.source_uri} - updated ${formatDate(job.updated_at)}`;
    main.append(message, meta);
    if (job.error) {
      const error = document.createElement("div");
      error.className = "job-error";
      error.textContent = job.error;
      main.appendChild(error);
    }

    const counts = document.createElement("div");
    counts.className = "job-counts";
    counts.textContent = `${job.processed_samples}/${job.total_samples} processed, ${job.failed_samples} failed`;

    row.append(status, main, counts);
    jobsList.appendChild(row);
  }
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

function setError(message) {
  if (!message) {
    errorOutput.hidden = true;
    errorOutput.textContent = "";
    return;
  }
  errorOutput.hidden = false;
  errorOutput.textContent = message;
}

function formatDate(value) {
  if (!value) {
    return "?";
  }
  return new Date(value).toLocaleString();
}
