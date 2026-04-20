const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const previewImage = document.getElementById("previewImage");
const emptyState = document.getElementById("emptyState");
const runButton = document.getElementById("runButton");
const statusText = document.getElementById("statusText");
const modelSelect = document.getElementById("modelSelect");
const summaryText = document.getElementById("summaryText");
const scoreGrid = document.getElementById("scoreGrid");
const groupBars = document.getElementById("groupBars");
const objectRows = document.getElementById("objectRows");
const classStats = document.getElementById("classStats");
const spatialFlags = document.getElementById("spatialFlags");
const selectAll = document.getElementById("selectAll");
const selectNone = document.getElementById("selectNone");

let uploadedFile = null;

function overlayCheckboxes() {
  return [...document.querySelectorAll(".class-toggle input[type='checkbox']")];
}

function setOverlaySelection(value) {
  overlayCheckboxes().forEach((checkbox) => {
    checkbox.checked = value;
  });
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function renderPlaceholderState() {
  scoreGrid.innerHTML = "";
  groupBars.innerHTML = "";
  objectRows.innerHTML = "";
  classStats.innerHTML = "";
  spatialFlags.innerHTML = "";
}

function renderScores(scores) {
  scoreGrid.innerHTML = Object.values(scores)
    .map((item) => `
      <article class="score-tile">
        <p class="score-value">${Number(item.score).toFixed(1)}</p>
        <p class="score-label">${item.label}</p>
        <p class="score-detail">${item.detail}</p>
      </article>
    `)
    .join("");
}

function renderGroupBars(groups) {
  groupBars.innerHTML = groups
    .map((group) => `
      <div class="bar-row">
        <div class="bar-meta">
          <span>${group.label}</span>
          <span>${Number(group.percentage).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${Math.min(group.percentage, 100)}%"></div>
        </div>
      </div>
    `)
    .join("");
}

function renderObjectCounts(counts) {
  objectRows.innerHTML = counts
    .map((item) => `
      <tr>
        <td>${item.label}</td>
        <td>${item.count}</td>
        <td>${Number(item.percentage).toFixed(2)}%</td>
      </tr>
    `)
    .join("");
}

function renderClassStats(stats) {
  const topStats = stats
    .filter((item) => item.pixels > 0)
    .sort((a, b) => b.percentage - a.percentage)
    .slice(0, 8);

  classStats.innerHTML = topStats
    .map((item) => `
      <div class="class-row">
        <div class="class-meta">
          <span>${item.label}</span>
          <span>${Number(item.percentage).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${Math.min(item.percentage, 100)}%"></div>
        </div>
      </div>
    `)
    .join("");
}

function renderSpatialFlags(flags) {
  spatialFlags.innerHTML = `
    <dt>Road-adjacent</dt>
    <dd>${flags.road_adjacent_active_mobility}</dd>
    <dt>Sidewalk-adjacent</dt>
    <dd>${flags.sidewalk_adjacent_active_mobility}</dd>
    <dt>Unclassified</dt>
    <dd>${flags.unclassified_active_mobility}</dd>
    <dt>Signal</dt>
    <dd>${flags.signal}</dd>
  `;
}

function renderResults(data) {
  previewImage.src = `data:image/png;base64,${data.segmentation_image}`;
  previewImage.style.display = "block";
  emptyState.style.display = "none";

  summaryText.textContent = data.summary;
  renderScores(data.planning_scores);
  renderGroupBars(data.group_stats);
  renderObjectCounts(data.object_counts);
  renderClassStats(data.class_stats);
  renderSpatialFlags(data.spatial_flags);
}

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  uploadedFile = file;
  fileName.textContent = file.name;
  setStatus("");

  const reader = new FileReader();
  reader.onload = (loadEvent) => {
    previewImage.src = loadEvent.target.result;
    previewImage.style.display = "block";
    emptyState.style.display = "none";
  };
  reader.readAsDataURL(file);
});

selectAll.addEventListener("click", () => setOverlaySelection(true));
selectNone.addEventListener("click", () => setOverlaySelection(false));

runButton.addEventListener("click", async () => {
  if (!uploadedFile) {
    setStatus("Choose an image before running analysis.", true);
    return;
  }

  const form = new FormData();
  form.append("image", uploadedFile);
  form.append("model", modelSelect.value);
  form.append("overlay_selection_sent", "1");
  overlayCheckboxes()
    .filter((checkbox) => checkbox.checked)
    .forEach((checkbox) => form.append("overlay_classes[]", checkbox.value));

  runButton.disabled = true;
  setStatus("Analyzing scene...");

  try {
    const response = await fetch("/urban-predict", {
      method: "POST",
      body: form,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server returned ${response.status}`);
    }

    renderResults(data);
    setStatus(`Analyzed with ${data.model_label}.`);
  } catch (error) {
    renderPlaceholderState();
    setStatus(error.message, true);
  } finally {
    runButton.disabled = false;
  }
});
