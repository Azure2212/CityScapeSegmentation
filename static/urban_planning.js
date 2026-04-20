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
const regionCards = document.getElementById("regionCards");
const classStats = document.getElementById("classStats");
const spatialFlags = document.getElementById("spatialFlags");
const warningList = document.getElementById("warningList");
const sceneProfile = document.getElementById("sceneProfile");
const relationList = document.getElementById("relationList");
const tagList = document.getElementById("tagList");
const comparisonPanel = document.getElementById("comparisonPanel");
const selectAll = document.getElementById("selectAll");
const selectNone = document.getElementById("selectNone");

let uploadedFile = null;

function overlayCheckboxes() {
  return [...document.querySelectorAll(".class-grid .class-toggle input[type='checkbox']")];
}

function comparisonCheckboxes() {
  return [...document.querySelectorAll(".compare-grid .compare-toggle input[type='checkbox']")];
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

function emptyCopy(message) {
  return `<p class="empty-copy">${message}</p>`;
}

function humanizeKey(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function renderPlaceholderState() {
  scoreGrid.innerHTML = "";
  groupBars.innerHTML = "";
  objectRows.innerHTML = "";
  regionCards.innerHTML = "";
  classStats.innerHTML = "";
  spatialFlags.innerHTML = "";
  warningList.innerHTML = "";
  sceneProfile.innerHTML = "";
  relationList.innerHTML = "";
  tagList.innerHTML = "";
  comparisonPanel.innerHTML = "";
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

  if (!topStats.length) {
    classStats.innerHTML = emptyCopy("No visible classes.");
    return;
  }

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

function renderWarnings(warnings) {
  if (!warnings.length) {
    warningList.innerHTML = emptyCopy("No analysis warnings for this scene.");
    return;
  }

  warningList.innerHTML = warnings
    .map((item) => `
      <article class="warning-item">
        <p class="warning-title">${humanizeKey(item.key)}</p>
        <p class="warning-copy">${item.detail}</p>
      </article>
    `)
    .join("");
}

function renderBandCards(bands) {
  return bands
    .map((band) => {
      const topClasses = band.top_classes.length
        ? `
          <ul class="band-class-list">
            ${band.top_classes.map((item) => `
              <li class="band-class-row">
                <span class="band-class-label">${item.label}</span>
                <strong class="band-class-value">${Number(item.percentage).toFixed(1)}%</strong>
              </li>
            `).join("")}
          </ul>
        `
        : `<p class="band-empty">No visible classes.</p>`;

      return `
        <article class="mini-card band-card">
          <div class="band-card-header">
            <p class="band-card-label">${band.label}</p>
            <span class="band-card-pill">${band.dominant_label || "No dominant class"}</span>
          </div>
          <div class="band-card-metric">
            <span class="band-card-metric-label">Dominant share</span>
            <span class="band-card-metric-value">${Number(band.dominant_percentage).toFixed(1)}%</span>
          </div>
          <div class="band-card-body">
            <p class="band-card-body-label">Top classes</p>
            ${topClasses}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderSceneProfile(layout) {
  const priors = Object.values(layout.priors)
    .map((item) => `
      <article class="state-card ${item.active ? "is-active" : "is-muted"}">
        <div class="mini-card-line">
          <span>${item.label}</span>
          <span>${Number(item.score).toFixed(1)}</span>
        </div>
        <p class="mini-copy">${item.detail}</p>
      </article>
    `)
    .join("");

  sceneProfile.innerHTML = `
    <article class="profile-hero">
      <p class="profile-kicker">Dominant layout</p>
      <h3>${layout.dominant_layout}</h3>
      <p class="profile-copy">Height-aware priors, band composition, and corridor openness are derived directly from the predicted mask.</p>
    </article>
    <div class="mini-section">
      <p class="mini-label">Vertical bands</p>
      <div class="band-grid">
        ${renderBandCards(layout.vertical_bands)}
      </div>
    </div>
    <div class="mini-section">
      <p class="mini-label">Horizontal bands</p>
      <div class="band-grid">
        ${renderBandCards(layout.horizontal_bands)}
      </div>
    </div>
    <div class="mini-section">
      <p class="mini-label">Street-scene priors</p>
      <div class="state-list">
        ${priors}
      </div>
    </div>
  `;
}

function renderRelations(flags) {
  const items = Object.values(flags);
  relationList.innerHTML = items
    .map((item) => `
      <article class="relation-card ${item.active ? "is-active" : "is-muted"}">
        <div class="mini-card-line">
          <span>${item.label}</span>
          <span class="state-pill">${item.active ? item.count : 0}</span>
        </div>
        <p class="mini-copy">${item.detail}</p>
      </article>
    `)
    .join("");
}

function renderTags(tags) {
  if (!tags.length) {
    tagList.innerHTML = emptyCopy("No strong scene tags for this image.");
    return;
  }

  tagList.innerHTML = tags
    .map((tag) => `
      <article class="tag-card">
        <h3>${tag.label}</h3>
        <p class="tag-copy">${tag.detail}</p>
        <div class="chip-row">
          ${tag.evidence.map((item) => `<span class="chip">${item}</span>`).join("")}
        </div>
      </article>
    `)
    .join("");
}

function renderRegionStats(regionStats) {
  if (!regionStats.classes.length) {
    regionCards.innerHTML = emptyCopy("No countable regions passed the area threshold.");
    return;
  }

  const summary = `
    <article class="profile-hero compact">
      <p class="profile-kicker">Approximate thing/stuff layer</p>
      <h3>${regionStats.total_regions} regions</h3>
      <p class="profile-copy">Connected components are used as approximate instances, not true panoptic objects.</p>
    </article>
  `;

  const cards = regionStats.classes
    .slice(0, 8)
    .map((item) => {
      const firstRegion = item.regions[0];
      const context = (firstRegion?.adjacent_stuff_classes || [])
        .map((adjacent) => `<span class="chip subtle">${adjacent.label} ${Number(adjacent.percentage).toFixed(1)}%</span>`)
        .join("");

      return `
        <article class="region-card">
          <div class="mini-card-line">
            <span>${item.label}</span>
            <span>${item.count} region(s)</span>
          </div>
          <p class="mini-copy">${Number(item.percentage).toFixed(1)}% area • ${item.dominant_vertical_band}/${item.dominant_horizontal_band}</p>
          <div class="chip-row">
            ${context || `<span class="chip subtle">No dominant stuff adjacency</span>`}
          </div>
        </article>
      `;
    })
    .join("");

  regionCards.innerHTML = summary + cards;
}

function renderHighlightList(title, items) {
  if (!items.length) {
    return "";
  }

  return `
    <div class="mini-section">
      <p class="mini-label">${title}</p>
      <div class="delta-list">
        ${items.map((item) => `
          <article class="mini-card">
            <div class="mini-card-line">
              <span>${item.label}</span>
              <span>${Number(item.range).toFixed(1)}%</span>
            </div>
            <p class="mini-copy">${item.min_label} ${Number(item.min_percentage).toFixed(1)}% vs ${item.max_label} ${Number(item.max_percentage).toFixed(1)}%</p>
          </article>
        `).join("")}
      </div>
    </div>
  `;
}

function renderModelComparison(comparison) {
  if (!comparison) {
    comparisonPanel.innerHTML = emptyCopy("Select comparison models to render cross-model agreement and disagreement.");
    return;
  }

  const comparedModels = comparison.models
    .map((item) => `
      <article class="mini-card">
        <div class="mini-card-line">
          <span>${item.label}</span>
          <span>${item.dominant_layout}</span>
        </div>
        <p class="mini-copy">${item.scene_tags.length ? item.scene_tags.join(", ") : "No strong tags"}</p>
      </article>
    `)
    .join("");

  const sharedTags = comparison.shared_tags.length
    ? `<div class="chip-row">${comparison.shared_tags.map((tag) => `<span class="chip">${tag.label}</span>`).join("")}</div>`
    : emptyCopy("No shared scene tags across the compared models.");

  const notes = comparison.disagreement_notes.length
    ? `<ul class="plain-list">${comparison.disagreement_notes.map((note) => `<li>${note}</li>`).join("")}</ul>`
    : emptyCopy("No disagreement notes.");

  const skipped = comparison.skipped_models.length
    ? `
      <div class="mini-section">
        <p class="mini-label">Skipped models</p>
        <ul class="plain-list">
          ${comparison.skipped_models.map((item) => `<li>${item.label}: ${item.reason}</li>`).join("")}
        </ul>
      </div>
    `
    : "";

  comparisonPanel.innerHTML = `
    <div class="mini-section">
      <p class="mini-label">Compared models</p>
      <div class="delta-list">
        ${comparedModels}
      </div>
    </div>
    <div class="mini-section">
      <p class="mini-label">Shared tags</p>
      ${sharedTags}
    </div>
    ${renderHighlightList("Class spreads", comparison.class_delta_highlights)}
    ${renderHighlightList("Group spreads", comparison.group_delta_highlights)}
    <div class="mini-section">
      <p class="mini-label">Disagreement notes</p>
      ${notes}
    </div>
    ${skipped}
  `;
}

function renderResults(data) {
  previewImage.src = `data:image/png;base64,${data.segmentation_image}`;
  previewImage.style.display = "block";
  emptyState.style.display = "none";

  summaryText.textContent = data.summary;
  renderScores(data.planning_scores);
  renderSpatialFlags(data.spatial_flags);
  renderWarnings(data.analysis_warnings || []);
  renderSceneProfile(data.layout_profile);
  renderRelations(data.relation_flags);
  renderTags(data.scene_tags || []);
  renderGroupBars(data.group_stats);
  renderObjectCounts(data.object_counts);
  renderRegionStats(data.region_stats);
  renderClassStats(data.class_stats);
  renderModelComparison(data.model_comparison || null);
}

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }

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

  comparisonCheckboxes()
    .filter((checkbox) => checkbox.checked)
    .forEach((checkbox) => form.append("compare_models[]", checkbox.value));

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
