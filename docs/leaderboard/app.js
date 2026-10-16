"use strict";

const DATA_SOURCES = ["./scores.csv", "../../leaderboard/scores.csv"];
const NUMERIC_FIELDS = new Set(["rank", "global_f1", "user_f1", "text_f1"]);
const ADAPTATION_LABELS = {
  none: "None",
  train: "Training time",
  test: "Inference time",
};

const state = {
  rows: [],
  filters: {
    dataset: "",
    family: "",
    model: "",
  },
  sort: {
    key: "global_f1",
    direction: "desc",
  },
};

const elements = {
  body: document.querySelector("#leaderboard-body"),
  datasetFilter: document.querySelector("#dataset-filter"),
  familyFilter: document.querySelector("#family-filter"),
  modelFilter: document.querySelector("#model-filter"),
  resetButton: document.querySelector("#reset-filters"),
  resultsSummary: document.querySelector("#results-summary"),
  totalConfigurations: document.querySelector("#total-configurations"),
  emptyState: document.querySelector("#empty-state"),
  table: document.querySelector("#leaderboard-table"),
  sortButtons: [...document.querySelectorAll(".sort-button")],
};

function parseCsv(csvText) {
  const records = [];
  let record = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < csvText.length; index += 1) {
    const character = csvText[index];
    const nextCharacter = csvText[index + 1];

    if (character === '"' && quoted && nextCharacter === '"') {
      field += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      record.push(field);
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && nextCharacter === "\n") {
        index += 1;
      }
      record.push(field);
      if (record.some((value) => value.length > 0)) {
        records.push(record);
      }
      record = [];
      field = "";
    } else {
      field += character;
    }
  }

  if (field.length > 0 || record.length > 0) {
    record.push(field);
    records.push(record);
  }

  const [headers, ...values] = records;
  return values.map((row) =>
    Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ""])),
  );
}

async function loadRows() {
  const failures = [];

  for (const source of DATA_SOURCES) {
    try {
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      return parseCsv(await response.text());
    } catch (error) {
      failures.push(`${source}: ${error.message}`);
    }
  }

  throw new Error(`Unable to load scores.csv (${failures.join("; ")})`);
}

function enrichRows(rows) {
  const enriched = rows.map((row, sourceIndex) => ({
    ...row,
    sourceIndex,
    global_f1: Number(row.global_f1),
    user_f1: Number(row.user_f1),
    text_f1: Number(row.text_f1),
  }));

  const variants = new Map();
  enriched.forEach((row) => {
    const variantKey = [
      row.dataset,
      row.label,
      row.task,
      row.adaptation,
      row.extended,
    ].join("::");
    const group = variants.get(variantKey) ?? [];
    group.push(row);
    variants.set(variantKey, group);
  });

  variants.forEach((variantRows) => {
    variantRows
      .sort((left, right) => right.global_f1 - left.global_f1 || left.model.localeCompare(right.model))
      .forEach((row, index) => {
        row.rank = index + 1;
      });
  });

  return enriched;
}

function uniqueValues(field) {
  return [...new Set(state.rows.map((row) => row[field]))].sort((left, right) =>
    left.localeCompare(right),
  );
}

function populateSelect(select, values) {
  const fragment = document.createDocumentFragment();
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    fragment.append(option);
  });
  select.append(fragment);
}

function filteredRows() {
  return state.rows.filter((row) => {
    const datasetMatches = !state.filters.dataset || row.dataset === state.filters.dataset;
    const familyMatches = !state.filters.family || row.model_family === state.filters.family;
    const modelMatches = !state.filters.model || row.model === state.filters.model;
    return datasetMatches && familyMatches && modelMatches;
  });
}

function sortedRows(rows) {
  const direction = state.sort.direction === "asc" ? 1 : -1;
  const key = state.sort.key;

  return [...rows].sort((left, right) => {
    let comparison;
    if (NUMERIC_FIELDS.has(key)) {
      comparison = left[key] - right[key];
    } else {
      comparison = String(left[key]).localeCompare(String(right[key]));
    }

    return comparison * direction || left.sourceIndex - right.sourceIndex;
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function metricStyle(score) {
  const clampedScore = Math.max(0, Math.min(1, score));
  const hue = 31 + clampedScore * 112;
  const lightness = 23 + clampedScore * 4;
  return `background: hsl(${hue.toFixed(1)} 48% ${lightness.toFixed(1)}%)`;
}

function sourceTable(row) {
  if (row.model_family === "Encoder") {
    return row.dataset === "DICES" ? 4 : 3;
  }
  return row.dataset === "DICES" ? 6 : 5;
}

function rowTemplate(row) {
  const familyClass = row.model_family.toLowerCase();
  const representation = row.representation === "none" ? "—" : row.representation;
  const extended = row.extended === "true";
  const rankClass = row.rank === 1 ? " rank-badge--first" : "";

  return `
    <tr>
      <td class="rank-cell">
        <span class="rank-badge${rankClass}" title="Rank within this exact task variant">${row.rank}</span>
      </td>
      <td class="dataset-cell">${escapeHtml(row.dataset)}</td>
      <td class="label-cell">${escapeHtml(row.label)}</td>
      <td><span class="family-pill family-pill--${familyClass}">${escapeHtml(row.model_family)}</span></td>
      <td class="model-cell">${escapeHtml(row.model)}</td>
      <td class="task-cell">${escapeHtml(row.task[0].toUpperCase() + row.task.slice(1))}</td>
      <td class="adaptation-cell">${escapeHtml(ADAPTATION_LABELS[row.adaptation])}</td>
      <td><span class="boolean-pill${extended ? " boolean-pill--yes" : ""}">${extended ? "Yes" : "No"}</span></td>
      <td class="method-cell">${escapeHtml(row.method)}</td>
      <td class="representation-cell">${escapeHtml(representation)}</td>
      <td class="metric-cell" style="${metricStyle(row.global_f1)}">${row.global_f1.toFixed(3)}</td>
      <td class="metric-cell" style="${metricStyle(row.user_f1)}">${row.user_f1.toFixed(3)}</td>
      <td class="metric-cell" style="${metricStyle(row.text_f1)}">${row.text_f1.toFixed(3)}</td>
      <td>
        <a class="source-link" href="${escapeHtml(row.source)}" target="_blank" rel="noreferrer">
          Table ${sourceTable(row)} <span aria-hidden="true">↗</span>
        </a>
      </td>
    </tr>
  `;
}

function updateSortControls() {
  elements.sortButtons.forEach((button) => {
    const active = button.dataset.sort === state.sort.key;
    button.dataset.active = String(active);
    button.dataset.direction = active ? state.sort.direction : "";
    button.setAttribute("aria-pressed", String(active));
    const label = button.textContent.trim().replace(/[↕↑↓]/g, "").trim();
    button.setAttribute(
      "aria-label",
      active
        ? `${label}, sorted ${state.sort.direction === "asc" ? "ascending" : "descending"}`
        : `Sort by ${label}`,
    );
  });
}

function render() {
  const visibleRows = sortedRows(filteredRows());
  elements.body.innerHTML = visibleRows.map(rowTemplate).join("");
  elements.table.hidden = visibleRows.length === 0;
  elements.emptyState.hidden = visibleRows.length !== 0;
  elements.resultsSummary.innerHTML = `Showing <strong>${visibleRows.length}</strong> of <strong>${state.rows.length}</strong> configurations`;
  updateSortControls();
}

function bindEvents() {
  const filterBindings = [
    [elements.datasetFilter, "dataset"],
    [elements.familyFilter, "family"],
    [elements.modelFilter, "model"],
  ];

  filterBindings.forEach(([select, field]) => {
    select.addEventListener("change", () => {
      state.filters[field] = select.value;
      render();
    });
  });

  elements.resetButton.addEventListener("click", () => {
    state.filters = { dataset: "", family: "", model: "" };
    state.sort = { key: "global_f1", direction: "desc" };
    elements.datasetFilter.value = "";
    elements.familyFilter.value = "";
    elements.modelFilter.value = "";
    render();
  });

  elements.sortButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      if (state.sort.key === key) {
        state.sort.direction = state.sort.direction === "desc" ? "asc" : "desc";
      } else {
        state.sort.key = key;
        state.sort.direction = key === "rank" ? "asc" : "desc";
      }
      render();
    });
  });
}

function showLoadError(error) {
  elements.resultsSummary.textContent = "The leaderboard data could not be loaded.";
  elements.table.hidden = true;
  elements.emptyState.hidden = false;
  elements.emptyState.querySelector("h3").textContent = "Unable to load scores.csv";
  elements.emptyState.querySelector("p").textContent = error.message;
}

async function initialize() {
  try {
    state.rows = enrichRows(await loadRows());
    elements.totalConfigurations.textContent = state.rows.length;
    populateSelect(elements.datasetFilter, uniqueValues("dataset"));
    populateSelect(elements.familyFilter, uniqueValues("model_family"));
    populateSelect(elements.modelFilter, uniqueValues("model"));
    bindEvents();
    render();
  } catch (error) {
    showLoadError(error);
  }
}

initialize();
