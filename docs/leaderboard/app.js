"use strict";

const DATA_SOURCES = ["./scores.csv", "../../leaderboard/scores.csv"];
const METRIC_FIELDS = ["global_f1", "user_f1", "text_f1"];
const NUMERIC_FIELDS = new Set(["rank", ...METRIC_FIELDS]);
const DATASET_ORDER = ["BREXIT", "DICES", "EPIC", "MD", "MHS"];
const VARIANT_ORDER = [
  "baseline::none",
  "named::none",
  "named::train",
  "unnamed::train",
  "named::test",
  "unnamed::test",
];
const TASK_LABELS = {
  baseline: "Baseline",
  named: "Named perspective",
  unnamed: "Unnamed perspective",
};
const ADAPTATION_LABELS = {
  none: "No adaptation",
  train: "Training-time adaptation",
  test: "Inference-time adaptation",
};

const state = {
  rows: [],
  groups: [],
  groupByKey: new Map(),
  visibleRows: [],
  view: "grouped",
  filters: {
    dataset: "",
    extended: "all",
    adaptation: "all",
    families: new Set(["Encoder", "Decoder"]),
    model: "",
    query: "",
  },
  sort: {
    key: "rank",
    direction: "asc",
  },
};

const elements = {
  body: document.querySelector("#leaderboard-body"),
  table: document.querySelector("#leaderboard-table"),
  togetherBody: document.querySelector("#together-leaderboard-body"),
  togetherTable: document.querySelector("#together-leaderboard-table"),
  datasetFilter: document.querySelector("#dataset-filter"),
  extendedButtons: [...document.querySelectorAll("#extended-filter button")],
  adaptationButtons: [...document.querySelectorAll("#adaptation-filter button")],
  viewButtons: [...document.querySelectorAll("#view-mode button")],
  familyInputs: [...document.querySelectorAll(".family-choice input")],
  modelFilter: document.querySelector("#model-filter"),
  searchFilter: document.querySelector("#search-filter"),
  resetButton: document.querySelector("#reset-filters"),
  copyButton: document.querySelector("#copy-results"),
  resultsSummary: document.querySelector("#results-summary"),
  totalConfigurations: document.querySelector("#total-configurations"),
  footerSummary: document.querySelector("#footer-summary"),
  emptyState: document.querySelector("#empty-state"),
  sortButtons: [...document.querySelectorAll(".sort-button")],
  toast: document.querySelector("#toast"),
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
  if (!headers) {
    throw new Error("scores.csv is empty");
  }

  return values.map((row) =>
    Object.fromEntries(headers.map((header, index) => [header.trim(), (row[index] ?? "").trim()])),
  );
}

async function loadRows() {
  const failures = [];

  for (const source of DATA_SOURCES) {
    try {
      const response = await fetch(source, { cache: "no-cache" });
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

function datasetPosition(dataset) {
  const position = DATASET_ORDER.indexOf(dataset);
  return position === -1 ? DATASET_ORDER.length : position;
}

function variantPosition(task, adaptation) {
  const position = VARIANT_ORDER.indexOf(`${task}::${adaptation}`);
  return position === -1 ? VARIANT_ORDER.length : position;
}

function compareGroups(left, right) {
  return (
    datasetPosition(left.dataset) - datasetPosition(right.dataset) ||
    left.label.localeCompare(right.label) ||
    Number(left.extended === "true") - Number(right.extended === "true") ||
    variantPosition(left.task, left.adaptation) - variantPosition(right.task, right.adaptation)
  );
}

function enrichRows(rows) {
  const enrichedRows = rows.map((row, sourceIndex) => ({
    ...row,
    sourceIndex,
    global_f1: Number(row.global_f1),
    user_f1: Number(row.user_f1),
    text_f1: Number(row.text_f1),
  }));

  const groupMap = new Map();
  enrichedRows.forEach((row) => {
    const key = [row.dataset, row.label, row.task, row.adaptation, row.extended].join("::");
    if (!groupMap.has(key)) {
      groupMap.set(key, {
        key,
        dataset: row.dataset,
        label: row.label,
        task: row.task,
        adaptation: row.adaptation,
        extended: row.extended,
        rows: [],
        ranges: {},
      });
    }
    groupMap.get(key).rows.push(row);
  });

  const groups = [...groupMap.values()].sort(compareGroups);
  groups.forEach((group) => {
    group.rows.sort(
      (left, right) =>
        right.global_f1 - left.global_f1 ||
        left.model.localeCompare(right.model) ||
        left.sourceIndex - right.sourceIndex,
    );

    group.rows.forEach((row) => {
      row.rank = 1 + group.rows.filter((candidate) => candidate.global_f1 > row.global_f1).length;
      row.groupKey = group.key;
    });

    METRIC_FIELDS.forEach((field) => {
      const values = group.rows.map((row) => row[field]);
      group.ranges[field] = {
        min: Math.min(...values),
        max: Math.max(...values),
      };
    });
  });

  return { rows: enrichedRows, groups };
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

function groupMatchesFilters(group) {
  const datasetMatches = !state.filters.dataset || group.dataset === state.filters.dataset;
  const extendedMatches =
    state.filters.extended === "all" || group.extended === state.filters.extended;
  const adaptationMatches =
    state.filters.adaptation === "all" || group.adaptation === state.filters.adaptation;
  return datasetMatches && extendedMatches && adaptationMatches;
}

function rowMatchesFilters(row) {
  const familyMatches = state.filters.families.has(row.model_family);
  const modelMatches = !state.filters.model || row.model === state.filters.model;
  const haystack = `${row.model} ${row.method} ${row.representation}`.toLowerCase();
  const queryMatches = !state.filters.query || haystack.includes(state.filters.query);
  return familyMatches && modelMatches && queryMatches;
}

function sortedRows(rows) {
  const { key, direction } = state.sort;
  const multiplier = direction === "asc" ? 1 : -1;

  return [...rows].sort((left, right) => {
    const comparison = NUMERIC_FIELDS.has(key)
      ? left[key] - right[key]
      : String(left[key]).localeCompare(String(right[key]));
    return comparison * multiplier || left.model.localeCompare(right.model) || left.sourceIndex - right.sourceIndex;
  });
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function sourceTable(row) {
  if (row.model_family === "Encoder") {
    return row.dataset === "DICES" ? 4 : 3;
  }
  return row.dataset === "DICES" ? 6 : 5;
}

function variantHeader(group, visibleCount) {
  const row = createElement("tr", "variant-row");
  const cell = createElement("th");
  cell.colSpan = 9;
  cell.scope = "rowgroup";

  const identity = createElement("div", "variant-identity");
  const dataset = createElement("span", "variant-dataset", group.dataset);
  const label = createElement("span", "variant-label", group.label);
  identity.append(dataset, label);

  const details = createElement("div", "variant-details");
  details.append(
    createElement("span", "variant-task", TASK_LABELS[group.task] ?? group.task),
    createElement("span", "variant-separator", "·"),
    createElement(
      "span",
      "variant-adaptation",
      ADAPTATION_LABELS[group.adaptation] ?? group.adaptation,
    ),
    createElement("span", "variant-separator", "·"),
    createElement(
      "span",
      group.extended === "true" ? "variant-extended variant-extended--yes" : "variant-extended",
      group.extended === "true" ? "Extended training" : "Standard training",
    ),
  );

  const countText =
    visibleCount === group.rows.length
      ? `${group.rows.length} ${group.rows.length === 1 ? "result" : "results"}`
      : `${visibleCount} of ${group.rows.length} results`;
  const count = createElement("span", "variant-count", countText);

  const content = createElement("div", "variant-content");
  const main = createElement("div", "variant-main");
  main.append(identity, details);
  content.append(main, count);
  cell.append(content);
  row.append(cell);
  return row;
}

function relativeHeatStyle(value, range, singleConfiguration = false) {
  if (singleConfiguration) {
    return { hue: 142, equal: false };
  }
  if (range.max === range.min) {
    return { hue: 77, equal: true };
  }
  const normalized = (value - range.min) / (range.max - range.min);
  return { hue: 8 + normalized * 134, equal: false };
}

function metricCell(row, group, field) {
  const range = group.ranges[field];
  const heat = relativeHeatStyle(row[field], range, group.rows.length === 1);
  const isBest = Math.abs(row[field] - range.max) < 1e-12;
  const cell = createElement(
    "td",
    `metric-cell${isBest ? " metric-cell--best" : ""}${heat.equal ? " metric-cell--equal" : ""}`,
  );
  cell.style.setProperty("--heat-hue", heat.hue.toFixed(1));
  cell.title = isBest ? `Best ${field.replace("_", " ")} in this variant` : "";

  const value = createElement("span", "metric-value", row[field].toFixed(3));
  cell.append(value);
  if (isBest) {
    cell.append(createElement("span", "sr-only", " (best in this variant)"));
  }
  return cell;
}

function resultRow(row, group) {
  const tableRow = createElement("tr", "result-row");

  const rankCell = createElement("td", "rank-cell");
  const rankBadge = createElement(
    "span",
    `rank-badge${row.rank === 1 ? " rank-badge--first" : ""}`,
    String(row.rank),
  );
  rankBadge.title = "Global F1 rank within the complete variant";
  rankCell.append(rankBadge);

  const familyCell = createElement("td");
  const family = createElement(
    "span",
    `family-pill family-pill--${row.model_family.toLowerCase()}`,
    row.model_family,
  );
  familyCell.append(family);

  const modelCell = createElement("td", "model-cell", row.model);
  const methodCell = createElement("td", "method-cell", row.method);
  const representationCell = createElement(
    "td",
    "representation-cell",
    row.representation === "none" ? "—" : row.representation,
  );

  const sourceCell = createElement("td", "source-cell");
  const source = createElement("a", "source-link");
  source.href = row.source;
  source.target = "_blank";
  source.rel = "noreferrer";
  source.append(
    document.createTextNode(`Table ${sourceTable(row)} `),
    createElement("span", "source-arrow", "↗"),
  );
  sourceCell.append(source);

  tableRow.append(
    rankCell,
    familyCell,
    modelCell,
    methodCell,
    representationCell,
    ...METRIC_FIELDS.map((field) => metricCell(row, group, field)),
    sourceCell,
  );
  return tableRow;
}

function togetherResultRow(row, group) {
  const tableRow = createElement("tr", "result-row together-row");

  const rankCell = createElement("td", "rank-cell");
  const rankBadge = createElement(
    "span",
    `rank-badge${row.rank === 1 ? " rank-badge--first" : ""}`,
    String(row.rank),
  );
  rankBadge.title = "Global F1 rank within the complete variant";
  rankCell.append(rankBadge);

  const datasetCell = createElement("td", "dataset-cell", row.dataset);
  const labelCell = createElement("td", "label-cell", row.label);

  const familyCell = createElement("td");
  const family = createElement(
    "span",
    `family-pill family-pill--${row.model_family.toLowerCase()}`,
    row.model_family,
  );
  familyCell.append(family);

  const modelCell = createElement("td", "model-cell", row.model);
  const taskCell = createElement("td", "task-cell", TASK_LABELS[row.task] ?? row.task);
  const adaptationCell = createElement(
    "td",
    "adaptation-cell",
    ADAPTATION_LABELS[row.adaptation] ?? row.adaptation,
  );

  const extendedCell = createElement("td");
  extendedCell.append(
    createElement(
      "span",
      row.extended === "true" ? "extended-pill extended-pill--yes" : "extended-pill",
      row.extended === "true" ? "Yes" : "No",
    ),
  );

  const methodCell = createElement("td", "method-cell", row.method);
  const representationCell = createElement(
    "td",
    "representation-cell",
    row.representation === "none" ? "—" : row.representation,
  );

  const sourceCell = createElement("td", "source-cell");
  const source = createElement("a", "source-link");
  source.href = row.source;
  source.target = "_blank";
  source.rel = "noreferrer";
  source.append(
    document.createTextNode(`Table ${sourceTable(row)} `),
    createElement("span", "source-arrow", "↗"),
  );
  sourceCell.append(source);

  tableRow.append(
    rankCell,
    datasetCell,
    labelCell,
    familyCell,
    modelCell,
    taskCell,
    adaptationCell,
    extendedCell,
    methodCell,
    representationCell,
    ...METRIC_FIELDS.map((field) => metricCell(row, group, field)),
    sourceCell,
  );
  return tableRow;
}

function updateSortControls() {
  const sortScope =
    state.view === "grouped" ? "inside each variant" : "across all visible configurations";
  elements.sortButtons.forEach((button) => {
    const active = button.dataset.sort === state.sort.key;
    button.dataset.active = String(active);
    button.dataset.direction = active ? state.sort.direction : "";
    button.setAttribute("aria-pressed", String(active));
    const label = button.textContent.trim();
    button.setAttribute(
      "aria-label",
      active
        ? `${label}, sorted ${state.sort.direction === "asc" ? "ascending" : "descending"} ${sortScope}`
        : `Sort ${sortScope} by ${label}`,
    );
  });
}

function render() {
  const matchingGroups = [];

  state.groups.forEach((group) => {
    if (!groupMatchesFilters(group)) {
      return;
    }

    const matchingRows = group.rows.filter(rowMatchesFilters);
    if (matchingRows.length === 0) {
      return;
    }

    matchingGroups.push({ group, rows: matchingRows });
  });

  const visibleRows = [];
  const groupedFragment = document.createDocumentFragment();
  const togetherFragment = document.createDocumentFragment();

  if (state.view === "grouped") {
    matchingGroups.forEach(({ group, rows }) => {
      const orderedRows = sortedRows(rows);
      groupedFragment.append(variantHeader(group, orderedRows.length));
      orderedRows.forEach((row) => {
        groupedFragment.append(resultRow(row, group));
        visibleRows.push(row);
      });
    });
  } else {
    const orderedRows = sortedRows(matchingGroups.flatMap(({ rows }) => rows));
    orderedRows.forEach((row) => {
      togetherFragment.append(togetherResultRow(row, state.groupByKey.get(row.groupKey)));
      visibleRows.push(row);
    });
  }

  elements.body.replaceChildren(groupedFragment);
  elements.togetherBody.replaceChildren(togetherFragment);
  state.visibleRows = visibleRows;
  elements.table.hidden = visibleRows.length === 0 || state.view !== "grouped";
  elements.togetherTable.hidden = visibleRows.length === 0 || state.view !== "together";
  elements.emptyState.hidden = visibleRows.length !== 0;
  elements.copyButton.disabled = visibleRows.length === 0;

  const visibleGroups = matchingGroups.length;
  const groupWord = visibleGroups === 1 ? "variant" : "variants";
  const viewLabel = state.view === "grouped" ? "grouped view" : "all-together view";
  elements.resultsSummary.innerHTML =
    `Showing <strong>${visibleRows.length}</strong> of <strong>${state.rows.length}</strong> ` +
    `configurations across <strong>${visibleGroups}</strong> ${groupWord} · ${viewLabel}`;
  updateSortControls();
}

function resetFilters() {
  state.filters.dataset = "";
  state.filters.extended = "all";
  state.filters.adaptation = "all";
  state.filters.families = new Set(["Encoder", "Decoder"]);
  state.filters.model = "";
  state.filters.query = "";
  state.sort =
    state.view === "grouped"
      ? { key: "rank", direction: "asc" }
      : { key: "global_f1", direction: "desc" };

  elements.datasetFilter.value = "";
  elements.modelFilter.value = "";
  elements.searchFilter.value = "";
  elements.extendedButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.value === "all"));
  });
  elements.adaptationButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.value === "all"));
  });
  elements.familyInputs.forEach((input) => {
    input.checked = true;
  });
  render();
}

function bindEvents() {
  elements.datasetFilter.addEventListener("change", () => {
    state.filters.dataset = elements.datasetFilter.value;
    render();
  });

  elements.extendedButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.filters.extended = button.dataset.value;
      elements.extendedButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  elements.adaptationButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.filters.adaptation = button.dataset.value;
      elements.adaptationButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  elements.viewButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.value;
      state.sort =
        state.view === "grouped"
          ? { key: "rank", direction: "asc" }
          : { key: "global_f1", direction: "desc" };
      elements.viewButtons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      render();
    });
  });

  elements.familyInputs.forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) {
        state.filters.families.add(input.value);
      } else {
        state.filters.families.delete(input.value);
      }
      render();
    });
  });

  elements.modelFilter.addEventListener("change", () => {
    state.filters.model = elements.modelFilter.value;
    render();
  });

  elements.searchFilter.addEventListener("input", () => {
    state.filters.query = elements.searchFilter.value.trim().toLowerCase();
    render();
  });

  elements.resetButton.addEventListener("click", resetFilters);

  elements.sortButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      if (state.sort.key === key) {
        state.sort.direction = state.sort.direction === "asc" ? "desc" : "asc";
      } else {
        state.sort.key = key;
        state.sort.direction = key === "rank" ? "asc" : "desc";
      }
      render();
    });
  });

  elements.copyButton.addEventListener("click", copyFilteredCsv);
}

function csvCell(value) {
  const stringValue = String(value);
  return /[",\n]/.test(stringValue) ? `"${stringValue.replaceAll('"', '""')}"` : stringValue;
}

async function copyFilteredCsv() {
  const headers = [
    "dataset",
    "label",
    "model_family",
    "model",
    "task",
    "adaptation",
    "extended",
    "method",
    "representation",
    "global_f1",
    "user_f1",
    "text_f1",
    "source",
  ];
  const csv = [
    headers.join(","),
    ...state.visibleRows.map((row) =>
      headers
        .map((header) =>
          csvCell(METRIC_FIELDS.includes(header) ? row[header].toFixed(3) : row[header]),
        )
        .join(","),
    ),
  ].join("\n");

  try {
    await navigator.clipboard.writeText(csv);
    showToast(`${state.visibleRows.length} filtered rows copied`);
  } catch (_error) {
    const textArea = createElement("textarea", "clipboard-fallback");
    textArea.value = csv;
    textArea.setAttribute("readonly", "");
    document.body.append(textArea);
    textArea.select();
    const copied = document.execCommand("copy");
    textArea.remove();
    showToast(copied ? `${state.visibleRows.length} filtered rows copied` : "Copy failed");
  }
}

let toastTimer;
function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 2200);
}

function showLoadError(error) {
  elements.resultsSummary.textContent = "The leaderboard data could not be loaded.";
  elements.table.hidden = true;
  elements.togetherTable.hidden = true;
  elements.emptyState.hidden = false;
  elements.copyButton.disabled = true;
  elements.emptyState.querySelector("h3").textContent = "Unable to load scores.csv";
  elements.emptyState.querySelector("p").textContent = error.message;
}

async function initialize() {
  try {
    const { rows, groups } = enrichRows(await loadRows());
    state.rows = rows;
    state.groups = groups;
    state.groupByKey = new Map(groups.map((group) => [group.key, group]));
    elements.totalConfigurations.textContent = rows.length;
    elements.footerSummary.textContent =
      `${rows.length} results · ${uniqueValues("dataset").length} datasets · ${uniqueValues("model").length} models`;
    populateSelect(elements.datasetFilter, uniqueValues("dataset"));
    populateSelect(elements.modelFilter, uniqueValues("model"));
    bindEvents();
    render();
  } catch (error) {
    showLoadError(error);
  }
}

initialize();
