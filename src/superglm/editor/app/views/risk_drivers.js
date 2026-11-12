// @ts-check
/**
 * What explains a risk on the Explain tab: a walk to the selected risk, one feature at a time. It
 * starts at risk A, the first risk held, or at the reference risk when risk A is the one selected.
 * A step is how far its feature alone moves a parameter's link predictor, read as the multiplier
 * that puts on the parameter ("×"), or, where the parameter's terms are shown as added amounts,
 * as the amount added on its link scale ("+"). What the features add only together, through the
 * model's interactions, is a step of its own.
 *
 * `driversModel` is pure and holds every number the walk shows; `renderDrivers` draws it.
 */

import { formatAmount } from "../distribution/line.js";
import { fmtSignificant } from "../format.js";

/** @typedef {import('../api/contracts.js').RiskResponse} RiskResponse */
/** @typedef {import('../api/contracts.js').RiskId} RiskId */
/**
 * A figure in force and, where the session's edits moved it, the original model's.
 * @typedef {{text: string, was: string | null}} DriverValue
 */
/**
 * A step's multiplier, or its added amount. `move` is the step on the parameter's link scale (the
 * log of the quantity shown, for a multiplier), which a bar draws; `text` is empty where the
 * feature does not act on the parameter.
 * @typedef {DriverValue & {move: number}} DriverStep
 */
/**
 * @typedef {object} DriverRow
 * @property {string} label  the feature, or what the features add together
 * @property {string} detail  its value at the start and at the end
 * @property {DriverStep[]} steps  one per parameter
 */
/** @typedef {{label: string, swatch: RiskId | null, values: DriverValue[]}} DriverEnd */
/**
 * @typedef {object} DriversModel
 * @property {string} title  where the walk starts and ends
 * @property {RiskId} id  the risk the walk ends at, whose colour the bars take
 * @property {string[]} columns  the labels of the parameters some feature acts on; a parameter
 *   that is one constant, such as a Tweedie's power, has no column
 * @property {DriverEnd} start
 * @property {DriverRow[]} rows  one per feature that leaves the start's value, in the features' order
 * @property {DriverRow | null} together  what the interactions add, where it shows
 * @property {DriverEnd} end
 * @property {string | null} empty  the one line that stands for the walk where nothing differs
 *   from its start
 * @property {number[]} reach  per parameter, the step a full bar stands for
 */

// A bar is full at a quarter up or down, or at the largest step drawn where that is larger.
const MIN_REACH = Math.log(1.25);

/** @param {number | string} value a feature's value as the view writes it */
export const shownValue = (value) =>
  typeof value === "number" ? String(Number(value.toPrecision(4))) : value;

/** @param {number} move */
const times = (move) => `×${fmtSignificant(Math.exp(move), 3)}`;
/** @param {number} move */
const plus = (move) => `${move < 0 ? "−" : "+"}${fmtSignificant(Math.abs(move), 3)}`;

/** @param {RiskResponse["parameters"][number]} parameter */
const isAdditive = (parameter) => parameter.display?.kind === "additive";

/**
 * The quantity the walk shows for a parameter at a link value, the one its steps act on: the
 * natural parameter through its link (above its floor for a floored log, between the walls for a
 * bounded logit), or for an overdispersion 1/theta. A link the browser does not invert gives NaN,
 * which the walk leaves blank.
 * @param {RiskResponse} reply @param {number} index the parameter's place in `parameters`
 * @param {number} eta
 */
export function parameterValue(reply, index, eta) {
  const { link, display } = reply.parameters[index];
  if (display?.kind === "inverse_theta") return Math.exp(-eta);
  switch (link.kind) {
    case "bounded_logit":
      return link.lower + (link.upper - link.lower) / (1 + Math.exp(-eta));
    case "logit":
      return 1 / (1 + Math.exp(-eta));
    case "identity":
      return eta;
    case "floored_log":
      return link.floor + Math.exp(eta);
    case "log":
      return Math.exp(eta);
    default:
      return Number.NaN;
  }
}

/**
 * A walk column's heading: what its figures are the value of, which for a log-normal's CV is
 * sigma, the standard deviation of the log amount its steps multiply.
 * @param {RiskResponse["parameters"][number]} parameter
 */
const columnLabel = (parameter) => parameter.value_label ?? parameter.label;

/**
 * What the walk says where no feature differs from its start.
 * @param {string | null} from the first risk's letter, or null where the walk starts at the reference risk
 * @param {string} to
 */
function nothingMoves(from, to) {
  return from === null
    ? `Risk ${to} is the reference risk. Change a value or add a risk to see what moves it.`
    : `Risk ${to} has the same values as risk ${from}. Change a value to see what moves it.`;
}

/**
 * The walk to the selected risk: from risk A, or from the reference risk when risk A, the first
 * held, is the one selected.
 * @param {RiskResponse} reply @param {RiskId} [selected] the first risk where the reply lacks it
 * @returns {DriversModel}
 */
export function driversModel(reply, selected) {
  const first = reply.risks[0];
  const last = reply.risks.find((risk) => risk.id === selected) ?? first;
  const base = last === first ? null : first;
  // Each parameter keeps its place in the predictors; `at` is where the walk's columns read.
  const at = reply.parameters.flatMap((parameter, p) =>
    reply.features.some((feature) => parameter.name in feature.terms) ? [p] : []
  );
  const parameters = at.map((p) => reply.parameters[p]);
  /** @param {RiskResponse["risks"][number]} risk */
  const letter = (risk) => (risk.id ?? "a").toUpperCase();
  // A higher theta is less overdispersion, so its steps turn over with the 1/theta shown.
  const signs = parameters.map((parameter) => (parameter.display?.kind === "inverse_theta" ? -1 : 1));
  /** @param {number} p @param {number} move */
  const stepText = (p, move) => (isAdditive(parameters[p]) ? plus(move) : times(move));
  /** @param {'fitted' | 'edited'} which the model the walk is read under */
  const walk = (which) => {
    const from = at.map((p) => (base ? base.eta[which] : reply.reference[which])[p]);
    const to = at.map((p) => last.eta[which][p]);
    const steps = reply.features.map((_, j) =>
      at.map((p) => last.effects[which][j][p] - (base ? base.effects[which][j][p] : 0))
    );
    const together = parameters.map(
      (_p, p) => to[p] - from[p] - steps.reduce((sum, step) => sum + step[p], 0)
    );
    return { from, to, steps, together };
  };
  const inForce = walk("edited");
  const original = walk("fitted");

  /** @param {number} p @param {number} eta */
  const figure = (p, eta) => {
    const value = parameterValue(reply, at[p], eta);
    if (!Number.isFinite(value)) return "";
    return parameters[p].role === "average" ? formatAmount(value, reply.amount_unit) : fmtSignificant(value, 3);
  };
  /** @param {string} text @param {string} was @returns {DriverValue} */
  const moved = (text, was) => ({ text, was: was === text || was === "" ? null : was });
  /** @param {number[]} now @param {number[]} before */
  const values = (now, before) => parameters.map((_p, p) => moved(figure(p, now[p]), figure(p, before[p])));
  /** @param {number[]} now @param {number[]} before @param {(p: number) => boolean} acts */
  const steps = (now, before, acts) =>
    parameters.map((_p, p) => ({
      move: signs[p] * now[p],
      ...moved(
        acts(p) || now[p] !== 0 ? stepText(p, signs[p] * now[p]) : "",
        acts(p) || before[p] !== 0 ? stepText(p, signs[p] * before[p]) : ""
      ),
    }));

  const rows = reply.features.flatMap((feature, j) => {
    const from = base ? base.values[feature.column] : feature.reference;
    const to = last.values[feature.column];
    if (to === from) return [];
    return [{
      label: feature.column,
      detail: `${shownValue(from)} → ${shownValue(to)}`,
      steps: steps(inForce.steps[j], original.steps[j], (p) => parameters[p].name in feature.terms),
    }];
  });
  const jointly = steps(inForce.together, original.together, () => true);
  const together = jointly.some((step, p) => step.text !== stepText(p, 0) || step.was !== null)
    ? { label: "Together", detail: "interactions", steps: jointly }
    : null;
  const drawn = [...rows, ...(together ? [together] : [])];
  const id = last.id ?? "a";
  const end = { label: letter(last), swatch: id, values: values(inForce.to, original.to) };
  const start = base
    ? { label: letter(base), swatch: base.id ?? "a", values: values(inForce.from, original.from) }
    : { label: "Reference", swatch: null, values: values(inForce.from, original.from) };
  const edited = [...start.values, ...end.values].some((value) => value.was !== null);
  return {
    title: `${base ? letter(base) : "Reference"} → ${letter(last)}`,
    id,
    columns: parameters.map(columnLabel),
    start,
    rows,
    together,
    end,
    empty: drawn.length || edited ? null : nothingMoves(base ? letter(base) : null, letter(last)),
    reach: parameters.map((_p, p) => Math.max(MIN_REACH, ...drawn.map((row) => Math.abs(row.steps[p].move)))),
  };
}

/**
 * Draws the walk as a table: the start, a row and a bar per feature, and the end.
 * @param {HTMLElement} host @param {DriversModel} model
 */
export function renderDrivers(host, model) {
  const doc = host.ownerDocument;
  /**
   * @template {keyof HTMLElementTagNameMap} K
   * @param {K} tag @param {string} cls @param {string} [text]
   * @returns {HTMLElementTagNameMap[K]}
   */
  const node = (tag, cls, text) => {
    const made = doc.createElement(tag);
    if (cls) made.className = cls;
    if (text !== undefined) made.textContent = text;
    return made;
  };
  /** @param {DriverValue} value */
  const figureCell = (value) => {
    const cell = node("td", "", value.text);
    if (value.was !== null) cell.append(node("span", "risk-drivers-was", `was ${value.was}`));
    return cell;
  };
  /** @param {DriverEnd} end */
  const endRow = (end) => {
    const row = node("tr", "risk-drivers-end");
    const head = node("th", "");
    head.scope = "row";
    if (end.swatch) {
      const swatch = node("i", "risk-swatch");
      swatch.dataset.risk = end.swatch;
      head.append(swatch);
    }
    head.append(end.label);
    row.append(head);
    for (const value of end.values) row.append(node("td", ""), figureCell(value));
    return row;
  };
  /** @param {DriverRow} step */
  const stepRow = (step) => {
    const row = node("tr", "risk-drivers-step");
    const head = node("th", "", step.label);
    head.scope = "row";
    head.append(node("span", "risk-drivers-detail", step.detail));
    row.append(head);
    step.steps.forEach((cell, p) => {
      const bars = node("td", "risk-bar-cell");
      if (cell.text !== "") {
        const track = node("span", "risk-bar-track");
        if (cell.move !== 0) {
          const share = Math.min(1, Math.abs(cell.move) / model.reach[p]) * 50;
          const bar = node("i", `risk-bar ${cell.move < 0 ? "is-down" : "is-up"}`);
          bar.dataset.risk = model.id;
          bar.style.left = `${cell.move < 0 ? 50 - share : 50}%`;
          bar.style.width = `${share}%`;
          track.append(bar);
        }
        bars.append(track);
        row.title = `${row.title ? `${row.title}; ` : `${step.label} ${step.detail}: `}${model.columns[p]} ${cell.text}`;
      }
      row.append(bars, figureCell(cell));
    });
    return row;
  };

  const title = node("h3", "risk-drivers-title", model.title);
  if (model.empty !== null) {
    host.replaceChildren(title, node("p", "risk-drivers-empty", model.empty));
    return;
  }
  const table = node("table", "risk-drivers-table");
  const headRow = node("tr", "");
  headRow.append(node("th", ""));
  for (const column of model.columns) {
    const th = node("th", "", column);
    th.scope = "col";
    th.colSpan = 2;
    headRow.append(th);
  }
  const thead = node("thead", "");
  thead.append(headRow);
  const tbody = node("tbody", "");
  tbody.append(
    endRow(model.start),
    ...model.rows.map(stepRow),
    ...(model.together ? [stepRow(model.together)] : []),
    endRow(model.end)
  );
  table.append(thead, tbody);
  host.replaceChildren(title, table);
}
