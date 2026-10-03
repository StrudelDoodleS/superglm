// The dark palette's legibility and separability, computed from the CSS.
//
// Text: WCAG 2.2 contrast, (L1 + 0.05) / (L2 + 0.05) over relative luminance,
// at least 4.5:1 (success criterion 1.4.3) for every text token on every
// ground a rule sets it on, in both themes. Every token a rule sets as
// `color`, and every ground a rule sets beside it, must be listed, so a new
// one cannot skip the check.
//
// Categorical palettes, the slots a chart has to tell apart: every mark at
// least 3:1 on the chart's ground (1.4.11), and every two neighbouring slots,
// the wrap from the last to the first included because the chart cycles them,
//   - at least 15 apart in normal vision and 6 apart under protanopia and
//     deuteranopia, as Euclidean distance x100 in OKLab (Ottosson, "A
//     perceptual color space for image processing", 2020), the deficiencies
//     simulated in linear sRGB with Machado, Oliveira and Fernandes, "A
//     Physiologically-based Model for Simulation of Color Vision Deficiency",
//     IEEE TVCG 15(6), 2009, at severity 1.0;
//   - at least 0.06 apart in OKLab lightness, so neighbours stay apart when
//     hue is lost (spec section 4 I: colours that must be told apart also
//     differ in lightness).
// The 6 floor leans on a second channel, which each palette has: the group
// label under the axis, the Build's one highlighted basis, the legends.
//
// SVG text takes its colour from `fill`, not `color`, so the text check
// finds the classes the app's scripts give <text> elements and holds every
// stylesheet fill on one of them to 4.5:1 on the ground it is drawn on.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

/** @typedef {[number, number, number]} Triple */

/** @param {string} path */
function appFile(path) {
  return readFileSync(new URL(`../../src/superglm/editor/app/${path}`, import.meta.url), "utf8");
}

/**
 * The custom properties declared in the first rule with this selector.
 * @param {string} css @param {string} selector @returns {Map<string, string>}
 */
function declared(css, selector) {
  const start = css.indexOf(`${selector} {`);
  assert.notEqual(start, -1, `no ${selector} rule`);
  const body = css.slice(start, css.indexOf("}", start));
  return new Map([...body.matchAll(/(--[\w-]+):\s*([^;]+);/g)].map((match) => [match[1], match[2].trim()]));
}

const LIGHT = declared(appFile("styles/tokens.css"), ":root");
const DARK = new Map([...LIGHT, ...declared(appFile("styles/dark.css"), ':root[data-theme="dark"]')]);
const THEMES = /** @type {const} */ ([["light", LIGHT], ["dark", DARK]]);

/** @param {Map<string, string>} theme @param {string} token @returns {Triple} sRGB in [0, 1] */
function srgb(theme, token) {
  const value = theme.get(token) ?? "";
  const match = /^#([0-9a-f]{6})$/i.exec(value);
  assert.ok(match, `${token} is ${value || "missing"}, not a six-digit hex colour`);
  const n = Number.parseInt(match[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

// The sRGB transfer function inverted (IEC 61966-2-1). WCAG 2.0 wrote the
// knee as 0.03928; no 8-bit channel lies between the two.
/** @param {number} c */
const decode = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/** @param {Map<string, string>} theme @param {string} token @returns {Triple} */
function linearRgb(theme, token) {
  const [r, g, b] = srgb(theme, token);
  return [decode(r), decode(g), decode(b)];
}

/** @param {Map<string, string>} theme @param {string} token */
function luminance(theme, token) {
  const [r, g, b] = linearRgb(theme, token);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** @param {Map<string, string>} theme @param {string} a @param {string} b */
function contrast(theme, a, b) {
  const [high, low] = [luminance(theme, a), luminance(theme, b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

/** @param {Triple} rgb linear sRGB @returns {Triple} OKLab L, a, b */
function oklab([r, g, b]) {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

// Machado, Oliveira and Fernandes (2009), severity 1.0, on linear sRGB.
/** @type {Record<string, Triple[]>} */
const DEFICIENCIES = {
  protanopia: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deuteranopia: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
};

/** @param {Triple[]} matrix @param {Triple} rgb @returns {Triple} */
function simulate(matrix, rgb) {
  /** @param {Triple} row */
  const channel = (row) => Math.min(1, Math.max(0, row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2]));
  return [channel(matrix[0]), channel(matrix[1]), channel(matrix[2])];
}

/** @param {Triple} p @param {Triple} q */
const distance = (p, q) => 100 * Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

/** @param {Map<string, string>} theme @param {string} name @returns {string[]} */
function slots(theme, name) {
  const tokens = [];
  while (theme.has(`--${name}-${tokens.length}`)) tokens.push(`--${name}-${tokens.length}`);
  return tokens;
}

// Text tokens and the grounds the stylesheets set them on.
/** @type {[string, string[]][]} */
const TEXT_ON_GROUND = [
  // Body text; the hover ground under hovered rows; .export-format-card when
  // checked; the theme switch's DAY/NIGHT on its track.
  ["--text", ["--surface", "--surface-subtle", "--surface-hover", "--blue-soft", "--switch-track"]],
  // .history-chip and .se-cell.sig-unknown sit on the hover ground, the
  // checked export card's <small> on blue-soft.
  ["--muted", ["--surface", "--surface-subtle", "--surface-hover", "--blue-soft"]],
  // .tool-rail button.active, .selection-item:focus-visible, .context-bar button[aria-pressed].
  ["--blue", ["--surface", "--surface-subtle", "--blue-soft"]],
  // #status.is-error on the workspace, .summary-table td.advisory-code in the inspector.
  ["--danger", ["--surface", "--surface-subtle"]],
  // Metric and report deltas.
  ["--better", ["--surface", "--surface-subtle"]],
  ["--worse", ["--surface", "--surface-subtle"]],
  // .app-alert and its buttons.
  ["--danger-text", ["--danger-surface", "--surface"]],
  // button.primary and .ui-popover, the Refit action while changes wait,
  // and the popover's secondary line.
  ["--surface", ["--text", "--primary-hover", "--blue"]],
  ["--popover-muted", ["--text"]],
  ["--sig-strong-fg", ["--sig-strong-bg"]],
  ["--sig-medium-fg", ["--sig-medium-bg"]],
  ["--sig-standard-fg", ["--sig-standard-bg"]],
  // Waiting chips; the status line's and History's waiting text on the
  // workspace and the panel.
  ["--sig-weak-fg", ["--sig-weak-bg", "--surface", "--surface-subtle"]],
  ["--sig-none-fg", ["--sig-none-bg"]],
];

test("every text token keeps 4.5:1 on every ground it is set on, in both themes", () => {
  const failures = [];
  for (const [themeName, theme] of THEMES) {
    for (const [text, grounds] of TEXT_ON_GROUND) {
      for (const ground of grounds) {
        const ratio = contrast(theme, text, ground);
        if (!(ratio >= 4.5)) failures.push(`${themeName} ${text} on ${ground}: ${ratio.toFixed(2)}:1`);
      }
    }
  }
  assert.deepEqual(failures, []);
});

const STYLESHEETS = [
  "styles.css",
  ...readdirSync(new URL("../../src/superglm/editor/app/styles/", import.meta.url))
    .filter((name) => name.endsWith(".css"))
    .map((name) => `styles/${name}`),
];

test("the text check lists every text colour and text ground the stylesheets set", () => {
  const gated = new Map(TEXT_ON_GROUND.map(([text, grounds]) => [text, new Set(grounds)]));
  const missing = [];
  for (const file of STYLESHEETS) {
    const css = appFile(file).replace(/\/\*[\s\S]*?\*\//g, "");
    for (const [, selector, body] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const text = /(?<![-\w])color\s*:\s*var\((--[\w-]+)\)/.exec(body)?.[1];
      if (!text) continue;
      const ground = /(?<![-\w])background(?:-color)?\s*:\s*var\((--[\w-]+)\)\s*(?:;|$)/.exec(body)?.[1];
      const rule = `${file}: ${selector.trim().replace(/\s+/g, " ")}`;
      if (!gated.has(text)) missing.push(`${text} (${rule})`);
      else if (ground && !gated.get(text)?.has(ground)) missing.push(`${text} on ${ground} (${rule})`);
    }
  }
  assert.deepEqual(missing, []);
});

/**
 * A call's top-level arguments, from just after its opening parenthesis.
 * @param {string} source @param {number} start @returns {string[]}
 */
function callArguments(source, start) {
  const args = [];
  let current = "";
  let depth = 0;
  /** @type {string|null} */
  let quote = null;
  for (let i = start; i < source.length; i += 1) {
    const ch = source[i];
    if (quote) {
      current += ch;
      if (ch === "\\") current += source[++i];
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") quote = ch;
    else if ("([{".includes(ch)) depth += 1;
    else if (")]}".includes(ch)) {
      if (depth === 0) return [...args, current];
      depth -= 1;
    } else if (ch === "," && depth === 0) {
      args.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  return args;
}

const SCRIPTS = readdirSync(new URL("../../src/superglm/editor/app/", import.meta.url), { recursive: true })
  .map(String)
  .filter((name) => name.endsWith(".js"));

/**
 * The classes the app's scripts give SVG <text> elements: through chart/svg.js's
 * text(parent, x, y, value, cls, anchor), whose class may be a choice of
 * literals, and in markup written as strings.
 * @returns {Set<string>}
 */
function svgTextClasses() {
  const classes = new Set();
  /** @param {string} list */
  const add = (list) => list.split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
  for (const file of SCRIPTS) {
    const js = appFile(file);
    for (const match of js.matchAll(/<text\b[^>]*?\bclass="([^"$]+)"/g)) add(match[1]);
    for (const match of js.matchAll(/(?<![.\w])text\(/g)) {
      const cls = callArguments(js, match.index + match[0].length)[4] ?? "";
      for (const literal of cls.matchAll(/"([^"]*)"/g)) add(literal[1]);
    }
  }
  return classes;
}

// SVG text classes and the grounds they are drawn on.
/** @type {Record<string, string[]>} */
const SVG_TEXT_ON_GROUND = {
  // The editor's chart (#chart is --surface): axis titles, ticks, the
  // legend, a focused category tick, and the point tooltip's near-opaque box.
  label: ["--surface"],
  "tick-label": ["--surface"],
  "x-tick-label": ["--surface"],
  legend: ["--surface"],
  "point-tooltip-label": ["--surface"],
  "point-tooltip-value": ["--surface"],
  // The selection anchor's tags ("click · 56") and a shape's range tag sit on
  // --surface pills; a waiting range's tag on the waiting tint.
  "anchor-tag-label": ["--surface"],
  "shape-range-label": ["--surface"],
  "pending-range-label": ["--sig-weak-bg"],
  // The profile search's trace plot.
  "profile-trace-label": ["--surface"],
  "profile-trace-best-label": ["--surface"],
  // The Cross-validation tab's chart panel.
  "cv-tick": ["--surface-subtle"],
  "cv-level": ["--surface-subtle"],
  "cv-axis-title": ["--surface-subtle"],
};

test("every SVG text fill keeps 4.5:1 on the ground it is drawn on, in both themes", () => {
  const textClasses = svgTextClasses();
  for (const name of ["anchor-tag-label", "tick-label", "point-tooltip-label", "profile-trace-label", "cv-tick"]) {
    assert.ok(textClasses.has(name), `the script scan missed .${name}`);
  }
  const failures = [];
  let checked = 0;
  for (const file of STYLESHEETS) {
    const css = appFile(file).replace(/\/\*[\s\S]*?\*\//g, "");
    for (const [, selectors, body] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const fill = /(?<![-\w])fill\s*:\s*var\((--[\w-]+)\)/.exec(body)?.[1];
      if (!fill) continue;
      for (const selector of selectors.split(",").map((part) => part.trim())) {
        const subject = selector.split(/[\s>+~]+/).pop() ?? "";
        const names = [...subject.matchAll(/\.([\w-]+)/g)].map((match) => match[1]);
        for (const name of names.filter((candidate) => textClasses.has(candidate))) {
          const grounds = SVG_TEXT_ON_GROUND[name];
          if (!grounds) {
            failures.push(`${file}: ${selector} sets SVG text to ${fill} on a ground not listed`);
            continue;
          }
          for (const [themeName, theme] of THEMES) {
            for (const ground of grounds) {
              checked += 1;
              const ratio = contrast(theme, fill, ground);
              if (!(ratio >= 4.5)) {
                failures.push(`${themeName} ${selector}: ${fill} on ${ground} ${ratio.toFixed(2)}:1`);
              }
            }
          }
        }
      }
    }
  }
  assert.ok(checked > 0, "no SVG text fill was checked");
  assert.deepEqual(failures, []);
});

test("the dark categorical palettes keep neighbours apart in colour and in lightness", () => {
  // chart.js cycles six group and twelve basis colours, summary.js ten trace colours.
  const palettes = /** @type {const} */ ([["group", 6], ["basis", 12], ["trace", 10]]);
  const failures = [];
  for (const [name, count] of palettes) {
    const tokens = slots(DARK, name);
    assert.equal(tokens.length, count, `--${name}-* slots`);
    for (const token of tokens) {
      const ratio = contrast(DARK, token, "--surface");
      if (!(ratio >= 3)) failures.push(`${token} on --surface: ${ratio.toFixed(2)}:1`);
    }
    tokens.forEach((token, i) => {
      const next = tokens[(i + 1) % tokens.length];
      const p = linearRgb(DARK, token);
      const q = linearRgb(DARK, next);
      const normal = distance(oklab(p), oklab(q));
      const deficient = Math.min(
        ...Object.values(DEFICIENCIES).map((matrix) => distance(oklab(simulate(matrix, p)), oklab(simulate(matrix, q)))),
      );
      const lightness = Math.abs(oklab(p)[0] - oklab(q)[0]);
      if (!(normal >= 15)) failures.push(`${token}/${next}: ${normal.toFixed(1)} apart`);
      if (!(deficient >= 6)) failures.push(`${token}/${next}: ${deficient.toFixed(1)} apart under CVD`);
      if (!(lightness >= 0.06)) failures.push(`${token}/${next}: ${lightness.toFixed(3)} apart in lightness`);
    });
  }
  assert.deepEqual(failures, []);
});

test("the dark theme is gruvbox's warm dark, with the edit in its own blue", () => {
  // Spec D9: gruvbox grounds and text (morhetz/gruvbox, MIT/X11).
  const anchors = {
    "--surface": "#1d2021",
    "--surface-subtle": "#282828",
    "--border": "#3c3836",
    "--border-strong": "#7c6f64",
    "--text": "#ebdbb2",
    "--muted": "#a89984",
    "--grey": "#928374",
    "--blue": "#83a8e8",
    "--orange": "#fe8019",
    "--yellow": "#d79921",
    "--group-0": "#fabd2f",
    "--group-1": "#d3869b",
    "--trace-0": "#83a598",
    "--trace-1": "#b8bb26",
  };
  assert.deepEqual(Object.fromEntries(Object.keys(anchors).map((token) => [token, DARK.get(token)])), anchors);
});

test("the exposure strip is more opaque on the dark ground only", () => {
  // The light strip's opacity lives in styles.css and editor_style.py restates it.
  assert.match(appFile("styles.css"), /\.exposure \{[^}]*fill-opacity: 0\.6;/);
  const dark = appFile("styles/dark.css");
  const rule = dark.slice(dark.indexOf(':root[data-theme="dark"] .exposure,'));
  assert.match(rule, /^:root\[data-theme="dark"\] \.exposure,\s*:root\[data-theme="dark"\] \.exposure-density,\s*:root\[data-theme="dark"\] \.legend-swatch \{\s*fill-opacity: 0\.8;\s*\}/);
});
