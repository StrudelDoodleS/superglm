// @ts-check
// Imperative SVG helpers shared by the chart renderer and its overlays.

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * @param {string} tag
 * @param {Record<string, string|number>} attrs
 * @returns {SVGElement}
 */
export function el(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const key in attrs) node.setAttribute(key, String(attrs[key]));
  return node;
}

/**
 * @param {Element} parent
 * @param {number} x1 @param {number} y1 @param {number} x2 @param {number} y2
 * @param {string} cls
 * @returns {SVGElement}
 */
export function line(parent, x1, y1, x2, y2, cls) {
  const node = el("line", { x1, y1, x2, y2, class: cls });
  parent.appendChild(node);
  return node;
}

/**
 * @param {Element} parent
 * @param {number} x @param {number} y
 * @param {string} value @param {string} cls @param {string} anchor
 * @returns {SVGElement}
 */
export function text(parent, x, y, value, cls, anchor) {
  const node = el("text", { x, y, class: cls, "text-anchor": anchor });
  node.textContent = value;
  parent.appendChild(node);
  return node;
}

/**
 * Set the attributes that differ from `attrs`.
 * @param {Element} node @param {Record<string, string|number>} attrs
 */
export function setAttributes(node, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    if (node.getAttribute(key) !== String(value)) node.setAttribute(key, String(value));
  }
}

/**
 * Make `live`'s attributes those of `next`, writing only the ones that differ.
 * @param {Element} live @param {Element} next
 */
function patchAttributes(live, next) {
  const names = next.getAttributeNames();
  for (const name of names) {
    const value = /** @type {string} */ (next.getAttribute(name));
    if (live.getAttribute(name) !== value) live.setAttribute(name, value);
  }
  if (live.attributes.length === names.length) return;
  for (const name of live.getAttributeNames()) {
    if (!next.hasAttribute(name)) live.removeAttribute(name);
  }
}

/**
 * Make `live` match `next`, a node of the same type and name.
 * @param {Node} live @param {Node} next
 */
export function patchNode(live, next) {
  if (live.nodeType !== 1) {
    if (live.nodeValue !== next.nodeValue) live.nodeValue = next.nodeValue;
    return;
  }
  patchAttributes(/** @type {Element} */ (live), /** @type {Element} */ (next));
  patchChildren(live, next);
}

/**
 * Make `live`'s children those of `next`, position by position, so a redraw costs the browser only
 * what changed: it keeps every node's styles, layout and paint. Nodes `live` lacks move out of `next`.
 * @param {Node} live @param {Node} next
 */
export function patchChildren(live, next) {
  let have = live.firstChild;
  let want = next.firstChild;
  while (want) {
    const following = want.nextSibling;
    if (!have) {
      live.appendChild(want);
    } else {
      const after = have.nextSibling;
      if (have.nodeType === want.nodeType && have.nodeName === want.nodeName) patchNode(have, want);
      else live.replaceChild(want, have);
      have = after;
    }
    want = following;
  }
  while (have) {
    const after = have.nextSibling;
    live.removeChild(have);
    have = after;
  }
}

const LABEL_WIDTH_LIMIT = 512;
/** @type {Map<string, number>} */
const labelWidths = new Map();
/** @type {SVGElement|null} */
let labelProbe = null;

/** A web font that arrives changes every width. */
if (typeof document !== "undefined") {
  document.fonts?.addEventListener?.("loadingdone", () => labelWidths.clear());
}

/** @returns {SVGElement|null} */
function measuringProbe() {
  if (labelProbe?.isConnected) return labelProbe;
  if (typeof document === "undefined" || !document.body) return null;
  // A size of its own and strict containment keep what it measures from laying out the page.
  const host = el("svg", {
    "aria-hidden": "true",
    style: "position: absolute; width: 0; height: 0; visibility: hidden; contain: strict"
  });
  labelProbe = el("text", {});
  host.appendChild(labelProbe);
  document.body.appendChild(host);
  return labelProbe;
}

/**
 * A label's width in its class's type, whether or not it is in the document. Zero where nothing
 * has layout.
 * @param {string} className @param {string} value
 * @returns {number}
 */
export function measureLabel(className, value) {
  const key = `${className}\u0000${value}`;
  const known = labelWidths.get(key);
  if (known !== undefined) return known;
  const probe = measuringProbe();
  const measure = /** @type {{getComputedTextLength?:()=>number}|null} */ (probe)?.getComputedTextLength;
  if (!probe || typeof measure !== "function") return 0;
  probe.setAttribute("class", className);
  probe.textContent = value;
  const width = measure.call(probe);
  if (labelWidths.size >= LABEL_WIDTH_LIMIT) labelWidths.clear();
  labelWidths.set(key, width);
  return width;
}
