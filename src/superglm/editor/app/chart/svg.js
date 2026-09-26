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
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
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
