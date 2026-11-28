// @ts-nocheck

// The check view's frame on a minimal element tree, shared by the tests of the view, its hover and
// its drawing.

export const CHAR_PX = 7;
export const LINE_PX = 16;
export const ASCENT_PX = 12;
/** A text node's lines: its tspans, or its own text. */
export const textLines = (node) => (node.children.length ? node.children.map((child) => child.textContent) : [node.text]);

// The large view and its hover, on a minimal element tree that records every attribute, text and
// markup write, so a test can say which nodes a gesture touched. A node is on the page and drawn,
// and holds the pointers it captured.
export class FakeNode {
  constructor(tag, log, attrs = {}) {
    this.tag = tag; this.log = log; this.attrs = new Map(Object.entries(attrs));
    this.children = []; this.parent = null; this.text = ""; this.html = ""; this.listeners = new Map();
    this.dataset = {}; this.rect = { left: 0, top: 0, width: 0, height: 0 };
    this.scrollTop = 0; this.scrollLeft = 0; this.clientWidth = 0; this.clientHeight = 0;
    this.ownerDocument = { createElementNS: (_ns, name) => new FakeNode(name, log) };
    this.isConnected = true; this.captured = new Set();
  }
  get classList() {
    const names = () => new Set((this.attrs.get("class") ?? "").split(/\s+/).filter(Boolean));
    const write = (set) => this.setAttribute("class", [...set].join(" "));
    return {
      add: (name) => { const set = names(); if (!set.has(name)) write(set.add(name)); },
      remove: (name) => { const set = names(); if (set.delete(name)) write(set); },
      contains: (name) => names().has(name),
    };
  }
  contains(other) {
    for (let at = other; at; at = at.parent) if (at === this) return true;
    return false;
  }
  getClientRects() { return [this.rect]; }
  // The frame's content is its plot, which may run past it.
  get scrollHeight() { return Math.max(this.clientHeight, ...this.children.map((child) => child.rect.height)); }
  setPointerCapture(id) { this.captured.add(id); }
  hasPointerCapture(id) { return this.captured.has(id); }
  releasePointerCapture(id) { this.captured.delete(id); }
  write(what) { this.log.push({ node: this, what }); }
  setAttribute(name, value) { this.write(name); this.attrs.set(name, String(value)); }
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  hasAttribute(name) { return this.attrs.has(name); }
  removeAttribute(name) { this.write(name); this.attrs.delete(name); }
  toggleAttribute(name, force) {
    const on = force ?? !this.attrs.has(name);
    if (on) this.setAttribute(name, ""); else this.removeAttribute(name);
    return on;
  }
  get hidden() { return this.attrs.has("hidden"); }
  set hidden(value) { this.toggleAttribute("hidden", Boolean(value)); }
  get textContent() { return this.text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this.write("text"); this.text = String(value); this.children = []; }
  get innerHTML() { return this.html; }
  set innerHTML(markup) {
    this.write("html"); this.html = markup; this.text = ""; this.children = [];
    const size = /<svg class="check-plot" viewBox="[^"]*" width="([\d.]+)" height="([\d.]+)"/.exec(markup);
    // As a browser does, the scroll is kept where the new content allows it.
    this.scrollTop = Math.min(this.scrollTop, Math.max(0, (size ? Number(size[2]) : 0) - this.clientHeight));
    if (!size) return;
    const svg = new FakeNode("svg", this.log, { class: "check-plot" });
    svg.rect = { left: this.rect.left - this.scrollLeft, top: this.rect.top - this.scrollTop, width: Number(size[1]), height: Number(size[2]) };
    const readout = new FakeNode("g", this.log, { class: "check-readout", hidden: "" });
    svg.adopt(readout); this.adopt(svg);
  }
  get localName() { return this.tag; }
  adopt(child) { child.parent = this; this.children.push(child); }
  append(...nodes) { this.write("children"); for (const node of nodes) this.adopt(node); }
  insertBefore(node, ref) {
    this.write("children");
    node.parent = this;
    const at = ref ? this.children.indexOf(ref) : -1;
    this.children.splice(at < 0 ? this.children.length : at, 0, node);
    return node;
  }
  remove() {
    this.parent?.write("children");
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }
  matches(selector) {
    if (selector.startsWith(".")) return (this.attrs.get("class") ?? "").split(/\s+/).includes(selector.slice(1));
    return this.tag === selector;
  }
  closest(selector) {
    for (let node = this; node; node = node.parent) if (node.matches(selector)) return node;
    return null;
  }
  querySelector(selector) {
    for (const child of this.children) {
      if (child.matches(selector)) return child;
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
  getBoundingClientRect() { return this.rect; }
  // Text is 7 px a character, its lines 16 px apart with the first baseline 12 px down.
  getComputedTextLength() { return this.textContent.length * CHAR_PX; }
  getBBox() {
    const lines = textLines(this);
    return { x: 0, y: -ASCENT_PX, width: Math.max(...lines.map((line) => line.length)) * CHAR_PX, height: lines.length * LINE_PX };
  }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter((l) => l !== listener)); }
  dispatch(type, event = {}) {
    const full = { preventDefault() { this.defaultPrevented = true; }, ...event };
    for (const listener of this.listeners.get(type) ?? []) listener(full);
    return full;
  }
}

export function fakeFrame({ width = 820, height = 520 } = {}) {
  const log = [];
  const node = (tag) => new FakeNode(tag, log);
  const nodes = {
    title: node("h2"), help: node("span"), split: node("span"), legend: node("div"), frame: node("div"), live: node("p"),
  };
  nodes.frame.clientWidth = width; nodes.frame.clientHeight = height;
  nodes.frame.rect = { left: 0, top: 0, width, height };
  // The animation frames asked for and not yet run, by id; a cancelled one does not run.
  const frames = new Map();
  let lastFrame = 0;
  globalThis.requestAnimationFrame = (callback) => { frames.set(++lastFrame, callback); return lastFrame; };
  globalThis.cancelAnimationFrame = (id) => { frames.delete(id); };
  const readout = () => nodes.frame.querySelector(".check-readout");
  const pointer = (type, x, y, init = {}) => nodes.frame.dispatch(type, { pointerId: 1, button: 0, clientX: x, clientY: y, ...init });
  return {
    nodes,
    recordWrites() { const from = log.length; return () => log.slice(from); },
    pointerMove(x, y, init) { pointer("pointermove", x, y, init); },
    pointerDown: (x, y, init) => pointer("pointerdown", x, y, init),
    pointerUp: (x, y, init) => pointer("pointerup", x, y, init),
    pointerLeave() { nodes.frame.dispatch("pointerleave"); },
    wheel: (x, y, init = {}) => nodes.frame.dispatch("wheel", { deltaY: -100, ctrlKey: true, clientX: x, clientY: y, ...init }),
    // The frame scrolled to `top`, its plot's svg moving up with it.
    scrollTo(top) {
      nodes.frame.scrollTop = top;
      const svg = nodes.frame.querySelector("svg");
      if (svg) svg.rect = { ...svg.rect, top: nodes.frame.rect.top - top };
      nodes.frame.dispatch("scroll");
    },
    key(key) { return nodes.frame.dispatch("keydown", { key }); },
    flushFrame() {
      const due = [...frames.values()];
      frames.clear();
      for (const callback of due) callback(0);
    },
    pendingFrames: () => frames.size,
    readout,
    readoutShown: () => !readout().hidden,
    readoutText: () => readout().querySelector("text")?.textContent ?? "",
  };
}

/** A document that hands each listener the events dispatched on it, as the wheel focus reads it. */
export function fakeDocument() {
  const listeners = new Map();
  return {
    addEventListener(type, listener) { listeners.set(type, [...(listeners.get(type) ?? []), listener]); },
    removeEventListener(type, listener) { listeners.set(type, (listeners.get(type) ?? []).filter((l) => l !== listener)); },
    dispatch(type, init) {
      const event = { type, preventDefault() {}, ...init };
      for (const listener of listeners.get(type) ?? []) listener(event);
      return event;
    },
  };
}
