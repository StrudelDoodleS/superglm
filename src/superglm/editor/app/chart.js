import { fmt } from "./format.js";
import { drawShapeOverlay } from "./chart/shape_overlay.js";
import {
  chartSize,
  evenlySpacedIndices,
  planCategoricalAxis,
  splitLabelGraphemes
} from "./chart/geometry.js";
import { el, line, text } from "./chart/svg.js";

const CATEGORICAL_MEASUREMENT_CACHE_LIMIT = 256;
// A curve drawn with more points than this hides them until they are
// selected, hovered, or within the lens around the pointer.
const DENSE_POINT_COUNT = 40;
const LENS_HALF_WIDTH = 26;
const CATEGORICAL_FONT_PROPERTIES = Object.freeze([
  "font-family",
  "font-size",
  "font-size-adjust",
  "font-style",
  "font-weight",
  "font-stretch",
  "font-variant",
  "font-feature-settings",
  "font-variation-settings",
  "font-kerning",
  "font-optical-sizing",
  "font-synthesis",
  "letter-spacing",
  "word-spacing",
  "line-height",
  "text-transform",
  "text-rendering",
  "direction",
  "writing-mode"
]);
const categoricalMeasurementCaches = new WeakMap();

export function groupedTerms(terms) {
  const order = ["spline", "ordered categorical", "categorical", "polynomial", "numeric"];
  const groups = new Map();
  for (const name of Object.keys(terms)) {
    const group = terms[name].term_type || terms[name].kind || "other";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(name);
  }
  const sorted = [];
  for (const group of order) {
    if (groups.has(group)) sorted.push([group, groups.get(group)]);
  }
  for (const [group, names] of groups.entries()) {
    if (!order.includes(group)) sorted.push([group, names]);
  }
  return sorted;
}

export function drawChart(term, selection, context) {
  // Full redraw renderer. The Python state payload is authoritative; this
  // module only turns the current payload into SVG plus scale metadata used by
  // interactions.js.
  const { svg } = context;
  const visualMode = context.visualMode();
  svg.innerHTML = "";
  // Draw at the chart's own CSS-pixel size so nothing is scaled: text keeps
  // its nominal size and the plot fills its panel. A hidden chart, or a DOM
  // without layout, draws at the fallback size.
  const { width, height } = chartSize(svg.clientWidth, svg.clientHeight);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const baseMargin = { left: 58, right: 46, top: 28, bottom: 52 };
  const view = resolveDisplayTerm(
    term,
    context.groupDisplayMode ? context.groupDisplayMode() : "expanded"
  );
  const x = view.x;
  const y = view.y;
  const original = view.original_y;
  const previous = view.previous_y || null;
  const exposure = view.exposure || null;
  if (!y.length) {
    svg.dataset.axisMeasurementCount = "0";
    return;
  }

  const xDomain = view.x_domain || [Math.min(...x), Math.max(...x)];
  const baseXMin = xDomain[0];
  const baseXMax = xDomain[1];
  const zoom = context.zoomState()[context.selectedTerm()];
  const xMin = zoom ? zoom.xMin : baseXMin;
  const xMax = zoom ? zoom.xMax : baseXMax;
  const categoricalLayout = view.levels
    ? categoricalAxisLayout(
        svg,
        view,
        xMin,
        xMax,
        width - baseMargin.left - baseMargin.right,
        height,
        baseMargin
      )
    : null;
  if (!categoricalLayout) svg.dataset.axisMeasurementCount = "0";
  const margin = {
    ...baseMargin,
    bottom: categoricalLayout ? categoricalLayout.bottom : baseMargin.bottom
  };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  definePlotClip(svg, margin, innerW, innerH);
  const ciValues = context.showCi() && view.ci_lower_y && view.ci_upper_y
    ? [...view.ci_lower_y, ...view.ci_upper_y]
    : [];
  const controlValues = visualMode === "handles" && term.controls && term.controls.y
    ? term.controls.y
    : [];
  const buildProgress = context.buildProgress ? context.buildProgress() : null;
  const buildActive = visualMode === "handles" &&
    context.buildProgress &&
    buildProgress !== null;
  const buildEnvelope = buildActive ? buildContributionEnvelope(term) : [];
  const buildValues = buildEnvelope.flat();
  const previousValues = previous || [];
  const yMinRaw = Math.min(
    ...y,
    ...original,
    ...previousValues,
    ...ciValues,
    ...controlValues,
    ...buildValues
  );
  const yMaxRaw = Math.max(
    ...y,
    ...original,
    ...previousValues,
    ...ciValues,
    ...controlValues,
    ...buildValues
  );
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.12, 0.05);
  const baseYMin = yMinRaw - yPad;
  const baseYMax = yMaxRaw + yPad;
  const yMin = zoom ? zoom.yMin : baseYMin;
  const yMax = zoom ? zoom.yMax : baseYMax;
  const sx = (v) => margin.left + ((v - xMin) / Math.max(xMax - xMin, 1e-12)) * innerW;
  const sy = (v) => margin.top + innerH - ((v - yMin) / Math.max(yMax - yMin, 1e-12)) * innerH;

  // Draw back-to-front: exposure context, axes/grid, reference intervals, then
  // curves and interactive handles/points.
  exposureLayer(svg, view, sx, margin, innerW, innerH, exposure);
  for (const tick of ticks(yMin, yMax, tickCount(innerH, 70))) {
    line(svg, margin.left, sy(tick), margin.left + innerW, sy(tick), "grid");
    text(svg, margin.left - 8, sy(tick) + 4, fmt(tick), "tick-label", "end");
  }
  if (categoricalLayout) {
    for (const tick of categoricalLayout.ticks) {
      const tickX = sx(Number(tick.value));
      const tickY = margin.top + innerH + 18;
      line(svg, tickX, margin.top + innerH, tickX, margin.top + innerH + 5, "tick");
      const tickLabel = text(
        svg,
        tickX,
        tickY,
        tick.displayLabel,
        tick.angle ? "tick-label x-tick-label angled" : "tick-label x-tick-label",
        tick.anchor
      );
      tickLabel.setAttribute("data-full-label", tick.fullLabel);
      tickLabel.setAttribute("data-popover-title", "Category");
      tickLabel.setAttribute("data-popover-body", tick.fullLabel);
      tickLabel.setAttribute("aria-label", tick.fullLabel);
      tickLabel.setAttribute("tabindex", "0");
      if (tick.angle) {
        tickLabel.setAttribute("transform", `rotate(${tick.angle} ${tickX} ${tickY})`);
      }
    }
  } else {
    for (const tick of continuousXTicks(xMin, xMax, innerW)) {
      const tickX = sx(tick.value);
      const tickY = margin.top + innerH + 20;
      line(svg, tickX, margin.top + innerH, tickX, margin.top + innerH + 5, "tick");
      text(svg, tickX, tickY, tick.label, "tick-label", "middle");
    }
  }
  // The baseline at relativity one: the level an edit is judged against.
  const baseline = Math.min(Math.max(1, yMin), yMax);
  line(svg, margin.left, sy(baseline), margin.left + innerW, sy(baseline), "zero");
  line(svg, margin.left, margin.top + innerH, margin.left + innerW, margin.top + innerH, "axis");

  text(
    svg,
    margin.left + innerW / 2,
    categoricalLayout ? categoricalLayout.titleY : height - 12,
    term.x_label,
    "label x-axis-title",
    "middle"
  );
  const yLabel = text(svg, 14, margin.top + innerH / 2, term.y_label, "label", "middle");
  yLabel.setAttribute("transform", `rotate(-90 14 ${margin.top + innerH / 2})`);
  // Shaped ranges sit above the grid and beneath the curves; a Build animation
  // shows the basis alone.
  if (!buildActive) drawShapeOverlay(svg, { term, view, sx, margin, innerW, innerH });

  if (context.showCi() && view.ci_lower_y && view.ci_upper_y) {
    if (view.levels) {
      errorBars(svg, x, view.ci_lower_y, view.ci_upper_y, sx, sy);
    } else {
      band(svg, x, view.ci_lower_y, view.ci_upper_y, sx, sy, "ci");
    }
  }
  if (visualMode === "handles" && context.showContrib && context.showContrib()) {
    basisContributions(svg, term, sx, sy, buildActive);
  }
  if (visualMode === "handles" && buildProgress !== null) {
    const progress = Math.min(Math.max(Number(buildProgress) || 0, 0), 1);
    const buildCurve = buildAccumulationCurve(term, progress);
    drawActiveBasis(svg, term, buildCurve.activeIndex, sx, sy);
    path(svg, buildCurve.x, buildCurve.y, sx, sy, "basis-build-halo");
    const build = path(svg, buildCurve.x, buildCurve.y, sx, sy, "basis-build");
    build.setAttribute("data-progress", progress.toFixed(4));
    build.setAttribute("data-active-basis", String(buildCurve.activeIndex));
    build.setAttribute("style", `stroke: ${mixBuildColor(progress)}`);
  }
  if (!buildActive) path(svg, x, original, sx, sy, "original");
  if (!buildActive && previous) path(svg, x, previous, sx, sy, "previous-edit");
  if (!buildActive) path(svg, x, y, sx, sy, "edited");
  const displaySelected = displaySelection(view, selection);
  const selectedBounds = selectionBounds(x, y, displaySelected, sx, sy, margin, innerW, innerH);
  const handlesMode = visualMode === "handles" && term.controls;
  const plot = { top: margin.top, height: innerH };
  if (!handlesMode && selectedBounds) drawSelectionBounds(svg, selectedBounds, plot);
  if (!handlesMode) {
    if (view.displayIsCollapsed) drawCollapsedLevelGroups(svg, view, sx, sy);
    else drawLevelGroups(svg, view, sx, sy);
  }
  const visiblePoints = visiblePointIndices(view, displaySelected);
  const basePoints = new Set(basePointIndices(view));
  const selectedPoints = [];
  const unselectedPoints = [];
  let pointLayer = null;
  if (!handlesMode) {
    pointLayer = el("g", {
      class: "point-layer",
      "data-dense": String(basePoints.size > DENSE_POINT_COUNT)
    });
    svg.appendChild(pointLayer);
    for (const i of visiblePoints) {
      if (displaySelected.has(i)) selectedPoints.push(i);
      else unselectedPoints.push(i);
    }
    for (const i of unselectedPoints) {
      drawPoint(svg, pointLayer, view, x, y, sx, sy, i, false, !basePoints.has(i));
    }
    for (const i of selectedPoints) {
      drawPoint(svg, pointLayer, view, x, y, sx, sy, i, true, !basePoints.has(i));
    }
  } else {
    drawControlHandles(svg, term, sx, sy, margin, innerH);
  }
  applyPlotClip(svg);
  const legendLayer = el("g", { class: "legend-layer" });
  svg.appendChild(legendLayer);
  legend(legendLayer, width - 10, 13, {
    originalProjected: view.displayIsCollapsed,
    hasPrevious: Boolean(previous),
    exposureLabel: exposure && exposure.y && exposure.y.length
      ? exposure.label || "exposure"
      : null
  });

  svg._scale = {
    sx, sy, x, y, xMin, xMax, yMin, yMax,
    baseXMin, baseXMax, baseYMin, baseYMax,
    margin, innerW, innerH,
    displayToSourceIndices: view.displayToSourceIndices,
    displayIsCollapsed: view.displayIsCollapsed
  };
  svg._selectionView = { term, view, handlesMode, pointLayer };
  positionSelectionMenu(svg, context.selectionMenu, handlesMode ? null : selectedBounds);
}

export function updateChartSelection(term, selection, context) {
  const { svg } = context;
  const scale = svg._scale;
  const selectionView = svg._selectionView;
  if (!scale || !selectionView) return;
  selectionView.term = term;

  const { view, handlesMode, pointLayer } = selectionView;
  const displaySelected = displaySelection(view, selection);
  const basePoints = new Set(basePointIndices(view));
  const showSupplementalPoints = displaySelected.size <= basePoints.size;
  const existingPoints = new Map();
  const selectedPoints = [];
  for (const point of svg.querySelectorAll("circle.point[data-index]")) {
    const index = Number(point.dataset.index);
    if (!Number.isInteger(index)) continue;
    const supplemental = point.dataset.selectionSupplemental === "true" ||
      !basePoints.has(index);
    if (supplemental) point.dataset.selectionSupplemental = "true";
    if (supplemental && (!showSupplementalPoints || !displaySelected.has(index))) {
      point.remove();
      continue;
    }
    const selected = displaySelected.has(index);
    point.classList.toggle("selected", selected);
    point.setAttribute("r", selected ? "4.6" : "3.4");
    if (selected) selectedPoints.push(point);
    existingPoints.set(index, point);
  }

  if (!handlesMode && pointLayer && showSupplementalPoints) {
    for (const index of displaySelected) {
      if (
        existingPoints.has(index) ||
        index < 0 ||
        index >= view.y.length
      ) continue;
      const point = drawPoint(
        svg,
        pointLayer,
        view,
        view.x,
        view.y,
        scale.sx,
        scale.sy,
        index,
        true,
        true
      );
      point.setAttribute("clip-path", "url(#plotInteractionClip)");
      selectedPoints.push(point);
    }
  }
  if (pointLayer) {
    selectedPoints.sort((left, right) => Number(left.dataset.index) - Number(right.dataset.index));
    for (const point of selectedPoints) pointLayer.appendChild(point);
  }

  const bounds = handlesMode
    ? null
    : selectionBounds(
        view.x,
        view.y,
        displaySelected,
        scale.sx,
        scale.sy,
        scale.margin,
        scale.innerW,
        scale.innerH
      );
  updateSelectionBounds(svg, bounds, { top: scale.margin.top, height: scale.innerH });
  positionSelectionMenu(svg, context.selectionMenu, bounds);
}

/**
 * A dense curve shows the points near the pointer: a lens that follows it
 * across the chart, so the handles are there when reached for and the line
 * stays clean otherwise. Bound once; the drawn circles change underneath.
 * @param {SVGSVGElement} svg
 */
export function bindPointLens(svg) {
  const reveal = (pointerX) => {
    const layer = svg.querySelector(".point-layer[data-dense='true']");
    if (!layer) return;
    for (const point of layer.querySelectorAll("circle.point[data-index]")) {
      const near = pointerX !== null &&
        Math.abs(Number(point.getAttribute("cx")) - pointerX) <= LENS_HALF_WIDTH;
      point.classList.toggle("near", near);
    }
  };
  svg.addEventListener("pointermove", (event) => reveal(svgPointerX(svg, event)));
  svg.addEventListener("pointerleave", () => reveal(null));
}

function svgPointerX(svg, event) {
  const matrix = svg.getScreenCTM();
  if (!matrix) return null;
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(matrix.inverse()).x;
}

function resolveDisplayTerm(term, mode) {
  const display = term.group_display;
  if (mode !== "collapsed" || !display || !display.available || !display.collapsed) {
    return {
      ...term,
      displayToSourceIndices: term.x.map((_, i) => [i]),
      displayIsCollapsed: false
    };
  }
  const collapsed = display.collapsed;
  return {
    ...term,
    x: collapsed.x,
    x_domain: collapsed.x_domain,
    y: collapsed.y,
    original_y: collapsed.original_y,
    previous_y: collapsed.previous_y || null,
    ci_lower_y: collapsed.ci_lower_y || null,
    ci_upper_y: collapsed.ci_upper_y || null,
    weights: collapsed.weights,
    exposure: collapsed.exposure,
    levels: collapsed.levels,
    handle_indices: collapsed.x.map((_, i) => i),
    displayToSourceIndices: collapsed.source_indices,
    displayLevels: collapsed.levels,
    displaySourceLevels: collapsed.source_levels,
    displayIsGroup: collapsed.is_group,
    displayIsCollapsed: true
  };
}

function displaySelection(view, selection) {
  if (!view.displayToSourceIndices) return selection;
  const selected = new Set();
  for (let i = 0; i < view.displayToSourceIndices.length; i++) {
    const source = view.displayToSourceIndices[i] || [];
    if (source.some((index) => selection.has(Number(index)))) selected.add(i);
  }
  return selected;
}

function definePlotClip(svg, margin, innerW, innerH) {
  // The clip path makes zoom/pan behave like a real plotting viewport instead
  // of letting paths spill outside the axes.
  const defs = el("defs", {});
  const clip = el("clipPath", { id: "plotClip" });
  clip.appendChild(el("rect", {
    x: margin.left,
    y: margin.top,
    width: innerW,
    height: innerH
  }));
  defs.appendChild(clip);
  // Chromium excludes the exact boundary of an SVG clip from pointer hit
  // testing. Give draggable marks enough room for their radius while keeping
  // zoomed-out marks away from the axes and labels.
  const interactionPad = 6;
  const interactionClip = el("clipPath", { id: "plotInteractionClip" });
  interactionClip.appendChild(el("rect", {
    x: margin.left - interactionPad,
    y: margin.top - interactionPad,
    width: innerW + interactionPad * 2,
    height: innerH + interactionPad * 2
  }));
  defs.appendChild(interactionClip);
  svg.appendChild(defs);
}

function applyPlotClip(svg) {
  const clipped = [
    ".original",
    ".previous-edit",
    ".edited",
    ".ci",
    ".ci-whisker",
    ".exposure",
    ".exposure-density",
    ".basis-contribution",
    ".basis-active",
    ".basis-build-halo",
    ".basis-build",
    ".level-group-link",
    ".level-group-marker",
    ".point",
    ".control-stem",
    ".control-handle"
  ].join(",");
  for (const node of svg.querySelectorAll(clipped)) {
    node.setAttribute("clip-path", "url(#plotClip)");
  }
  for (const node of svg.querySelectorAll(".point,.control-handle")) {
    node.setAttribute("clip-path", "url(#plotInteractionClip)");
  }
}

function visiblePointIndices(term, selection) {
  // Large continuous grids draw a representative point subset for performance,
  // but selected points are always forced visible.
  const base = basePointIndices(term);
  const out = new Set(base);
  if (selection.size <= base.length) {
    for (const i of selection) out.add(i);
  }
  return Array.from(out).sort((a, b) => a - b);
}

function basePointIndices(term) {
  return term.handle_indices || term.x.map((_, i) => i);
}

function drawPoint(svg, pointLayer, term, x, y, sx, sy, i, selected, supplemental = false) {
  const attrs = {
    cx: sx(x[i]), cy: sy(y[i]), r: selected ? 4.6 : 3.4,
    class: selected ? "point selected" : "point",
    "data-index": i
  };
  if (supplemental) attrs["data-selection-supplemental"] = "true";
  const circle = el("circle", attrs);
  circle.addEventListener("pointerenter", () => {
    showPointTooltip(svg, circle, pointTooltipLines(term, i));
  });
  circle.addEventListener("pointermove", () => {
    showPointTooltip(svg, circle, pointTooltipLines(term, i));
  });
  circle.addEventListener("pointerleave", () => hidePointTooltip(svg));
  pointLayer.appendChild(circle);
  return circle;
}

function pointTooltipLines(term, i) {
  const label = pointLabel(term, i);
  const exposure = Array.isArray(term.weights) ? term.weights[i] : null;
  return [
    label,
    `Relativity: ${fmt(term.y[i])}`,
    `Exposure: ${fmt(exposure)}`
  ];
}

function pointLabel(term, i) {
  if (Array.isArray(term.levels) && term.levels[i] !== undefined) return String(term.levels[i]);
  if (Array.isArray(term.displayLevels) && term.displayLevels[i] !== undefined) {
    return String(term.displayLevels[i]);
  }
  const label = term.x_label || "x";
  return `${label}: ${fmt(term.x[i])}`;
}

function showPointTooltip(svg, target, lines) {
  hidePointTooltip(svg);
  const cx = Number(target.getAttribute("cx") || 0);
  const cy = Number(target.getAttribute("cy") || 0);
  const width = tooltipWidth(lines);
  const height = 12 + lines.length * 16;
  const { width: svgWidth, height: svgHeight } = svg.viewBox.baseVal;
  const x = Math.max(8, Math.min(svgWidth - width - 8, cx + 12));
  const y = Math.max(8, Math.min(svgHeight - height - 8, cy - height - 12));
  const group = el("g", { class: "point-tooltip" });
  group.appendChild(el("rect", { x, y, width, height, rx: 4, ry: 4 }));
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const node = text(
      group,
      x + 8,
      y + 18 + lineIndex * 16,
      lines[lineIndex],
      lineIndex === 0 ? "point-tooltip-label" : "point-tooltip-value",
      "start"
    );
    if (lineIndex === 0) node.setAttribute("font-weight", "700");
  }
  svg.appendChild(group);
}

function hidePointTooltip(svg) {
  for (const node of svg.querySelectorAll(".point-tooltip")) node.remove();
}

function tooltipWidth(lines) {
  const longest = Math.max(...lines.map((line) => String(line).length));
  return Math.min(Math.max(longest * 7 + 18, 120), 240);
}

function drawLevelGroups(svg, term, sx, sy) {
  const groups = Array.isArray(term.level_groups) ? term.level_groups : [];
  for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
    const group = groups[groupIndex];
    const indices = Array.isArray(group.indices)
      ? group.indices.map(Number).filter((i) => i >= 0 && i < term.x.length)
      : [];
    if (indices.length < 2) continue;
    indices.sort((a, b) => term.x[a] - term.x[b]);
    const xs = indices.map((i) => term.x[i]);
    const ys = indices.map((i) => term.y[i]);
    const color = levelGroupColor(groupIndex, 0.9);
    const link = path(svg, xs, ys, sx, sy, "level-group-link");
    link.setAttribute("style", `stroke: ${color}`);
    for (const i of indices) {
      drawLevelGroupMarker(svg, term.x[i], term.y[i], sx, sy, groupIndex);
    }
    const mid = Math.floor(indices.length / 2);
    const label = text(
      svg,
      sx(term.x[indices[mid]]),
      Math.min(...ys.map(sy)) - 12,
      group.label || "group",
      "level-group-label",
      "middle"
    );
    label.setAttribute("style", `fill: ${levelGroupColor(groupIndex, 1)}`);
  }
}

function drawCollapsedLevelGroups(svg, term, sx, sy) {
  const isGroup = Array.isArray(term.displayIsGroup) ? term.displayIsGroup : [];
  let groupIndex = 0;
  for (let i = 0; i < term.x.length; i++) {
    if (!isGroup[i]) continue;
    const color = levelGroupColor(groupIndex, 1);
    drawLevelGroupMarker(svg, term.x[i], term.y[i], sx, sy, groupIndex);
    const label = text(
      svg,
      sx(term.x[i]),
      sy(term.y[i]) - 12,
      term.levels?.[i] || "group",
      "level-group-label",
      "middle"
    );
    label.setAttribute("style", `fill: ${color}`);
    groupIndex += 1;
  }
}

function drawLevelGroupMarker(svg, x, y, sx, sy, groupIndex) {
  const cx = sx(x);
  const cy = sy(y);
  const size = 13;
  const color = levelGroupColor(groupIndex, 0.95);
  const fill = levelGroupColor(groupIndex, 0.16);
  const attrs = {
    class: "level-group-marker",
    style: `stroke: ${color}; fill: ${fill}`
  };
  if (groupIndex % 3 === 0) {
    svg.appendChild(el("rect", {
      ...attrs,
      x: cx - size / 2,
      y: cy - size / 2,
      width: size,
      height: size,
      rx: 2,
      ry: 2
    }));
  } else if (groupIndex % 3 === 1) {
    svg.appendChild(el("polygon", {
      ...attrs,
      points: `${cx},${cy - size / 2} ${cx + size / 2},${cy} ${cx},${cy + size / 2} ${cx - size / 2},${cy}`
    }));
  } else {
    const h = size * 0.62;
    svg.appendChild(el("polygon", {
      ...attrs,
      points: `${cx},${cy - h} ${cx + size / 2},${cy + h / 2} ${cx - size / 2},${cy + h / 2}`
    }));
  }
}

function levelGroupColor(index, alpha = 1) {
  const colors = [
    [196, 116, 0],
    [126, 34, 206],
    [5, 150, 105],
    [220, 38, 38],
    [8, 145, 178],
    [37, 99, 235]
  ];
  const rgb = colors[Math.abs(Number(index) || 0) % colors.length];
  const opacity = Math.max(0, Math.min(1, Number(alpha)));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${opacity})`;
}

function drawControlHandles(svg, term, sx, sy, margin, innerH) {
  const controls = term.controls;
  if (!controls || !controls.x || !controls.y) return;
  const yBase = margin.top + innerH;
  for (let i = 0; i < controls.x.length; i++) {
    const cx = sx(controls.x[i]);
    const cy = sy(controls.y[i]);
    line(svg, cx, margin.top, cx, yBase, "control-stem");
    svg.appendChild(el("rect", {
      x: cx - 5,
      y: cy - 5,
      width: 10,
      height: 10,
      rx: 2,
      ry: 2,
      class: "control-handle",
      "data-control-index": i,
      "data-basis-index": controls.basis_index ? controls.basis_index[i] : i
    }));
  }
}

function basisContributions(svg, term, sx, sy, buildActive = false) {
  const { basis, logEffects } = contributionComponents(term);
  for (let i = 0; i < basis.length; i++) {
    const row = basis[i];
    if (!Array.isArray(row) || row.length !== term.x.length) continue;
    const beta = Array.isArray(logEffects) ? Number(logEffects[i] || 0) : 0;
    const y = row.map((v) => Math.exp((Number(v) || 0) * beta));
    const contribution = path(svg, term.x, y, sx, sy, "basis-contribution");
    contribution.setAttribute("data-basis-index", i);
  }
}

function buildAccumulationCurve(term, progress) {
  const { basis, logEffects } = contributionComponents(term);
  const x = term.x || [];
  if (!x.length) return { x: [], y: [], activeIndex: -1 };
  const eta = new Array(x.length).fill(0);
  const activeIndex = activeBasisIndex(basis, progress);
  const scaled = Math.max(0, Math.min(1, Number(progress) || 0)) * basis.length;
  const activeWeight = activeIndex < 0 ? 0 : scaled - Math.floor(scaled);
  for (let j = 0; j < basis.length; j++) {
    const row = basis[j];
    if (!Array.isArray(row) || row.length !== x.length) continue;
    const beta = Number(logEffects[j] || 0);
    const weight = activeIndex < 0 || j < activeIndex ? 1 : (j === activeIndex ? activeWeight : 0);
    if (weight <= 0) continue;
    for (let i = 0; i < row.length; i++) eta[i] += (Number(row[i]) || 0) * beta * weight;
  }
  return { x, y: eta.map((value) => Math.exp(value)), activeIndex };
}

function activeBasisIndex(basis, progress) {
  if (!basis.length) return -1;
  const p = Math.max(0, Math.min(1, Number(progress) || 0));
  if (p >= 1) return -1;
  return Math.min(basis.length - 1, Math.floor(p * basis.length));
}

function drawActiveBasis(svg, term, index, sx, sy) {
  if (index < 0) return;
  const { basis, logEffects } = contributionComponents(term);
  const row = basis[index];
  if (!Array.isArray(row) || row.length !== term.x.length) return;
  const beta = Number(logEffects[index] || 0);
  const y = row.map((v) => Math.exp((Number(v) || 0) * beta));
  const active = path(svg, term.x, y, sx, sy, "basis-active");
  active.setAttribute("data-basis-index", index);
  active.setAttribute("style", `stroke: ${basisColor(index, 0.72)}`);
}

function buildContributionEnvelope(term) {
  const { basis, logEffects } = contributionComponents(term);
  const finalEta = finalContributionEta(basis, logEffects, term.x.length);
  const values = [finalEta.map((value) => Math.exp(value))];
  for (let j = 0; j < basis.length; j++) {
    const row = basis[j];
    if (!Array.isArray(row) || row.length !== term.x.length) continue;
    const beta = Number(logEffects[j] || 0);
    values.push(row.map((v) => Math.exp((Number(v) || 0) * beta)));
  }
  return values;
}

function finalContributionEta(basis, logEffects, n) {
  const eta = new Array(n).fill(0);
  for (let j = 0; j < basis.length; j++) {
    const row = basis[j];
    if (!Array.isArray(row) || row.length !== n) continue;
    const beta = Number(logEffects[j] || 0);
    for (let i = 0; i < row.length; i++) eta[i] += (Number(row[i]) || 0) * beta;
  }
  return eta;
}

function mixBuildColor(progress) {
  const t = Math.max(0, Math.min(1, Number(progress) || 0));
  const start = [22, 163, 74];
  const end = [9, 105, 218];
  const rgb = start.map((value, i) => Math.round(value + (end[i] - value) * t));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function basisColor(index, alpha = 1) {
  const colors = [
    [22, 163, 74],
    [217, 119, 6],
    [124, 58, 237],
    [8, 145, 178],
    [220, 38, 38],
    [37, 99, 235],
    [194, 65, 12],
    [101, 163, 13],
    [190, 24, 93],
    [15, 118, 110],
    [147, 51, 234],
    [202, 138, 4]
  ];
  const rgb = colors[Math.abs(Number(index) || 0) % colors.length];
  const opacity = Math.max(0, Math.min(1, Number(alpha)));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${opacity})`;
}

function contributionComponents(term) {
  const controls = term.controls || {};
  const basis = Array.isArray(controls.build_basis) && controls.build_basis.length
    ? controls.build_basis
    : controls.basis;
  const logEffects = Array.isArray(controls.build_log_effect) && controls.build_log_effect.length
    ? controls.build_log_effect
    : controls.log_effect;
  return {
    basis: Array.isArray(basis) ? basis : [],
    logEffects: Array.isArray(logEffects) ? logEffects : []
  };
}

function selectionBounds(x, y, selection, sx, sy, margin, innerW, innerH) {
  // The floating action menu anchors to selected points, not the drag box, so
  // it stays useful after selection is committed.
  const selected = Array.from(selection)
    .filter((i) => i >= 0 && i < y.length)
    .map((i) => ({ x: sx(x[i]), y: sy(y[i]) }))
    .filter((point) => (
      point.x >= margin.left &&
      point.x <= margin.left + innerW &&
      point.y >= margin.top &&
      point.y <= margin.top + innerH
    ));
  if (!selected.length) return null;
  const xs = selected.map((point) => point.x);
  const ys = selected.map((point) => point.y);
  let x0 = Math.min(...xs) - 10;
  let x1 = Math.max(...xs) + 10;
  let y0 = Math.min(...ys) - 10;
  let y1 = Math.max(...ys) + 10;
  const minSize = 28;
  if (x1 - x0 < minSize) {
    const mid = (x0 + x1) / 2;
    x0 = mid - minSize / 2;
    x1 = mid + minSize / 2;
  }
  if (y1 - y0 < minSize) {
    const mid = (y0 + y1) / 2;
    y0 = mid - minSize / 2;
    y1 = mid + minSize / 2;
  }
  x0 = Math.max(margin.left, x0);
  x1 = Math.min(margin.left + innerW, x1);
  y0 = Math.max(margin.top, y0);
  y1 = Math.min(margin.top + innerH, y1);
  return { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
}

// The selection is drawn twice: a light band down the whole plot over the
// selected stretch, and a quiet frame around the points the palette sits by.
function selectionRects(bounds, plot) {
  return [
    ["selection-bounds-halo", { x: bounds.x, y: plot.top, width: bounds.width, height: plot.height, rx: 0, ry: 0 }],
    ["selection-bounds", { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height, rx: 4, ry: 4 }]
  ];
}

function drawSelectionBounds(svg, bounds, plot) {
  for (const [className, attrs] of selectionRects(bounds, plot)) {
    svg.appendChild(el("rect", { ...attrs, class: className }));
  }
}

function updateSelectionBounds(svg, bounds, plot) {
  for (const [className, attrs] of selectionRects(bounds || { x: 0, y: 0, width: 0, height: 0 }, plot)) {
    let node = svg.querySelector(`.${className}`);
    if (!bounds) {
      if (node) node.remove();
      continue;
    }
    if (!node) {
      node = el("rect", { class: className });
      const foreground = svg.querySelector([
        ".level-group-link",
        ".level-group-marker",
        ".level-group-label",
        ".point-layer"
      ].join(","));
      svg.insertBefore(node, foreground);
    }
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  }
}

function positionSelectionMenu(svg, selectionMenu, bounds) {
  // Bounds are in SVG coordinates; menu positioning needs viewport coordinates
  // because the menu is ordinary HTML layered over the SVG.
  if (!selectionMenu) return;
  if (!bounds) {
    selectionMenu.hidden = true;
    return;
  }
  const chartBox = svg.getBoundingClientRect();
  if (!chartBox.width || !chartBox.height) {
    selectionMenu.hidden = true;
    return;
  }
  selectionMenu.hidden = false;
  selectionMenu.style.left = "0px";
  selectionMenu.style.top = "0px";
  fitSelectionMenuWidth(selectionMenu, chartBox.width - 16);
  const menuBox = selectionMenu.getBoundingClientRect();
  const parentBox = (selectionMenu.offsetParent || svg.parentElement).getBoundingClientRect();
  const topLeft = svgClientPoint(svg, bounds.x, bounds.y);
  const bottomRight = svgClientPoint(svg, bounds.x + bounds.width, bounds.y + bounds.height);
  const pad = 8;
  const localLeft = topLeft.x - parentBox.left;
  const localTop = topLeft.y - parentBox.top;
  const localRight = bottomRight.x - parentBox.left;
  const localBottom = bottomRight.y - parentBox.top;
  const centeredLeft = (localLeft + localRight) / 2 - menuBox.width / 2;
  const centeredTop = (localTop + localBottom) / 2 - menuBox.height / 2;
  const candidates = [
    { left: centeredLeft, top: localTop - menuBox.height - 12 },
    { left: centeredLeft, top: localBottom + 12 },
    { left: localRight + 12, top: centeredTop },
    { left: localLeft - menuBox.width - 12, top: centeredTop }
  ];
  const scale = svg._scale;
  if (scale && scale.margin) {
    const plotBottom = svgClientPoint(
      svg,
      scale.margin.left,
      scale.margin.top + scale.innerH
    );
    candidates.push({
      left: centeredLeft,
      top: plotBottom.y - parentBox.top + pad
    });
  }
  // The palette keeps to the plot area when it fits there, so it never sits
  // on the axes or their labels; otherwise the chart's own box bounds it.
  const chartLimits = {
    minLeft: pad,
    maxLeft: Math.max(pad, chartBox.width - menuBox.width - pad),
    minTop: pad,
    maxTop: Math.max(pad, chartBox.height - menuBox.height - pad)
  };
  const limits = scale && scale.margin
    ? plotLimits(svg, scale, parentBox, menuBox, pad, chartLimits)
    : chartLimits;
  const positioned = candidates.map((candidate) => ({
    left: Math.max(limits.minLeft, Math.min(limits.maxLeft, candidate.left)),
    top: Math.max(limits.minTop, Math.min(limits.maxTop, candidate.top))
  }));
  let best = positioned[0];
  let bestIntersections = selectionMenuPointIntersections(svg, parentBox, menuBox, best);
  for (const candidate of positioned.slice(1)) {
    const intersections = selectionMenuPointIntersections(svg, parentBox, menuBox, candidate);
    if (intersections >= bestIntersections) continue;
    best = candidate;
    bestIntersections = intersections;
  }
  selectionMenu.style.left = `${best.left}px`;
  selectionMenu.style.top = `${best.top}px`;
}

// The palette wraps its rows itself: as wide as its widest row, never wider
// than the chart. Left to CSS, the row break would stretch it to the chart.
function fitSelectionMenuWidth(selectionMenu, maxWidth) {
  selectionMenu.style.width = "";
  const style = window.getComputedStyle(selectionMenu);
  const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  let widest = 0;
  let row = 0;
  for (const child of selectionMenu.children) {
    if (child.hidden) continue;
    if (child.classList.contains("selection-break")) {
      widest = Math.max(widest, row);
      row = 0;
      continue;
    }
    const childStyle = window.getComputedStyle(child);
    row += child.getBoundingClientRect().width +
      parseFloat(childStyle.marginLeft) + parseFloat(childStyle.marginRight);
  }
  widest = Math.max(widest, row);
  if (!widest) return;
  selectionMenu.style.width = `${Math.min(Math.ceil(widest + padding) + 1, Math.max(maxWidth, 1))}px`;
}

function plotLimits(svg, scale, parentBox, menuBox, pad, fallback) {
  const topLeft = svgClientPoint(svg, scale.margin.left, scale.margin.top);
  const bottomRight = svgClientPoint(
    svg,
    scale.margin.left + scale.innerW,
    scale.margin.top + scale.innerH
  );
  const limits = {
    minLeft: topLeft.x - parentBox.left + pad,
    maxLeft: bottomRight.x - parentBox.left - menuBox.width - pad,
    minTop: topLeft.y - parentBox.top + pad,
    maxTop: bottomRight.y - parentBox.top - menuBox.height - pad
  };
  const fits = limits.maxLeft >= limits.minLeft && limits.maxTop >= limits.minTop;
  return fits ? limits : fallback;
}

function selectionMenuPointIntersections(svg, parentBox, menuBox, candidate) {
  const clearance = 2;
  const menuLeft = parentBox.left + candidate.left - clearance;
  const menuTop = parentBox.top + candidate.top - clearance;
  const menuRight = menuLeft + menuBox.width + clearance * 2;
  const menuBottom = menuTop + menuBox.height + clearance * 2;
  let intersections = 0;
  for (const point of svg.querySelectorAll(visiblePointSelector(svg))) {
    const pointBox = point.getBoundingClientRect();
    if (
      pointBox.right >= menuLeft &&
      pointBox.left <= menuRight &&
      pointBox.bottom >= menuTop &&
      pointBox.top <= menuBottom
    ) {
      intersections += 1;
    }
  }
  return intersections;
}

// The points the palette must not cover: on a dense curve only the selected
// ones show, so only they count.
function visiblePointSelector(svg) {
  const layer = svg.querySelector(".point-layer");
  return layer && layer.getAttribute("data-dense") === "true"
    ? "circle.point.selected[data-index]"
    : "circle.point[data-index]";
}

function svgClientPoint(svg, x, y) {
  const matrix = svg.getScreenCTM();
  if (matrix) {
    const point = svg.createSVGPoint();
    point.x = x;
    point.y = y;
    return point.matrixTransform(matrix);
  }
  const box = svg.getBoundingClientRect();
  const viewBox = svg.viewBox.baseVal;
  return {
    x: box.left + (x - viewBox.x) * box.width / Math.max(viewBox.width, 1e-12),
    y: box.top + (y - viewBox.y) * box.height / Math.max(viewBox.height, 1e-12)
  };
}

function categoricalAxisLayout(svg, view, xMin, xMax, availableWidth, svgHeight, baseMargin) {
  const labels = view.levels.map(String);
  if (view.x.length !== labels.length) {
    throw new RangeError("categorical axis values and labels must have the same length");
  }
  const rows = labels.map((label, index) => ({ value: view.x[index], label }));
  const visibleRows = rows.filter((row) => row.value >= xMin && row.value <= xMax);
  const candidateIndices = evenlySpacedIndices(visibleRows.length, 30);
  const candidates = candidateIndices.map((index) => visibleRows[index]);
  const candidateLabels = candidates.map((row) => row.label);
  const measurements = measureCategoricalLabels(svg, candidateLabels);
  svg.dataset.axisMeasurementCount = String(candidateLabels.length);
  return planCategoricalAxis({
    values: candidates.map((row) => row.value),
    labels: candidateLabels,
    measurements,
    availableWidth,
    svgHeight,
    baseLeft: baseMargin.left,
    baseBottom: baseMargin.bottom
  });
}

function measureCategoricalLabels(svg, labels) {
  const layer = el("g", { class: "axis-measure-layer", "aria-hidden": "true" });
  layer.setAttribute("visibility", "hidden");
  svg.appendChild(layer);
  try {
    const probe = text(layer, 0, 0, "", "tick-label", "start");
    const cache = categoricalMeasurementCache(svg, probe);
    if (cache.ellipsisWidth === null) {
      cache.ellipsisWidth = measureText(probe, "…");
    }
    return labels.map((label) => {
      const cached = cache.labels.get(label);
      if (cached) {
        cache.labels.delete(label);
        cache.labels.set(label, cached);
        return cached;
      }
      const graphemes = splitLabelGraphemes(label);
      const prefixWidths = [];
      let prefix = "";
      for (const grapheme of graphemes) {
        prefix += grapheme;
        prefixWidths.push(measureText(probe, prefix));
      }
      probe.textContent = label;
      const box = probe.getBBox();
      const measurement = Object.freeze({
        fullWidth: prefixWidths.at(-1) || 0,
        prefixWidths: Object.freeze(prefixWidths),
        ellipsisWidth: cache.ellipsisWidth,
        height: Math.max(1, box.height)
      });
      cache.labels.set(label, measurement);
      evictOldCategoricalMeasurements(cache.labels);
      return measurement;
    });
  } finally {
    layer.remove();
  }
}

function categoricalMeasurementCache(svg, probe) {
  const fontSignature = categoricalFontSignature(probe);
  const current = categoricalMeasurementCaches.get(svg);
  if (current && current.fontSignature === fontSignature) return current;
  const cache = {
    fontSignature,
    ellipsisWidth: null,
    labels: new Map()
  };
  categoricalMeasurementCaches.set(svg, cache);
  return cache;
}

function categoricalFontSignature(probe) {
  const style = window.getComputedStyle(probe);
  const properties = CATEGORICAL_FONT_PROPERTIES.map(
    (property) => [property, style.getPropertyValue(property)]
  );
  // A web font that arrives after the first draw changes every width, so
  // whether the family is loaded yet is part of the signature.
  properties.push(["loaded", String(fontLoaded(style))]);
  return JSON.stringify(properties);
}

function fontLoaded(style) {
  const fonts = document.fonts;
  if (!fonts || typeof fonts.check !== "function") return true;
  try {
    return fonts.check(`${style.getPropertyValue("font-size")} ${style.getPropertyValue("font-family")}`);
  } catch {
    return true;
  }
}

function evictOldCategoricalMeasurements(cache) {
  while (cache.size > CATEGORICAL_MEASUREMENT_CACHE_LIMIT) {
    cache.delete(cache.keys().next().value);
  }
}

function measureText(probe, value) {
  probe.textContent = value;
  return probe.getComputedTextLength();
}

function continuousXTicks(xMin, xMax, innerW) {
  return ticks(xMin, xMax, tickCount(innerW, 130)).map((value) => ({ value, label: fmt(value) }));
}

// One tick per `spacing` px and at least three: the 940x520 fallback keeps its
// six ticks each way, and a wider chart gets more ticks, not the same six
// spread out.
function tickCount(extent, spacing) {
  return Math.max(3, Math.round(extent / spacing));
}

function ticks(min, max, n) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min || 0];
  const step = (max - min) / Math.max(n - 1, 1);
  return Array.from({ length: n }, (_, i) => min + i * step);
}

function exposureLayer(svg, term, sx, margin, innerW, innerH, exposure) {
  // Exposure uses a secondary visual scale inside the plot area. It is
  // contextual, not part of the relativity y-axis scale.
  if (!exposure || !exposure.y || !exposure.y.length) return;
  const maxWeight = Math.max(...exposure.y);
  if (!Number.isFinite(maxWeight) || maxWeight <= 0) return;
  const x = exposure.x || term.x;
  const yBase = margin.top + innerH;
  // A strip along the axis, up to a third of the plot: context for the curve.
  const maxH = innerH / 3;
  const exposureY = (v) => yBase - maxH * v / maxWeight;
  if (exposure.kind === "density") {
    exposureDensity(svg, x, exposure.y, sx, exposureY, yBase);
  } else {
    const nominalW = x.length > 1
      ? Math.abs(sx(x[1]) - sx(x[0])) * 0.7
      : innerW * 0.4;
    for (let i = 0; i < exposure.y.length; i++) {
      const h = Math.max(1, maxH * exposure.y[i] / maxWeight);
      svg.appendChild(el("rect", {
        x: sx(x[i]) - nominalW / 2,
        y: yBase - h,
        width: nominalW,
        height: h,
        rx: 2,
        ry: 2,
        class: "exposure"
      }));
    }
  }
  exposureAxis(svg, margin.left + innerW, yBase, maxH, maxWeight);
}

function exposureDensity(svg, x, y, sx, exposureY, yBase) {
  const top = x.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(v).toFixed(2)} ${exposureY(y[i]).toFixed(2)}`).join(" ");
  const right = `L ${sx(x[x.length - 1]).toFixed(2)} ${yBase.toFixed(2)}`;
  const left = `L ${sx(x[0]).toFixed(2)} ${yBase.toFixed(2)} Z`;
  svg.appendChild(el("path", { d: `${top} ${right} ${left}`, class: "exposure-density" }));
}

// The strip's scale: its top and bottom, ticked on the right edge. The legend
// names it.
function exposureAxis(svg, x, yBase, maxH, maxWeight) {
  line(svg, x, yBase - maxH, x, yBase, "exposure-axis");
  for (const value of [0, maxWeight]) {
    const y = yBase - maxH * value / maxWeight;
    line(svg, x, y, x + 4, y, "exposure-axis");
    text(svg, x + 7, y + 3.5, fmt(value), "tick-label small", "start");
  }
}

function path(svg, x, y, sx, sy, cls) {
  const d = x.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(v).toFixed(2)} ${sy(y[i]).toFixed(2)}`).join(" ");
  const node = el("path", { d, class: cls });
  svg.appendChild(node);
  return node;
}

function band(svg, x, lower, upper, sx, sy, cls) {
  const top = x.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(v).toFixed(2)} ${sy(upper[i]).toFixed(2)}`).join(" ");
  const bottom = x.slice().reverse().map((v, revI) => {
    const i = x.length - 1 - revI;
    return `L ${sx(v).toFixed(2)} ${sy(lower[i]).toFixed(2)}`;
  }).join(" ");
  svg.appendChild(el("path", { d: `${top} ${bottom} Z`, class: cls }));
}

function errorBars(svg, x, lower, upper, sx, sy) {
  const cap = 6;
  for (let i = 0; i < x.length; i++) {
    const px = sx(x[i]);
    const lo = sy(lower[i]);
    const hi = sy(upper[i]);
    line(svg, px, hi, px, lo, "ci-whisker");
    line(svg, px - cap, hi, px + cap, hi, "ci-whisker");
    line(svg, px - cap, lo, px + cap, lo, "ci-whisker");
  }
}

// One quiet row above the plot, ending at `right`: the series, then the
// exposure strip's swatch.
function legend(svg, right, y, { originalProjected, hasPrevious, exposureLabel }) {
  const items = [["original", originalProjected ? "original projection" : "original"]];
  if (hasPrevious) items.push(["previous-edit", "previous edit"]);
  items.push(["edited", "current edit"]);
  if (exposureLabel) items.push(["legend-swatch", exposureLabel]);
  const keyWidth = 22;
  const gap = 18;
  const widths = items.map(([, label]) => keyWidth + 6 + label.length * 5.6);
  let x = right - widths.reduce((total, width) => total + width + gap, -gap);
  items.forEach(([cls, label], index) => {
    if (cls === "legend-swatch") {
      svg.appendChild(el("rect", { x, y: y - 5, width: keyWidth, height: 10, rx: 2, ry: 2, class: cls }));
    } else {
      line(svg, x, y, x + keyWidth, y, cls);
    }
    text(svg, x + keyWidth + 6, y + 4, label, "legend", "start");
    x += widths[index] + gap;
  });
}
