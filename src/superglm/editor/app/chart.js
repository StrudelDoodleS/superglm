import { fmt } from "./format.js";

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
  const { svg, modeSelect, zoomState } = context;
  svg.innerHTML = "";
  const width = 940, height = 520;
  const margin = { left: 76, right: 76, top: 48, bottom: 72 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  definePlotClip(svg, margin, innerW, innerH);
  const x = term.x;
  const y = term.y;
  const original = term.original_y;
  const exposure = term.exposure || null;
  if (!y.length) return;

  const xDomain = term.x_domain || [Math.min(...x), Math.max(...x)];
  const baseXMin = xDomain[0];
  const baseXMax = xDomain[1];
  const ciValues = context.showCi() && term.ci_lower_y && term.ci_upper_y
    ? [...term.ci_lower_y, ...term.ci_upper_y]
    : [];
  const controlValues = modeSelect.value === "handles" && term.controls && term.controls.y
    ? term.controls.y
    : [];
  const buildProgress = context.buildProgress ? context.buildProgress() : null;
  const buildActive = modeSelect.value === "handles" &&
    context.buildProgress &&
    buildProgress !== null;
  const buildEnvelope = buildActive ? buildContributionEnvelope(term) : [];
  const buildValues = buildEnvelope.flat();
  const yMinRaw = Math.min(...y, ...original, ...ciValues, ...controlValues, ...buildValues);
  const yMaxRaw = Math.max(...y, ...original, ...ciValues, ...controlValues, ...buildValues);
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.12, 0.05);
  const baseYMin = yMinRaw - yPad;
  const baseYMax = yMaxRaw + yPad;
  const zoom = zoomState[context.selectedTerm()];
  const xMin = zoom ? zoom.xMin : baseXMin;
  const xMax = zoom ? zoom.xMax : baseXMax;
  const yMin = zoom ? zoom.yMin : baseYMin;
  const yMax = zoom ? zoom.yMax : baseYMax;
  const sx = (v) => margin.left + ((v - xMin) / Math.max(xMax - xMin, 1e-12)) * innerW;
  const sy = (v) => margin.top + innerH - ((v - yMin) / Math.max(yMax - yMin, 1e-12)) * innerH;

  // Draw back-to-front: exposure context, axes/grid, reference intervals, then
  // curves and interactive handles/points.
  exposureLayer(svg, term, sx, margin, innerW, innerH, exposure);
  for (const tick of ticks(yMin, yMax, 6)) {
    line(svg, margin.left, sy(tick), margin.left + innerW, sy(tick), "grid");
    text(svg, margin.left - 10, sy(tick) + 4, fmt(tick), "tick-label", "end");
  }
  for (const tick of xTicks(term, xMin, xMax)) {
    line(svg, sx(tick.value), margin.top + innerH, sx(tick.value), margin.top + innerH + 5, "tick");
    text(svg, sx(tick.value), margin.top + innerH + 22, tick.label, "tick-label", "middle");
  }
  const baseline = Math.min(Math.max(1, yMin), yMax);
  line(svg, margin.left, sy(baseline), margin.left + innerW, sy(baseline), "zero");
  line(svg, margin.left, margin.top, margin.left, margin.top + innerH, "axis");
  line(svg, margin.left, margin.top + innerH, margin.left + innerW, margin.top + innerH, "axis");

  text(svg, width / 2, 24, term.title, "label", "middle");
  text(svg, width / 2, height - 20, term.x_label, "label", "middle");
  const yLabel = text(svg, 22, margin.top + innerH / 2, term.y_label, "label", "middle");
  yLabel.setAttribute("transform", `rotate(-90 22 ${margin.top + innerH / 2})`);

  if (context.showCi() && term.ci_lower_y && term.ci_upper_y) {
    if (term.levels) {
      errorBars(svg, x, term.ci_lower_y, term.ci_upper_y, sx, sy);
    } else {
      band(svg, x, term.ci_lower_y, term.ci_upper_y, sx, sy, "ci");
    }
  }
  if (modeSelect.value === "handles" && context.showContrib && context.showContrib()) {
    basisContributions(svg, term, sx, sy, buildActive);
  }
  if (modeSelect.value === "handles" && buildProgress !== null) {
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
  if (!buildActive) path(svg, x, y, sx, sy, "edited");
  const selectedBounds = selectionBounds(x, y, selection, sx, sy, margin, innerW, innerH);
  const handlesMode = modeSelect.value === "handles" && term.controls;
  if (!handlesMode && selectedBounds) drawSelectionBounds(svg, selectedBounds);
  if (!handlesMode) drawLevelGroups(svg, term, sx, sy);
  const visiblePoints = visiblePointIndices(term, selection);
  const selectedPoints = [];
  const unselectedPoints = [];
  if (!handlesMode) {
    for (const i of visiblePoints) {
      if (selection.has(i)) selectedPoints.push(i);
      else unselectedPoints.push(i);
    }
    for (const i of unselectedPoints) drawPoint(svg, x, y, sx, sy, i, false);
    for (const i of selectedPoints) drawPoint(svg, x, y, sx, sy, i, true);
  } else {
    drawControlHandles(svg, term, sx, sy, margin, innerH);
  }
  applyPlotClip(svg);
  legend(svg, width - 145, 26);

  svg._scale = {
    sx, sy, x, y, xMin, xMax, yMin, yMax,
    baseXMin, baseXMax, baseYMin, baseYMax,
    margin, innerW, innerH
  };
  positionSelectionMenu(svg, context.selectionMenu, handlesMode ? null : selectedBounds);
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
  svg.appendChild(defs);
}

function applyPlotClip(svg) {
  const clipped = [
    ".original",
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
}

function visiblePointIndices(term, selection) {
  // Large continuous grids draw a representative point subset for performance,
  // but selected points are always forced visible.
  const base = term.handle_indices || term.x.map((_, i) => i);
  const out = new Set(base);
  if (selection.size <= base.length) {
    for (const i of selection) out.add(i);
  }
  return Array.from(out).sort((a, b) => a - b);
}

function drawPoint(svg, x, y, sx, sy, i, selected) {
  const circle = el("circle", {
    cx: sx(x[i]), cy: sy(y[i]), r: selected ? 4.6 : 3.4,
    class: selected ? "point selected" : "point",
    "data-index": i
  });
  svg.appendChild(circle);
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

function drawSelectionBounds(svg, bounds) {
  for (const className of ["selection-bounds-halo", "selection-bounds"]) {
    svg.appendChild(el("rect", {
      x: bounds.x,
      y: bounds.y,
      width: bounds.width,
      height: bounds.height,
      rx: 4,
      ry: 4,
      class: className
    }));
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
  const menuBox = selectionMenu.getBoundingClientRect();
  const parentBox = (selectionMenu.offsetParent || svg.parentElement).getBoundingClientRect();
  const topLeft = svgClientPoint(svg, bounds.x, bounds.y);
  const bottomRight = svgClientPoint(svg, bounds.x + bounds.width, bounds.y + bounds.height);
  const pad = 8;
  const localLeft = topLeft.x - parentBox.left;
  const localTop = topLeft.y - parentBox.top;
  const localRight = bottomRight.x - parentBox.left;
  const localBottom = bottomRight.y - parentBox.top;
  let left = (localLeft + localRight) / 2 - menuBox.width / 2;
  let top = localTop - menuBox.height - 12;
  if (top < pad) top = localBottom + 12;
  left = Math.max(pad, Math.min(chartBox.width - menuBox.width - pad, left));
  top = Math.max(pad, Math.min(chartBox.height - menuBox.height - pad, top));
  selectionMenu.style.left = `${left}px`;
  selectionMenu.style.top = `${top}px`;
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

function xTicks(term, xMin, xMax) {
  if (term.levels && term.levels.length <= 14) {
    return term.x.map((value, i) => ({ value, label: term.levels[i] }));
  }
  return ticks(xMin, xMax, 6).map((value) => ({ value, label: fmt(value) }));
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
  const maxH = innerH * 0.22;
  const exposureY = (v) => yBase - maxH * v / maxWeight;
  if (exposure.kind === "density") {
    exposureDensity(svg, x, exposure.y, sx, exposureY, yBase);
  } else {
    const nominalW = x.length > 1
      ? Math.abs(sx(x[1]) - sx(x[0])) * 0.82
      : innerW * 0.55;
    for (let i = 0; i < exposure.y.length; i++) {
      const h = Math.max(1, maxH * exposure.y[i] / maxWeight);
      svg.appendChild(el("rect", {
        x: sx(x[i]) - nominalW / 2,
        y: yBase - h,
        width: nominalW,
        height: h,
        class: "exposure"
      }));
    }
  }
  exposureAxis(svg, margin.left + innerW, yBase, maxH, maxWeight, exposure.label || "exposure");
}

function exposureDensity(svg, x, y, sx, exposureY, yBase) {
  const top = x.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(v).toFixed(2)} ${exposureY(y[i]).toFixed(2)}`).join(" ");
  const right = `L ${sx(x[x.length - 1]).toFixed(2)} ${yBase.toFixed(2)}`;
  const left = `L ${sx(x[0]).toFixed(2)} ${yBase.toFixed(2)} Z`;
  svg.appendChild(el("path", { d: `${top} ${right} ${left}`, class: "exposure-density" }));
}

function exposureAxis(svg, x, yBase, maxH, maxWeight, label) {
  line(svg, x, yBase - maxH, x, yBase, "exposure-axis");
  for (const value of [0, maxWeight / 2, maxWeight]) {
    const y = yBase - maxH * value / maxWeight;
    line(svg, x, y, x + 5, y, "exposure-axis");
    text(svg, x + 8, y + 4, fmt(value), "tick-label", "start");
  }
  const labelNode = text(svg, x + 46, yBase - maxH / 2, label, "label", "middle");
  labelNode.setAttribute("transform", `rotate(-90 ${x + 46} ${yBase - maxH / 2})`);
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

function line(svg, x1, y1, x2, y2, cls) {
  svg.appendChild(el("line", { x1, y1, x2, y2, class: cls }));
}

function text(svg, x, y, value, cls, anchor) {
  const node = el("text", { x, y, class: cls, "text-anchor": anchor });
  node.textContent = value;
  svg.appendChild(node);
  return node;
}

function legend(svg, x, y) {
  line(svg, x, y, x + 28, y, "original");
  text(svg, x + 36, y + 4, "original", "legend", "start");
  line(svg, x, y + 22, x + 28, y + 22, "edited");
  text(svg, x + 36, y + 26, "edited", "legend", "start");
  svg.appendChild(el("circle", { cx: x + 14, cy: y + 44, r: 4.6, class: "point selected" }));
  text(svg, x + 36, y + 48, "selected", "legend", "start");
}

function el(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
