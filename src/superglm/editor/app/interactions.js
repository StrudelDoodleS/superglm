export function bindInteractions(context) {
  // One active interaction at a time. During drags we mutate the local payload
  // for immediate visual feedback, then POST the final values so Python can
  // commit history, metrics, and summary invalidation.
  const interaction = {
    dragStart: null,
    brush: null,
    pointDrag: null,
    controlDrag: null,
    panDrag: null,
    zoomBox: null,
    orderDrag: null,
    pendingClickIndex: null
  };
  const { svg, modeSelect } = context;

  svg.addEventListener("pointerdown", async (event) => {
    if (!context.hasState()) return;
    if ((event.shiftKey || event.button === 1) && beginPan(context, interaction, event)) {
      event.preventDefault();
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      interaction.pointDrag = null;
      interaction.controlDrag = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      svg.setPointerCapture(event.pointerId);
      return;
    }
    const index = event.target && event.target.dataset ? event.target.dataset.index : undefined;
    const controlIndex = event.target && event.target.dataset ? event.target.dataset.controlIndex : undefined;
    if (modeSelect.value === "handles") {
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      if (controlIndex === undefined) return;
      const term = context.currentTerm();
      if (!term.controls) return;
      const i = Number(controlIndex);
      interaction.controlDrag = {
        index: i,
        startValue: term.controls.y[i],
        value: term.controls.y[i],
        baseY: term.y.slice(),
        basis: term.controls.basis ? term.controls.basis[i] : null
      };
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (modeSelect.value === "move" && index !== undefined) {
      interaction.pendingClickIndex = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      const i = Number(index);
      const term = context.currentTerm();
      const selection = context.currentSelection();
      const indices = selection.has(i) ? Array.from(selection).sort((a, b) => a - b) : [i];
      interaction.pointDrag = {
        index: i,
        startValue: term.y[i],
        indices,
        values: indices.map((j) => term.y[j]),
        delta: 0
      };
      if (!selection.has(i)) {
        context.getState().selection[context.selectedTerm()] = [i];
        context.render();
      }
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (modeSelect.value === "zoom") {
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      beginBoxZoom(context, interaction, event);
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (modeSelect.value !== "select") {
      interaction.pendingClickIndex = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      return;
    }
    if (index !== undefined && isModifierLevelSelection(event, context.currentTerm())) {
      event.preventDefault();
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      interaction.brush = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      const indices = togglePointSelection(context.currentSelection(), Number(index));
      await context.postJSON("/select", { term: context.selectedTerm(), indices });
      return;
    }
    if (index !== undefined && beginOrderDrag(context, interaction, event, Number(index))) {
      event.preventDefault();
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      interaction.brush = null;
      clearBoxZoom(interaction);
      svg.setPointerCapture(event.pointerId);
      return;
    }
    interaction.pendingClickIndex = index !== undefined ? Number(index) : null;
    interaction.dragStart = svgPoint(context, event);
    interaction.brush = svgRect({
      class: "brush",
      x: interaction.dragStart.x,
      y: interaction.dragStart.y,
      width: 0,
      height: 0
    });
    svg.appendChild(interaction.brush);
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (interaction.panDrag) {
      panZoomView(context, interaction, svgPoint(context, event));
      return;
    }
    if (interaction.controlDrag) {
      const term = context.currentTerm();
      const value = yFromPoint(context, svgPoint(context, event));
      interaction.controlDrag.value = value;
      if (term.controls && term.controls.y) {
        term.controls.y[interaction.controlDrag.index] = value;
      }
      if (term.controls && term.controls.log_effect) {
        const logValue = Math.log(Math.max(value, 1e-12));
        term.controls.log_effect[interaction.controlDrag.index] = logValue;
        if (term.controls.build_log_effect && term.controls.basis_index) {
          const buildIndex = term.controls.basis_index[interaction.controlDrag.index];
          term.controls.build_log_effect[buildIndex] = logValue;
        }
      }
      previewControlCurve(term, interaction.controlDrag, value);
      context.drawChart(term, context.currentSelection());
      return;
    }
    if (interaction.pointDrag) {
      const term = context.currentTerm();
      const value = yFromPoint(context, svgPoint(context, event));
      interaction.pointDrag.delta = value - interaction.pointDrag.startValue;
      for (let k = 0; k < interaction.pointDrag.indices.length; k++) {
        term.y[interaction.pointDrag.indices[k]] = Math.max(
          1e-12,
          interaction.pointDrag.values[k] + interaction.pointDrag.delta
        );
      }
      context.drawChart(term, context.currentSelection());
      return;
    }
    if (interaction.orderDrag) {
      updateOrderDrag(context, interaction, svgPoint(context, event));
      return;
    }
    if (interaction.zoomBox) {
      updateBoxZoom(interaction, svgPoint(context, event));
      return;
    }
    if (!interaction.dragStart || !interaction.brush) return;
    const point = svgPoint(context, event);
    interaction.brush.setAttribute("x", Math.min(interaction.dragStart.x, point.x));
    interaction.brush.setAttribute("y", Math.min(interaction.dragStart.y, point.y));
    interaction.brush.setAttribute("width", Math.abs(point.x - interaction.dragStart.x));
    interaction.brush.setAttribute("height", Math.abs(point.y - interaction.dragStart.y));
  });

  svg.addEventListener("pointerup", async (event) => {
    if (interaction.panDrag) {
      interaction.panDrag = null;
      return;
    }
    if (interaction.controlDrag) {
      const term = context.currentTerm();
      const payload = {
        term: context.selectedTerm(),
        handle_index: interaction.controlDrag.index,
        value: interaction.controlDrag.value,
        handle_count: term.controls ? term.controls.count : undefined
      };
      interaction.controlDrag = null;
      await context.postJSON("/control", payload, { refreshMetrics: true, refreshSummary: true });
      return;
    }
    if (interaction.pointDrag) {
      const payload = {
        term: context.selectedTerm(),
        indices: interaction.pointDrag.indices,
        values: interaction.pointDrag.indices.map((i) => context.currentTerm().y[i])
      };
      interaction.pointDrag = null;
      await context.postJSON("/drag", payload, { refreshMetrics: true, refreshSummary: true });
      return;
    }
    if (interaction.orderDrag) {
      const drag = interaction.orderDrag;
      updateOrderDrag(context, interaction, svgPoint(context, event));
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      if (drag.active && drag.targetIndex !== null) {
        await context.postJSON(
          "/reorder_levels",
          { term: context.selectedTerm(), target_index: drag.targetIndex },
          { refreshMetrics: false, refreshSummary: false }
        );
      }
      return;
    }
    if (interaction.zoomBox) {
      const zoomBox = interaction.zoomBox;
      const point = svgPoint(context, event);
      clearBoxZoom(interaction);
      applyBoxZoom(context, zoomBox.start, point);
      return;
    }
    if (!interaction.dragStart) return;
    const point = svgPoint(context, event);
    const moved = Math.abs(point.x - interaction.dragStart.x) > 3 ||
      Math.abs(point.y - interaction.dragStart.y) > 3;
    const indices = moved ? indicesInBox(context, interaction.dragStart, point) : (
      interaction.pendingClickIndex === null ? null : [interaction.pendingClickIndex]
    );
    if (interaction.brush) interaction.brush.remove();
    interaction.brush = null;
    interaction.dragStart = null;
    interaction.pendingClickIndex = null;
    if (indices === null) return;
    await context.postJSON("/select", { term: context.selectedTerm(), indices });
  });

  svg.addEventListener("wheel", (event) => {
    if (!context.hasState() || !svg._scale) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    zoomAround(context, svgPoint(context, event), factor);
  }, { passive: false });

  document.addEventListener("keydown", async (event) => {
    if (!context.hasState() || isEditableTarget(event.target)) return;
    const primary = event.ctrlKey || event.metaKey;
    if (!primary || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "z" && !event.shiftKey) {
      event.preventDefault();
      await context.postJSON("/op", { operation: "undo" }, {
        refreshMetrics: true,
        refreshSummary: true
      });
    } else if (key === "y" || (key === "z" && event.shiftKey)) {
      event.preventDefault();
      await context.postJSON("/op", { operation: "redo" }, {
        refreshMetrics: true,
        refreshSummary: true
      });
    }
  });

  return {
    resetZoomView: () => resetZoomView(context)
  };
}

function isModifierLevelSelection(event, term) {
  return Boolean(
    term &&
    Array.isArray(term.levels) &&
    term.levels.length > 0 &&
    (event.ctrlKey || event.metaKey)
  );
}

function togglePointSelection(selection, index) {
  const next = new Set(selection);
  if (next.has(index)) next.delete(index);
  else next.add(index);
  return Array.from(next).sort((a, b) => a - b);
}

function beginBoxZoom(context, interaction, event) {
  if (!context.svg._scale) return;
  const start = svgPoint(context, event);
  const brush = svgRect({
    class: "brush box-zoom",
    x: start.x,
    y: start.y,
    width: 0,
    height: 0
  });
  context.svg.appendChild(brush);
  interaction.zoomBox = { start, brush };
}

function updateBoxZoom(interaction, point) {
  const zoomBox = interaction.zoomBox;
  if (!zoomBox || !zoomBox.brush) return;
  zoomBox.brush.setAttribute("x", Math.min(zoomBox.start.x, point.x));
  zoomBox.brush.setAttribute("y", Math.min(zoomBox.start.y, point.y));
  zoomBox.brush.setAttribute("width", Math.abs(point.x - zoomBox.start.x));
  zoomBox.brush.setAttribute("height", Math.abs(point.y - zoomBox.start.y));
}

function clearBoxZoom(interaction) {
  if (interaction.zoomBox && interaction.zoomBox.brush) {
    interaction.zoomBox.brush.remove();
  }
  interaction.zoomBox = null;
}

function beginOrderDrag(context, interaction, event, index) {
  const term = context.currentTerm();
  if (!term || !Array.isArray(term.levels) || !term.levels.length) return false;
  const selection = context.currentSelection();
  if (!selection.has(index)) return false;
  clearOrderDropPreview(interaction);
  interaction.orderDrag = {
    start: svgPoint(context, event),
    indices: Array.from(selection).sort((a, b) => a - b),
    targetIndex: index,
    active: false,
    preview: null
  };
  return true;
}

function updateOrderDrag(context, interaction, point) {
  const drag = interaction.orderDrag;
  if (!drag) return;
  const moved = Math.abs(point.x - drag.start.x) > 4 || Math.abs(point.y - drag.start.y) > 4;
  drag.active = drag.active || moved;
  drag.targetIndex = targetIndexFromPoint(context, point);
  if (drag.active) drawOrderDropPreview(context, drag);
  else clearOrderDropPreview(interaction);
}

function targetIndexFromPoint(context, point) {
  const term = context.currentTerm();
  const levels = term && Array.isArray(term.levels) ? term.levels : [];
  if (!levels.length) return 0;
  const data = dataFromPoint(context, point);
  return Math.max(0, Math.min(levels.length, Math.round(data.x)));
}

function drawOrderDropPreview(context, drag) {
  const term = context.currentTerm();
  const scale = context.svg._scale;
  if (!term || !Array.isArray(term.levels) || !scale) return;
  clearOrderDropPreview({ orderDrag: drag });
  const geometry = orderDropGeometry(term, scale, drag);
  if (!geometry) return;
  const group = svgNode("g", {
    class: "order-drop-preview",
    "data-target-index": drag.targetIndex
  });
  group.appendChild(svgNode("rect", {
    class: "order-drop-ghost",
    x: geometry.ghostX,
    y: scale.margin.top,
    width: geometry.ghostW,
    height: scale.innerH,
    rx: 5,
    ry: 5
  }));
  group.appendChild(svgNode("line", {
    class: "order-drop-rail",
    x1: geometry.railX,
    y1: scale.margin.top,
    x2: geometry.railX,
    y2: scale.margin.top + scale.innerH
  }));
  group.appendChild(svgNode("path", {
    class: "order-drop-arrow",
    d: [
      `M ${geometry.railX - 5} ${scale.margin.top + 8}`,
      `L ${geometry.railX} ${scale.margin.top + 2}`,
      `L ${geometry.railX + 5} ${scale.margin.top + 8}`
    ].join(" ")
  }));
  context.svg.appendChild(group);
  drag.preview = group;
}

function clearOrderDropPreview(interaction) {
  const drag = interaction.orderDrag;
  if (drag && drag.preview) {
    drag.preview.remove();
    drag.preview = null;
  }
}

function orderDropGeometry(term, scale, drag) {
  const n = term.levels.length;
  if (!n || !drag.indices.length) return null;
  const selected = new Set(drag.indices);
  const remaining = Array.from({ length: n }, (_, i) => i).filter((i) => !selected.has(i));
  const rawTarget = Number(drag.targetIndex);
  const target = Math.max(0, Math.min(n, Number.isFinite(rawTarget) ? Math.round(rawTarget) : 0));
  const insertAt = remaining.filter((i) => i < target).length;
  const dataStep = categoricalDataStep(term);
  const pixelStep = Math.abs(scale.sx(term.x[0] + dataStep) - scale.sx(term.x[0]));
  const count = drag.indices.length;
  const firstSlot = Number(term.x[0] || 0) + insertAt * dataStep;
  const ghostX = scale.sx(firstSlot - dataStep / 2);
  const ghostW = Math.max(10, pixelStep * count);
  const railX = ghostX;
  return {
    ghostX,
    ghostW,
    railX
  };
}

function categoricalDataStep(term) {
  const x = Array.isArray(term.x) ? term.x.map(Number).filter(Number.isFinite) : [];
  if (x.length < 2) return 1;
  const diffs = [];
  for (let i = 1; i < x.length; i++) {
    const diff = Math.abs(x[i] - x[i - 1]);
    if (diff > 1e-12) diffs.push(diff);
  }
  return diffs.length ? Math.min(...diffs) : 1;
}

function svgPoint(context, event) {
  // Pointer events arrive in browser client coordinates. Use the SVG matrix
  // when available so selection boxes and handles stay aligned after resizing.
  const matrix = context.svg.getScreenCTM();
  if (matrix) {
    const point = context.svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const svgPoint = point.matrixTransform(matrix.inverse());
    return { x: svgPoint.x, y: svgPoint.y };
  }
  const box = context.svg.getBoundingClientRect();
  const viewBox = context.svg.viewBox.baseVal;
  return {
    x: viewBox.x + (event.clientX - box.left) * viewBox.width / Math.max(box.width, 1e-12),
    y: viewBox.y + (event.clientY - box.top) * viewBox.height / Math.max(box.height, 1e-12)
  };
}

function yFromPoint(context, point) {
  const { yMin, yMax, margin, innerH } = context.svg._scale;
  return Math.max(
    1e-12,
    yMax - ((point.y - margin.top) / Math.max(innerH, 1e-12)) * (yMax - yMin)
  );
}

function dataFromPoint(context, point) {
  const { xMin, xMax, yMin, yMax, margin, innerW, innerH } = context.svg._scale;
  const x = xMin + ((point.x - margin.left) / Math.max(innerW, 1e-12)) * (xMax - xMin);
  const y = yMax - ((point.y - margin.top) / Math.max(innerH, 1e-12)) * (yMax - yMin);
  return { x, y };
}

function zoomAround(context, point, factor) {
  // Zoom is stored per term. The base ranges come from the current payload and
  // act as hard bounds so users can always get home.
  if (!context.svg._scale) return;
  const scale = context.svg._scale;
  const center = dataFromPoint(context, point);
  const xRange = scale.xMax - scale.xMin;
  const yRange = scale.yMax - scale.yMin;
  const minXRange = (scale.baseXMax - scale.baseXMin) / 200;
  const minYRange = (scale.baseYMax - scale.baseYMin) / 200;
  let xMin = center.x - (center.x - scale.xMin) * factor;
  let xMax = center.x + (scale.xMax - center.x) * factor;
  let yMin = center.y - (center.y - scale.yMin) * factor;
  let yMax = center.y + (scale.yMax - center.y) * factor;
  if (xMax - xMin < minXRange || yMax - yMin < minYRange) return;
  if (factor > 1 &&
      xRange >= scale.baseXMax - scale.baseXMin &&
      yRange >= scale.baseYMax - scale.baseYMin) {
    resetZoomView(context);
    return;
  }
  xMin = Math.max(scale.baseXMin, xMin);
  xMax = Math.min(scale.baseXMax, xMax);
  yMin = Math.max(scale.baseYMin, yMin);
  yMax = Math.min(scale.baseYMax, yMax);
  if (xMax <= xMin || yMax <= yMin) return;
  context.zoomState[context.selectedTerm()] = { xMin, xMax, yMin, yMax };
  context.render();
}

function applyBoxZoom(context, start, end) {
  if (!context.svg._scale) return;
  const scale = context.svg._scale;
  const x0 = Math.max(scale.margin.left, Math.min(start.x, end.x));
  const x1 = Math.min(scale.margin.left + scale.innerW, Math.max(start.x, end.x));
  const y0 = Math.max(scale.margin.top, Math.min(start.y, end.y));
  const y1 = Math.min(scale.margin.top + scale.innerH, Math.max(start.y, end.y));
  if (x1 - x0 < 8 || y1 - y0 < 8) return;
  const lo = dataFromPoint(context, { x: x0, y: y1 });
  const hi = dataFromPoint(context, { x: x1, y: y0 });
  const xMin = Math.max(scale.baseXMin, Math.min(lo.x, hi.x));
  const xMax = Math.min(scale.baseXMax, Math.max(lo.x, hi.x));
  const yMin = Math.max(scale.baseYMin, Math.min(lo.y, hi.y));
  const yMax = Math.min(scale.baseYMax, Math.max(lo.y, hi.y));
  if (xMax <= xMin || yMax <= yMin) return;
  context.zoomState[context.selectedTerm()] = { xMin, xMax, yMin, yMax };
  context.render();
}

function beginPan(context, interaction, event) {
  // Panning is intentionally chorded behind Shift or middle click so ordinary
  // drag-select remains the default interaction.
  if (!context.svg._scale) return false;
  const scale = context.svg._scale;
  interaction.panDrag = {
    start: svgPoint(context, event),
    xMin: scale.xMin,
    xMax: scale.xMax,
    yMin: scale.yMin,
    yMax: scale.yMax,
    baseXMin: scale.baseXMin,
    baseXMax: scale.baseXMax,
    baseYMin: scale.baseYMin,
    baseYMax: scale.baseYMax,
    innerW: scale.innerW,
    innerH: scale.innerH
  };
  return true;
}

function panZoomView(context, interaction, point) {
  const panDrag = interaction.panDrag;
  if (!panDrag) return;
  const xRange = panDrag.xMax - panDrag.xMin;
  const yRange = panDrag.yMax - panDrag.yMin;
  const dx = (point.x - panDrag.start.x) / Math.max(panDrag.innerW, 1e-12) * xRange;
  const dy = (point.y - panDrag.start.y) / Math.max(panDrag.innerH, 1e-12) * yRange;
  let xMin = panDrag.xMin - dx;
  let xMax = panDrag.xMax - dx;
  let yMin = panDrag.yMin + dy;
  let yMax = panDrag.yMax + dy;
  [xMin, xMax] = clampPanRange(xMin, xMax, panDrag.baseXMin, panDrag.baseXMax);
  [yMin, yMax] = clampPanRange(yMin, yMax, panDrag.baseYMin, panDrag.baseYMax);
  context.zoomState[context.selectedTerm()] = { xMin, xMax, yMin, yMax };
  context.render();
}

function clampPanRange(min, max, baseMin, baseMax) {
  const range = max - min;
  const baseRange = baseMax - baseMin;
  if (!Number.isFinite(range) || range >= baseRange) return [baseMin, baseMax];
  if (min < baseMin) {
    max += baseMin - min;
    min = baseMin;
  }
  if (max > baseMax) {
    min -= max - baseMax;
    max = baseMax;
  }
  return [min, max];
}

function resetZoomView(context) {
  delete context.zoomState[context.selectedTerm()];
  context.render();
}

function previewControlCurve(term, drag, value) {
  // Raw-basis handles can preview by applying the moved basis contribution in
  // relativity space. Fallback handles post to Python before exact rebuilding.
  if (!drag.basis || !drag.baseY || drag.basis.length !== drag.baseY.length) return;
  const start = Math.max(drag.startValue, 1e-12);
  const next = Math.max(value, 1e-12);
  const deltaLog = Math.log(next) - Math.log(start);
  for (let i = 0; i < term.y.length; i++) {
    term.y[i] = Math.max(1e-12, drag.baseY[i] * Math.exp(drag.basis[i] * deltaLog));
  }
}

function indicesInBox(context, a, b) {
  // Selection is strict: only points whose displayed markers fall inside the
  // drag box are selected. Plain clicks outside points do nothing.
  const { sx, sy, x, y } = context.svg._scale;
  const x0 = Math.min(a.x, b.x), x1 = Math.max(a.x, b.x);
  const y0 = Math.min(a.y, b.y), y1 = Math.max(a.y, b.y);
  const indices = [];
  for (let i = 0; i < y.length; i++) {
    const px = sx(x[i]), py = sy(y[i]);
    if (px >= x0 && px <= x1 && py >= y0 && py <= y1) indices.push(i);
  }
  return indices;
}

function isEditableTarget(target) {
  if (!target) return false;
  const tag = target.tagName ? target.tagName.toLowerCase() : "";
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}

function svgRect(attrs) {
  return svgNode("rect", attrs);
}

function svgNode(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
