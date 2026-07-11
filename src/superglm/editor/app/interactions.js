export function bindInteractions(context) {
  // One active interaction at a time. Drag previews are private payload clones;
  // only the action controller may replace the confirmed remote snapshot.
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
  const { svg } = context;
  const wheelOptions = { passive: false };

  async function onPointerDown(event) {
    const activeTerm = context.currentTerm();
    if (!activeTerm) return;
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
    const mode = context.mode();
    if (mode === "handles") {
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      if (controlIndex === undefined) return;
      const preview = structuredClone(activeTerm);
      if (!preview.controls) return;
      const i = Number(controlIndex);
      interaction.controlDrag = {
        term: context.selectedTerm(),
        preview,
        index: i,
        startValue: preview.controls.y[i],
        value: preview.controls.y[i],
        baseY: preview.y.slice(),
        basis: preview.controls.basis ? preview.controls.basis[i] : null
      };
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (mode === "move" && index !== undefined) {
      interaction.pendingClickIndex = null;
      clearBoxZoom(interaction);
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      const selection = context.currentSelection();
      const displayIndex = Number(index);
      const sourceForPoint = sourceIndicesForDisplayIndex(context, displayIndex);
      const selectionTouchesPoint = sourceForPoint.some((sourceIndex) => selection.has(sourceIndex));
      const indices = selectionTouchesPoint
        ? Array.from(selection).sort((a, b) => a - b)
        : sourceForPoint;
      const preview = structuredClone(activeTerm);
      interaction.pointDrag = {
        term: context.selectedTerm(),
        preview,
        selection: selectionTouchesPoint ? selection : new Set(sourceForPoint),
        displayIndex,
        sourceForPoint,
        startValue: displayedTermValue(preview, displayIndex, context),
        indices,
        values: indices.map((j) => preview.y[j]),
        delta: 0
      };
      if (!selectionTouchesPoint) {
        context.drawChart(preview, interaction.pointDrag.selection);
      }
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (mode === "zoom") {
      interaction.pendingClickIndex = null;
      interaction.dragStart = null;
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      beginBoxZoom(context, interaction, event);
      svg.setPointerCapture(event.pointerId);
      return;
    }
    if (mode !== "select") {
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
      const source = sourceIndicesForDisplayIndex(context, Number(index));
      const indices = toggleSourceSelection(context.currentSelection(), source);
      await context.actions.executeStateMutation({
        name: "select",
        path: "/select",
        payload: { term: context.selectedTerm(), indices }
      });
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
  }

  function onPointerMove(event) {
    if (interaction.panDrag) {
      panZoomView(context, interaction, svgPoint(context, event));
      return;
    }
    if (interaction.controlDrag) {
      const drag = interaction.controlDrag;
      const term = drag.preview;
      const value = yFromPoint(context, svgPoint(context, event));
      drag.value = value;
      if (term.controls && term.controls.y) {
        term.controls.y[drag.index] = value;
      }
      if (term.controls && term.controls.log_effect) {
        const logValue = Math.log(Math.max(value, 1e-12));
        term.controls.log_effect[drag.index] = logValue;
        if (term.controls.build_log_effect && term.controls.basis_index) {
          const buildIndex = term.controls.basis_index[drag.index];
          term.controls.build_log_effect[buildIndex] = logValue;
        }
      }
      previewControlCurve(term, drag, value);
      context.setPreviewTerm(drag.term, term, Array.from(context.currentSelection()));
      return;
    }
    if (interaction.pointDrag) {
      const drag = interaction.pointDrag;
      const term = drag.preview;
      const value = yFromPoint(context, svgPoint(context, event));
      drag.delta = value - drag.startValue;
      updateDraggedSourceValues(term, drag, context);
      context.setPreviewTerm(drag.term, term, Array.from(drag.selection));
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
  }

  async function onPointerUp(event) {
    if (interaction.panDrag) {
      interaction.panDrag = null;
      return;
    }
    if (interaction.controlDrag) {
      const drag = interaction.controlDrag;
      const term = drag.preview;
      const payload = {
        term: drag.term,
        handle_index: drag.index,
        value: drag.value,
        handle_count: term.controls ? term.controls.count : undefined
      };
      interaction.controlDrag = null;
      await context.actions.executeStateMutation({ name: "control", path: "/control", payload });
      return;
    }
    if (interaction.pointDrag) {
      const drag = interaction.pointDrag;
      const term = drag.preview;
      const displayValue = displayedTermValue(term, drag.displayIndex, context);
      const payload = {
        term: drag.term,
        indices: drag.indices,
        values: valuesForSourceIndices(
          term,
          drag.indices,
          drag.sourceForPoint,
          displayValue
        )
      };
      interaction.pointDrag = null;
      await context.actions.executeStateMutation({ name: "drag", path: "/drag", payload });
      return;
    }
    if (interaction.orderDrag) {
      const drag = interaction.orderDrag;
      updateOrderDrag(context, interaction, svgPoint(context, event));
      clearOrderDropPreview(interaction);
      interaction.orderDrag = null;
      if (drag.active && drag.targetIndex !== null) {
        await context.actions.executeStateMutation({
          name: "reorder_levels",
          path: "/reorder_levels",
          payload: { term: context.selectedTerm(), target_index: drag.targetIndex }
        });
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
    const displayIndices = moved ? indicesInBox(context, interaction.dragStart, point) : (
      interaction.pendingClickIndex === null ? null : [interaction.pendingClickIndex]
    );
    if (interaction.brush) interaction.brush.remove();
    interaction.brush = null;
    interaction.dragStart = null;
    interaction.pendingClickIndex = null;
    if (displayIndices === null) return;
    const indices = sourceIndicesForDisplayIndices(context, displayIndices);
    await context.actions.executeStateMutation({
      name: "select",
      path: "/select",
      payload: { term: context.selectedTerm(), indices }
    });
  }

  function onWheel(event) {
    if (!context.currentTerm() || !svg._scale) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    zoomAround(context, svgPoint(context, event), factor);
  }

  function onPointerCancel() {
    cancelActiveInteraction(context, interaction);
  }

  function onLostPointerCapture() {
    if (hasActiveInteraction(interaction)) {
      cancelActiveInteraction(context, interaction);
    }
  }

  svg.addEventListener("pointerdown", onPointerDown);
  svg.addEventListener("pointermove", onPointerMove);
  svg.addEventListener("pointerup", onPointerUp);
  svg.addEventListener("pointercancel", onPointerCancel);
  svg.addEventListener("lostpointercapture", onLostPointerCapture);
  svg.addEventListener("wheel", onWheel, wheelOptions);

  return {
    resetZoomView: () => context.clearZoom(context.selectedTerm()),
    destroy() {
      svg.removeEventListener("pointerdown", onPointerDown);
      svg.removeEventListener("pointermove", onPointerMove);
      svg.removeEventListener("pointerup", onPointerUp);
      svg.removeEventListener("pointercancel", onPointerCancel);
      svg.removeEventListener("lostpointercapture", onLostPointerCapture);
      svg.removeEventListener("wheel", onWheel, wheelOptions);
      cancelActiveInteraction(context, interaction);
    }
  };
}

function hasActiveInteraction(interaction) {
  return Boolean(
    interaction.dragStart ||
    interaction.brush ||
    interaction.pointDrag ||
    interaction.controlDrag ||
    interaction.panDrag ||
    interaction.zoomBox ||
    interaction.orderDrag ||
    interaction.pendingClickIndex !== null
  );
}

function cancelActiveInteraction(context, interaction) {
  const hadPreview = Boolean(interaction.pointDrag || interaction.controlDrag);
  if (interaction.brush) interaction.brush.remove();
  clearBoxZoom(interaction);
  clearOrderDropPreview(interaction);
  interaction.dragStart = null;
  interaction.brush = null;
  interaction.pointDrag = null;
  interaction.controlDrag = null;
  interaction.panDrag = null;
  interaction.orderDrag = null;
  interaction.pendingClickIndex = null;
  if (hadPreview) context.clearPreviewTerm();
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

function toggleSourceSelection(selection, sourceIndices) {
  const next = new Set(selection);
  const allSelected = sourceIndices.every((index) => next.has(index));
  for (const index of sourceIndices) {
    if (allSelected) next.delete(index);
    else next.add(index);
  }
  return Array.from(next).sort((a, b) => a - b);
}

function sourceIndicesForDisplayIndex(context, displayIndex) {
  const term = context.currentTerm();
  const index = Number(displayIndex);
  const scale = context.svg._scale || {};
  if (!scale.displayIsCollapsed) return expandedGroupSourceForIndex(term, index);
  const mapping = scale.displayToSourceIndices;
  if (!Array.isArray(mapping)) return [index];
  const source = mapping[index];
  if (!Array.isArray(source) || !source.length) return [index];
  return source.map(Number).filter((i) => Number.isInteger(i) && i >= 0);
}

function sourceIndicesForDisplayIndices(context, displayIndices) {
  const out = new Set();
  for (const displayIndex of displayIndices) {
    for (const sourceIndex of sourceIndicesForDisplayIndex(context, displayIndex)) {
      out.add(sourceIndex);
    }
  }
  return Array.from(out).sort((a, b) => a - b);
}

function expandedGroupSourceForIndex(term, index) {
  const groups = Array.isArray(term && term.level_groups) ? term.level_groups : [];
  for (const group of groups) {
    const indices = Array.isArray(group.indices) ? group.indices.map(Number) : [];
    if (indices.includes(Number(index))) return indices;
  }
  return [Number(index)];
}

function displayedTermValue(term, displayIndex, context) {
  const scale = context.svg._scale || {};
  if (!scale.displayIsCollapsed) return term.y[displayIndex];
  const collapsed = term.group_display && term.group_display.collapsed;
  if (!collapsed || !Array.isArray(collapsed.y)) return term.y[displayIndex];
  return collapsed.y[displayIndex];
}

function updateDraggedSourceValues(term, drag, context) {
  for (let k = 0; k < drag.indices.length; k++) {
    term.y[drag.indices[k]] = Math.max(1e-12, drag.values[k] + drag.delta);
  }
  syncCollapsedDisplayFromRaw(term, context, drag.indices);
}

function syncCollapsedDisplayFromRaw(term, context, sourceIndices) {
  const scale = context.svg._scale || {};
  if (!scale.displayIsCollapsed) return;
  const collapsed = term.group_display && term.group_display.collapsed;
  if (!collapsed || !Array.isArray(collapsed.y)) return;
  const touched = new Set(sourceIndices);
  const mapping = Array.isArray(scale.displayToSourceIndices) ? scale.displayToSourceIndices : [];
  for (let displayIndex = 0; displayIndex < mapping.length; displayIndex++) {
    const source = Array.isArray(mapping[displayIndex]) ? mapping[displayIndex].map(Number) : [];
    if (!source.some((index) => touched.has(index))) continue;
    const values = source.map((index) => term.y[index]).filter(Number.isFinite);
    if (values.length) {
      collapsed.y[displayIndex] = values.reduce((acc, value) => acc + value, 0) / values.length;
    }
  }
}

function valuesForSourceIndices(term, sourceIndices, sourceForMovedDisplayPoint, displayValue) {
  const moved = new Set(sourceForMovedDisplayPoint);
  return sourceIndices.map((sourceIndex) => (
    moved.has(sourceIndex) && Number.isFinite(displayValue) ? displayValue : term.y[sourceIndex]
  ));
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
  const scale = context.svg._scale || {};
  if (scale.displayIsCollapsed) return false;
  if (!term || !Array.isArray(term.levels) || !term.levels.length) return false;
  if ((term.term_type || term.kind || "") !== "categorical") return false;
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
    context.clearZoom(context.selectedTerm());
    return;
  }
  xMin = Math.max(scale.baseXMin, xMin);
  xMax = Math.min(scale.baseXMax, xMax);
  yMin = Math.max(scale.baseYMin, yMin);
  yMax = Math.min(scale.baseYMax, yMax);
  if (xMax <= xMin || yMax <= yMin) return;
  context.setZoom(context.selectedTerm(), { xMin, xMax, yMin, yMax });
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
  context.setZoom(context.selectedTerm(), { xMin, xMax, yMin, yMax });
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
  context.setZoom(context.selectedTerm(), { xMin, xMax, yMin, yMax });
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

function svgRect(attrs) {
  return svgNode("rect", attrs);
}

function svgNode(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
