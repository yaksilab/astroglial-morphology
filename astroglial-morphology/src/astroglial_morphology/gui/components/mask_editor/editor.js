// Streamlit CCv2 mask editor.
//
// The component receives:
//   data.image     : PNG data URL of the projection
//   data.masks_b64 : base64 of Int32Array (row-major, length = height * width)
//   data.width     : image width
//   data.height    : image height
//   data.max_label : maximum label id currently in the mask
//
// It emits:
//   trigger "save" : { masks_b64, width, height, max_label }
//
// ROI counts stay in the browser status bar. Do not call setStateValue while
// painting — that reruns Streamlit and remounts the canvas from the original
// masks, wiping in-progress edits.
//
// Editing operations run entirely in the browser to keep the round-trip cheap.
// The mask is a Uint32Array kept in a WebGL-free pixel-perfect canvas.

export default function(component) {
  const {
    parentElement,
    data,
    setTriggerValue,
  } = component;

  const state = {
    width: data?.width || 0,
    height: data?.height || 0,
    maxLabel: data?.max_label || 0,
    masks: null,
    tool: "select",
    brushSize: 3,
    newLabel: false,
    alpha: 0.45,
    overlayOn: true,
    selection: new Set(),
    activeLabel: null,
    palette: buildPalette(4096),
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isPointerDown: false,
    lastPointer: null,
    strokeAffected: new Set(),
    strokeLabel: null,
    strokePoints: [],
    lastPaint: null,
    splitPoints: [],
    undoStack: [],
    redoStack: [],
  };

  const imageCanvas = parentElement.querySelector('canvas[data-role="image"]');
  const maskCanvas = parentElement.querySelector('canvas[data-role="masks"]');
  const overlayCanvas = parentElement.querySelector('canvas[data-role="overlay"]');
  const stage = parentElement.querySelector('.mask-editor__stage');
  const infoEl = parentElement.querySelector('[data-role="info"]');
  const selEl = parentElement.querySelector('[data-role="selection"]');
  const countsEl = parentElement.querySelector('[data-role="counts"]');
  const brushEl = parentElement.querySelector('[data-role="brush"]');
  const brushValEl = parentElement.querySelector('[data-role="brush-value"]');
  const alphaEl = parentElement.querySelector('[data-role="alpha"]');
  const alphaValEl = parentElement.querySelector('[data-role="alpha-value"]');
  const newLabelEl = parentElement.querySelector('[data-role="new-label"]');
  const overlayToggleEl = parentElement.querySelector('[data-role="overlay-toggle"]');

  brushEl.value = state.brushSize;
  brushValEl.textContent = state.brushSize;
  alphaEl.value = Math.round(state.alpha * 100);
  alphaValEl.textContent = alphaEl.value;

  const image = new Image();
  image.decoding = "async";
  image.onload = () => {
    resizeCanvases();
    drawImage();
    redrawMask();
    fitToStage();
  };
  image.src = data?.image || "";

  state.masks = decodeMaskB64(data?.masks_b64 || "", state.width * state.height);

  function decodeMaskB64(b64, expectedLength) {
    if (!b64) {
      return new Int32Array(expectedLength);
    }
    const bin = atob(b64);
    const buffer = new ArrayBuffer(bin.length);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bin.length; i += 1) {
      bytes[i] = bin.charCodeAt(i);
    }
    return new Int32Array(buffer);
  }

  function encodeMaskB64() {
    const bytes = new Uint8Array(state.masks.buffer);
    let bin = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(bin);
  }

  function buildPalette(n) {
    const palette = new Uint8Array(n * 3);
    palette[0] = 0;
    palette[1] = 0;
    palette[2] = 0;
    for (let i = 1; i < n; i += 1) {
      const hue = (i * 137.508) % 360;
      const [r, g, b] = hsvToRgb(hue, 0.75, 0.95);
      palette[i * 3] = r;
      palette[i * 3 + 1] = g;
      palette[i * 3 + 2] = b;
    }
    return palette;
  }

  function hsvToRgb(h, s, v) {
    const c = v * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r = 0, g = 0, b = 0;
    if (hp < 1) { r = c; g = x; }
    else if (hp < 2) { r = x; g = c; }
    else if (hp < 3) { g = c; b = x; }
    else if (hp < 4) { g = x; b = c; }
    else if (hp < 5) { r = x; b = c; }
    else { r = c; b = x; }
    const m = v - c;
    return [
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255),
    ];
  }

  function paletteFor(label) {
    if (label <= 0) return [0, 0, 0, 0];
    const idx = label % (state.palette.length / 3);
    return [
      state.palette[idx * 3],
      state.palette[idx * 3 + 1],
      state.palette[idx * 3 + 2],
      Math.round(255 * state.alpha),
    ];
  }

  function resizeCanvases() {
    const { width, height } = state;
    for (const canvas of [imageCanvas, maskCanvas, overlayCanvas]) {
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = "100%";
      canvas.style.height = "100%";
    }
  }

  function drawImage() {
    const ctx = imageCanvas.getContext("2d");
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.drawImage(image, 0, 0);
  }

  function redrawMask() {
    const ctx = maskCanvas.getContext("2d");
    const imgData = ctx.createImageData(state.width, state.height);
    const rgba = imgData.data;
    const masks = state.masks;
    for (let i = 0; i < masks.length; i += 1) {
      const label = masks[i];
      const p = i * 4;
      if (label <= 0) {
        rgba[p] = 0;
        rgba[p + 1] = 0;
        rgba[p + 2] = 0;
        rgba[p + 3] = 0;
        continue;
      }
      const [r, g, b] = paletteFor(label);
      const isSelected = state.selection.has(label);
      rgba[p] = r;
      rgba[p + 1] = g;
      rgba[p + 2] = b;
      rgba[p + 3] = isSelected
        ? Math.min(255, Math.round(255 * (state.alpha + 0.25)))
        : Math.round(255 * (state.alpha ?? 0.45));
    }
    ctx.putImageData(imgData, 0, 0);
    updateCounts();
  }

  function updateCounts() {
    const seen = new Set();
    for (let i = 0; i < state.masks.length; i += 1) {
      const label = state.masks[i];
      if (label > 0) seen.add(label);
    }
    countsEl.textContent = `ROIs: ${seen.size}`;
    if (state.selection.size === 0) {
      selEl.textContent = "Selection: none";
    } else {
      selEl.textContent = `Selection: ${Array.from(state.selection).join(", ")}`;
    }
  }

  function fitToStage() {
    stage.style.aspectRatio = `${state.width} / ${state.height}`;
    applyTransform();
  }

  function applyTransform() {
    const transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.scale})`;
    for (const canvas of [imageCanvas, maskCanvas, overlayCanvas]) {
      canvas.style.transformOrigin = "0 0";
      canvas.style.transform = transform;
    }
  }

  function eventToImageCoords(evt) {
    const rect = overlayCanvas.getBoundingClientRect();
    const relX = (evt.clientX - rect.left) / rect.width;
    const relY = (evt.clientY - rect.top) / rect.height;
    return {
      x: Math.max(0, Math.min(state.width - 1, Math.floor(relX * state.width))),
      y: Math.max(0, Math.min(state.height - 1, Math.floor(relY * state.height))),
    };
  }

  function saveHistory() {
    state.undoStack.push(state.masks.slice(0));
    if (state.undoStack.length > 20) state.undoStack.shift();
    state.redoStack.length = 0;
  }

  function undo() {
    if (state.undoStack.length === 0) return;
    state.redoStack.push(state.masks.slice(0));
    state.masks = state.undoStack.pop();
    redrawMask();
    setInfo("Undo");
  }

  function redo() {
    if (state.redoStack.length === 0) return;
    state.undoStack.push(state.masks.slice(0));
    state.masks = state.redoStack.pop();
    redrawMask();
    setInfo("Redo");
  }

  function setInfo(text) {
    infoEl.textContent = text;
  }

  function applyOverlayVisibility() {
    const visible = Boolean(state.overlayOn);
    maskCanvas.style.opacity = visible ? "1" : "0";
    maskCanvas.style.visibility = visible ? "visible" : "hidden";
    maskCanvas.classList.toggle("is-hidden", !visible);
    if (overlayToggleEl) {
      overlayToggleEl.classList.toggle("is-active", visible);
      overlayToggleEl.setAttribute("aria-pressed", visible ? "true" : "false");
    }
  }

  function toggleOverlay() {
    state.overlayOn = !state.overlayOn;
    applyOverlayVisibility();
    if (state.overlayOn) {
      redrawMask();
    }
    setInfo(state.overlayOn ? "Mask overlay on" : "Mask overlay off");
  }

  function nextLabel() {
    state.maxLabel += 1;
    return state.maxLabel;
  }

  function deleteSelection() {
    if (state.selection.size === 0) return;
    saveHistory();
    for (let i = 0; i < state.masks.length; i += 1) {
      if (state.selection.has(state.masks[i])) {
        state.masks[i] = 0;
      }
    }
    state.selection.clear();
    redrawMask();
    setInfo("Deleted selection");
  }

  function mergeSelection() {
    if (state.selection.size < 2) {
      setInfo("Select at least two ROIs to merge");
      return;
    }
    saveHistory();
    const target = Math.min(...state.selection);
    for (let i = 0; i < state.masks.length; i += 1) {
      if (state.selection.has(state.masks[i])) {
        state.masks[i] = target;
      }
    }
    state.selection.clear();
    state.selection.add(target);
    redrawMask();
    setInfo(`Merged into ${target}`);
  }

  function paintAt(x, y, radius, label) {
    const rr = radius * radius;
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        if (dx * dx + dy * dy > rr) continue;
        const px = x + dx;
        const py = y + dy;
        if (px < 0 || py < 0 || px >= state.width || py >= state.height) continue;
        const idx = py * state.width + px;
        if (label === 0) {
          if (state.masks[idx] !== 0) {
            state.strokeAffected.add(state.masks[idx]);
          }
          state.masks[idx] = 0;
        } else {
          state.strokeAffected.add(label);
          state.masks[idx] = label;
        }
      }
    }
  }

  function paintSegment(x0, y0, x1, y1, radius, label) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(dist));
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      paintAt(
        Math.round(x0 + (x1 - x0) * t),
        Math.round(y0 + (y1 - y0) * t),
        radius,
        label,
      );
    }
  }

  function fillPolygon(points, label) {
    if (points.length < 3 || !label) return;
    let minY = points[0].y;
    let maxY = points[0].y;
    for (let i = 1; i < points.length; i += 1) {
      const y = points[i].y;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    minY = Math.max(0, minY);
    maxY = Math.min(state.height - 1, maxY);
    const n = points.length;
    for (let y = minY; y <= maxY; y += 1) {
      const xs = [];
      for (let i = 0; i < n; i += 1) {
        const a = points[i];
        const b = points[(i + 1) % n];
        if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
          const dy = b.y - a.y;
          if (dy === 0) continue;
          xs.push(a.x + ((y - a.y) * (b.x - a.x)) / dy);
        }
      }
      xs.sort((left, right) => left - right);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const x0 = Math.max(0, Math.ceil(xs[k]));
        const x1 = Math.min(state.width - 1, Math.floor(xs[k + 1]));
        const row = y * state.width;
        for (let x = x0; x <= x1; x += 1) {
          state.masks[row + x] = label;
        }
      }
    }
  }

  function commitBrushFill() {
    const pts = state.strokePoints;
    const label = state.strokeLabel;
    if (pts.length >= 3 && label) {
      const first = pts[0];
      const last = pts[pts.length - 1];
      paintSegment(last.x, last.y, first.x, first.y, state.brushSize, label);
      fillPolygon(pts, label);
      state.selection = new Set([label]);
      setInfo(`Filled ROI ${label}`);
    }
    state.strokePoints = [];
    state.lastPaint = null;
  }

  function pointerDown(evt) {
    overlayCanvas.setPointerCapture(evt.pointerId);
    state.isPointerDown = true;
    state.lastPointer = { x: evt.clientX, y: evt.clientY };
    const p = eventToImageCoords(evt);
    if (state.tool === "select") {
      const label = state.masks[p.y * state.width + p.x];
      if (label > 0) {
        if (evt.shiftKey) {
          if (state.selection.has(label)) state.selection.delete(label);
          else state.selection.add(label);
        } else {
          state.selection = new Set([label]);
        }
      } else if (!evt.shiftKey) {
        state.selection.clear();
      }
      redrawMask();
    } else if (state.tool === "brush" || state.tool === "erase") {
      saveHistory();
      state.strokeAffected.clear();
      state.strokePoints = [];
      state.lastPaint = p;
      if (state.tool === "brush") {
        if (state.newLabel || state.selection.size === 0) {
          state.strokeLabel = nextLabel();
        } else {
          state.strokeLabel = Math.min(...state.selection);
        }
        state.strokePoints = [p];
      } else {
        state.strokeLabel = 0;
      }
      paintAt(p.x, p.y, state.brushSize, state.strokeLabel);
      redrawMask();
    } else if (state.tool === "split") {
      state.splitPoints = [p];
      drawSplitLine();
    }
  }

  function pointerMove(evt) {
    if (!state.isPointerDown) return;
    const p = eventToImageCoords(evt);
    if (state.tool === "brush" || state.tool === "erase") {
      const prev = state.lastPaint || p;
      paintSegment(prev.x, prev.y, p.x, p.y, state.brushSize, state.strokeLabel);
      state.lastPaint = p;
      if (state.tool === "brush") {
        const last = state.strokePoints[state.strokePoints.length - 1];
        if (!last || last.x !== p.x || last.y !== p.y) {
          state.strokePoints.push(p);
        }
      }
      redrawMask();
    } else if (state.tool === "split") {
      state.splitPoints.push(p);
      drawSplitLine();
    } else if (state.tool === "pan") {
      const dx = evt.clientX - state.lastPointer.x;
      const dy = evt.clientY - state.lastPointer.y;
      state.offsetX += dx;
      state.offsetY += dy;
      state.lastPointer = { x: evt.clientX, y: evt.clientY };
      applyTransform();
    }
  }

  function pointerUp(evt) {
    if (!state.isPointerDown) return;
    state.isPointerDown = false;
    if (state.tool === "split" && state.splitPoints.length > 1) {
      commitSplit();
    } else if (state.tool === "brush") {
      commitBrushFill();
      redrawMask();
    }
    state.strokeAffected.clear();
    state.strokeLabel = null;
    clearOverlay();
  }

  function clearOverlay() {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, state.width, state.height);
  }

  function drawSplitLine() {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 1;
    ctx.beginPath();
    state.splitPoints.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt.x + 0.5, pt.y + 0.5);
      else ctx.lineTo(pt.x + 0.5, pt.y + 0.5);
    });
    ctx.stroke();
  }

  function commitSplit() {
    if (state.selection.size !== 1) {
      setInfo("Select exactly one ROI before splitting");
      clearOverlay();
      state.splitPoints = [];
      return;
    }
    const target = Array.from(state.selection)[0];
    saveHistory();
    // Rasterize the split line and remove those pixels, then flood-fill
    // components remaining and relabel one side.
    const cutMask = new Uint8Array(state.width * state.height);
    for (let i = 1; i < state.splitPoints.length; i += 1) {
      const a = state.splitPoints[i - 1];
      const b = state.splitPoints[i];
      rasterizeLine(a.x, a.y, b.x, b.y, cutMask, 2);
    }
    for (let i = 0; i < state.masks.length; i += 1) {
      if (state.masks[i] === target && cutMask[i]) {
        state.masks[i] = 0;
      }
    }
    const components = floodFillLabel(target);
    if (components.length <= 1) {
      setInfo("Split did not produce separate regions");
      clearOverlay();
      state.splitPoints = [];
      redrawMask();
      return;
    }
    for (let ci = 1; ci < components.length; ci += 1) {
      const newId = nextLabel();
      for (const idx of components[ci]) {
        state.masks[idx] = newId;
      }
    }
    clearOverlay();
    state.splitPoints = [];
    redrawMask();
    setInfo(`Split ROI ${target} into ${components.length} parts`);
  }

  function rasterizeLine(x0, y0, x1, y1, buffer, thickness) {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;
    let x = x0;
    let y = y0;
    while (true) {
      for (let ty = -thickness; ty <= thickness; ty += 1) {
        for (let tx = -thickness; tx <= thickness; tx += 1) {
          const px = x + tx;
          const py = y + ty;
          if (px >= 0 && py >= 0 && px < state.width && py < state.height) {
            buffer[py * state.width + px] = 1;
          }
        }
      }
      if (x === x1 && y === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) { err -= dy; x += sx; }
      if (e2 < dx) { err += dx; y += sy; }
    }
  }

  function floodFillLabel(label) {
    const visited = new Uint8Array(state.masks.length);
    const components = [];
    for (let i = 0; i < state.masks.length; i += 1) {
      if (state.masks[i] !== label || visited[i]) continue;
      const stack = [i];
      const collected = [];
      while (stack.length) {
        const idx = stack.pop();
        if (visited[idx]) continue;
        visited[idx] = 1;
        if (state.masks[idx] !== label) continue;
        collected.push(idx);
        const x = idx % state.width;
        const y = Math.floor(idx / state.width);
        if (x > 0) stack.push(idx - 1);
        if (x < state.width - 1) stack.push(idx + 1);
        if (y > 0) stack.push(idx - state.width);
        if (y < state.height - 1) stack.push(idx + state.width);
      }
      if (collected.length) components.push(collected);
    }
    return components;
  }

  function wheelZoom(evt) {
    evt.preventDefault();
    const rect = overlayCanvas.getBoundingClientRect();
    const cx = evt.clientX - rect.left;
    const cy = evt.clientY - rect.top;
    const factor = evt.deltaY < 0 ? 1.1 : 0.9;
    const newScale = Math.max(0.2, Math.min(state.scale * factor, 20));
    state.offsetX -= cx * (newScale - state.scale) / state.scale;
    state.offsetY -= cy * (newScale - state.scale) / state.scale;
    state.scale = newScale;
    applyTransform();
  }

  function setTool(name) {
    state.tool = name;
    parentElement
      .querySelectorAll('.mask-editor__tool')
      .forEach((btn) => btn.classList.toggle('is-active', btn.dataset.tool === name));
    if (name === "brush") {
      setInfo("Draw an outline; release to fill the ROI");
    } else {
      setInfo(`Tool: ${name}`);
    }
  }

  parentElement.querySelectorAll('.mask-editor__tool').forEach((btn) => {
    btn.onclick = () => setTool(btn.dataset.tool);
  });
  setTool("select");

  brushEl.oninput = () => {
    state.brushSize = parseInt(brushEl.value, 10);
    brushValEl.textContent = state.brushSize;
  };
  alphaEl.oninput = () => {
    state.alpha = parseInt(alphaEl.value, 10) / 100;
    alphaValEl.textContent = alphaEl.value;
    redrawMask();
  };
  newLabelEl.onchange = () => {
    state.newLabel = newLabelEl.checked;
  };
  if (overlayToggleEl) {
    overlayToggleEl.onclick = toggleOverlay;
  }
  applyOverlayVisibility();

  parentElement.querySelector('[data-role="undo"]').onclick = undo;
  parentElement.querySelector('[data-role="redo"]').onclick = redo;
  parentElement.querySelector('[data-role="delete"]').onclick = deleteSelection;
  parentElement.querySelector('[data-role="merge"]').onclick = mergeSelection;
  parentElement.querySelector('[data-role="commit"]').onclick = () => {
    const payload = {
      masks_b64: encodeMaskB64(),
      width: state.width,
      height: state.height,
      max_label: state.maxLabel,
      timestamp: Date.now(),
    };
    setTriggerValue("save", payload);
    setInfo("Saved");
  };

  function handleKeyDown(evt) {
    if (evt.target.tagName === 'INPUT') return;
    if (evt.key === 's' || evt.key === 'S') setTool('select');
    else if (evt.key === 'b' || evt.key === 'B') setTool('brush');
    else if (evt.key === 'e' || evt.key === 'E') setTool('erase');
    else if (evt.key === 'x' || evt.key === 'X') setTool('split');
    else if (evt.key === 'o' || evt.key === 'O') toggleOverlay();
    else if (evt.key === ' ') setTool('pan');
    else if (evt.key === 'Delete' || evt.key === 'Backspace') deleteSelection();
    else if ((evt.ctrlKey || evt.metaKey) && evt.key === 'z') undo();
    else if ((evt.ctrlKey || evt.metaKey) && evt.key === 'y') redo();
  }

  const wheelOptions = { passive: false };
  overlayCanvas.addEventListener('pointerdown', pointerDown);
  overlayCanvas.addEventListener('pointermove', pointerMove);
  overlayCanvas.addEventListener('pointerup', pointerUp);
  overlayCanvas.addEventListener('pointercancel', pointerUp);
  overlayCanvas.addEventListener('wheel', wheelZoom, wheelOptions);
  parentElement.addEventListener('keydown', handleKeyDown);

  return () => {
    image.onload = null;
    overlayCanvas.removeEventListener('pointerdown', pointerDown);
    overlayCanvas.removeEventListener('pointermove', pointerMove);
    overlayCanvas.removeEventListener('pointerup', pointerUp);
    overlayCanvas.removeEventListener('pointercancel', pointerUp);
    overlayCanvas.removeEventListener('wheel', wheelZoom, wheelOptions);
    parentElement.removeEventListener('keydown', handleKeyDown);
  };
}
