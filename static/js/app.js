// Neurite Outgrowth Analyser - web edition: user interface.
import { Viewer } from './viewer.js';
import { COLUMNS, cellText, readoutText } from './format.js';
import { buildZip, download, drawScaleBar } from './exporter.js';

const Pipeline = globalThis.NeuritePipeline; // pure helpers (statistics, ROI readout)
const STORE_KEY = 'neurite-outgrowth-web-v1';
const DEMO = ['demo_image_0h_post-scratch.tif', 'demo_image_120h_post-scratch.tif'];

const DEFAULTS = {
  topHat: 15,
  otsuOffset: 0,
  manualThresh: 100,
  minArea: 150,
  minLength: 50,
  ppm: 0.287,
  mode: 'Color',
  color: '#ff0000',
  thickness: 3,
  sbOn: false,
  sbLength: 1000,
  sbPosition: 'Bottom Right',
  sbLabel: '1000 µm',
  sbColor: '#ffffff',
};

const SLIDERS = [
  { key: 'topHat', label: 'Top-hat size', min: 3, max: 51, step: 2, hint: 'Structuring element for top-hat (odd).' },
  { key: 'otsuOffset', label: 'Otsu offset', min: -50, max: 50, step: 1, hint: 'Adjust Otsu threshold.' },
  { key: 'manualThresh', label: 'Manual thresh (0 = auto)', min: 0, max: 255, step: 1, hint: '0 uses Otsu + offset.' },
  { key: 'minArea', label: 'Min area (px)', min: 0, max: 5000, step: 10, hint: 'Discard small components.' },
  { key: 'minLength', label: 'Min length (px)', min: 0, max: 200, step: 1, hint: 'Discard short skeletons.' },
];
const DETECTION_KEYS = ['topHat', 'otsuOffset', 'manualThresh', 'minArea', 'minLength'];

const $ = (id) => document.getElementById(id);

const state = {
  params: loadParams(),
  ref: null,
  images: [],
  sel: null, // item id
  tab: 'roi',
  preview: null, // { key, canvas, info }
  busy: false,
};

/* ---------- worker RPC ---------- */

const worker = new Worker(new URL('./worker.js', import.meta.url));
const pending = new Map();
let seq = 0;
worker.onmessage = (ev) => {
  const { seq: s, ok, result, error } = ev.data;
  const p = pending.get(s);
  if (!p) return;
  pending.delete(s);
  if (ok) p.resolve(result);
  else p.reject(new Error(error));
};
worker.onerror = (ev) => {
  setStatus('Image engine failed to load: ' + (ev.message || 'unknown error'), 'error');
  for (const p of pending.values()) p.reject(new Error('Worker error'));
  pending.clear();
};
function call(op, args, transfer) {
  return new Promise((resolve, reject) => {
    const s = ++seq;
    pending.set(s, { resolve, reject });
    worker.postMessage({ seq: s, op, args }, transfer || []);
  });
}

/* ---------- parameters ---------- */

function loadParams() {
  const p = Object.assign({}, DEFAULTS);
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    for (const k of Object.keys(DEFAULTS)) {
      if (typeof saved[k] === typeof DEFAULTS[k]) p[k] = saved[k];
    }
  } catch (e) { /* storage unavailable or corrupt: use defaults */ }
  return p;
}

function saveParams() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(state.params));
  } catch (e) { /* ignore */ }
}

function detectionParams() {
  const p = state.params;
  return {
    topHat: p.topHat,
    otsuOffset: p.otsuOffset,
    manualThresh: p.manualThresh,
    minArea: p.minArea,
    minLength: p.minLength,
  };
}

function overlayParams() {
  const p = state.params;
  const hex = /^#[0-9a-f]{6}$/i.test(p.color) ? p.color : DEFAULTS.color;
  return {
    mode: p.mode,
    thickness: p.thickness,
    color: [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)),
  };
}

function scaleBar() {
  const p = state.params;
  return { on: p.sbOn, lengthUm: p.sbLength, position: p.sbPosition, label: p.sbLabel, color: p.sbColor };
}

function clampNum(v, min, max, step) {
  if (!Number.isFinite(v)) return null;
  v = Math.max(min, Math.min(max, v));
  return min + Math.round((v - min) / step) * step;
}

function buildParamControls() {
  const host = $('sliders');
  for (const s of SLIDERS) {
    const row = document.createElement('div');
    row.className = 'param';
    row.innerHTML =
      `<label for="r-${s.key}">${s.label}</label>` +
      `<div class="pair"><input type="range" id="r-${s.key}" min="${s.min}" max="${s.max}" step="${s.step}">` +
      `<input type="number" id="n-${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" aria-label="${s.label}"></div>` +
      `<small>${s.hint}</small>`;
    host.appendChild(row);
    const range = $('r-' + s.key);
    const num = $('n-' + s.key);
    const apply = (raw, commit) => {
      const v = clampNum(Number(raw), s.min, s.max, s.step);
      if (v === null) return;
      range.value = v;
      if (commit || Number(num.value) !== v) num.value = v;
      setParam(s.key, v);
    };
    range.addEventListener('input', () => apply(range.value, true));
    num.addEventListener('change', () => apply(num.value, true));
  }

  const bind = (id, key, read, ev) => {
    $(id).addEventListener(ev || 'change', () => {
      const v = read($(id));
      if (v !== null) setParam(key, v);
    });
  };
  bind('ppm', 'ppm', (el) => (Number(el.value) > 0 ? Number(el.value) : null));
  bind('mode', 'mode', (el) => el.value);
  bind('color', 'color', (el) => el.value, 'input');
  bind('thickness', 'thickness', (el) => clampNum(Number(el.value), 1, 10, 1), 'input');
  bind('sbOn', 'sbOn', (el) => el.checked);
  bind('sbPosition', 'sbPosition', (el) => el.value);
  bind('sbLabel', 'sbLabel', (el) => el.value, 'input');
  bind('sbColor', 'sbColor', (el) => el.value, 'input');
  $('sbLength').addEventListener('change', () => {
    const v = Number($('sbLength').value);
    if (!(v > 0)) return;
    // Keep the label in step with the length unless the user wrote their own label.
    if (state.params.sbLabel === `${state.params.sbLength} µm`) {
      state.params.sbLabel = `${v} µm`;
      $('sbLabel').value = state.params.sbLabel;
    }
    setParam('sbLength', v);
  });
}

function showParams() {
  const p = state.params;
  for (const s of SLIDERS) {
    $('r-' + s.key).value = p[s.key];
    $('n-' + s.key).value = p[s.key];
  }
  $('ppm').value = p.ppm;
  $('mode').value = p.mode;
  $('color').value = p.color;
  $('thickness').value = p.thickness;
  $('thicknessVal').textContent = p.thickness;
  $('sbOn').checked = p.sbOn;
  $('sbLength').value = p.sbLength;
  $('sbPosition').value = p.sbPosition;
  $('sbLabel').value = p.sbLabel;
  $('sbColor').value = p.sbColor;
}

function setParam(key, value) {
  if (state.params[key] === value) return;
  state.params[key] = value;
  saveParams();
  $('thicknessVal').textContent = state.params.thickness;
  if (DETECTION_KEYS.includes(key) || key === 'ppm') {
    markStale();
    if (key !== 'ppm') schedulePreview();
    updateReadout();
  } else if (key === 'mode' || key === 'color' || key === 'thickness') {
    scheduleRepaint();
  } else {
    refreshView();
  }
}

/* ---------- loading images ---------- */

let nextId = 1;

function isTiff(file) {
  return /\.tiff?$/i.test(file.name) || file.type === 'image/tiff';
}

async function loadFile(file) {
  const id = 'img' + nextId++;
  let width;
  let height;
  let imageData;
  if (isTiff(file)) {
    const buffer = await file.arrayBuffer();
    const r = await call('decodeTiff', { id, buffer }, [buffer]);
    width = r.width;
    height = r.height;
    imageData = new ImageData(r.rgba, width, height);
  } else {
    const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    width = bmp.width;
    height = bmp.height;
    const tmp = document.createElement('canvas');
    tmp.width = width;
    tmp.height = height;
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    tctx.drawImage(bmp, 0, 0);
    bmp.close();
    imageData = tctx.getImageData(0, 0, width, height);
    const copy = new Uint8Array(imageData.data); // the worker keeps this copy
    await call('store', { id, width, height, rgba: copy }, [copy.buffer]);
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').putImageData(imageData, 0, 0);
  return { id, name: file.name, width, height, canvas, roi: null, custom: false, result: null, overlay: null, stale: false };
}

function fullRoi(it) {
  return [[0, 0], [it.width, 0], [it.width, it.height], [0, it.height]];
}

function refRoiFor(it) {
  if (!state.ref) return fullRoi(it);
  return state.ref.roi.map((p) => [Math.min(it.width, p[0]), Math.min(it.height, p[1])]);
}

function propagateRefRoi(force) {
  for (const it of state.images) {
    if (force || !it.custom) {
      const roi = refRoiFor(it);
      if (String(roi) !== String(it.roi)) {
        it.roi = roi;
        if (it.result) it.stale = true;
      }
      it.custom = false;
    }
  }
}

async function setReference(file) {
  await busy(`Loading reference ${file.name}…`, async () => {
    const it = await loadFile(file);
    const r = await call('autoRoi', { id: it.id });
    it.roi = r.roi;
    it.autoFound = r.found;
    if (state.ref) await call('remove', { id: state.ref.id });
    state.ref = it;
    propagateRefRoi(false);
    state.sel = it.id;
    state.tab = 'roi';
    setStatus(r.found
      ? `Reference loaded; scratch gap detected at rows ${r.roi[0][1]}–${r.roi[2][1]}.`
      : 'Reference loaded; no scratch gap detected, ROI set to the full image.');
  });
}

async function addImages(files) {
  if (!files.length) return;
  await busy('Loading images…', async () => {
    let n = 0;
    const failed = [];
    for (const f of files) {
      setStatus(`Loading ${f.name} (${++n}/${files.length})…`, 'busy');
      try {
        const it = await loadFile(f);
        it.roi = refRoiFor(it);
        state.images.push(it);
        if (!state.sel) state.sel = it.id;
      } catch (err) {
        failed.push(`${f.name}: ${err.message}`);
      }
    }
    if (failed.length) setStatus('Could not load ' + failed.join('; '), 'error');
    else setStatus(`${state.images.length} image${state.images.length === 1 ? '' : 's'} ready.`);
  });
}

async function removeImage(id) {
  const i = state.images.findIndex((it) => it.id === id);
  if (i < 0) return;
  state.images.splice(i, 1);
  await call('remove', { id });
  if (state.sel === id) state.sel = state.ref ? state.ref.id : state.images[0] ? state.images[0].id : null;
  renderAll();
}

async function loadDemo() {
  await busy('Downloading demo images…', async () => {
    const files = [];
    for (const name of DEMO) {
      const resp = await fetch(name);
      if (!resp.ok) throw new Error(`Could not fetch ${name} (HTTP ${resp.status})`);
      files.push(new File([await resp.blob()], name, { type: 'image/tiff' }));
    }
    for (const it of state.images.splice(0)) await call('remove', { id: it.id });
    state.busy = false;
    await setReference(files[0]);
    await addImages(files);
  });
}

/* ---------- analysis ---------- */

function toCanvas(rgba, w, h, canvas) {
  const c = canvas || document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  return c;
}

async function runAnalysis() {
  if (!state.images.length) {
    setStatus('No images to analyse.', 'error');
    return;
  }
  await busy('Analysing…', async () => {
    const params = detectionParams();
    const ppm = state.params.ppm;
    let n = 0;
    for (const it of state.images) {
      setStatus(`Analysing ${it.name} (${++n}/${state.images.length})…`, 'busy');
      const roi = it.roi.map((p) => p.slice());
      const r = await call('analyse', { id: it.id, roi, params, overlay: overlayParams() });
      it.result = {
        row: Pipeline.summarise(it.name, r.lengths, roi, ppm),
        thr: r.thr,
        otsu: r.otsu,
        components: r.components,
        kept: r.kept,
      };
      it.overlay = toCanvas(r.overlay, it.width, it.height, it.overlay);
      it.stale = false;
      renderResults();
    }
    if (state.sel === (state.ref && state.ref.id) && !state.images.includes(state.ref)) {
      state.sel = state.images[0].id;
    }
    state.tab = 'overlay';
    setStatus(`Analysis complete: ${state.images.length} image${state.images.length === 1 ? '' : 's'}.`);
  });
}

function markStale() {
  let any = false;
  for (const it of state.images) {
    if (it.result && !it.stale) {
      it.stale = true;
      any = true;
    }
  }
  if (any) renderResults();
}

let repaintTimer = 0;
let repainting = false;
let repaintAgain = false;
function scheduleRepaint() {
  clearTimeout(repaintTimer);
  repaintTimer = setTimeout(repaintOverlays, 120);
}
async function repaintOverlays() {
  if (repainting) {
    repaintAgain = true;
    return;
  }
  repainting = true;
  try {
    do {
      repaintAgain = false;
      const ov = overlayParams();
      // Selected image first so the viewer updates quickly.
      const order = state.images.filter((it) => it.overlay).sort((a, b) => (b.id === state.sel) - (a.id === state.sel));
      for (const it of order) {
        if (repaintAgain) break;
        const r = await call('overlay', { id: it.id, overlay: ov });
        if (r.overlay) toCanvas(r.overlay, it.width, it.height, it.overlay);
        if (it.id === state.sel) refreshView();
      }
    } while (repaintAgain);
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    repainting = false;
  }
}

async function exportZip() {
  const done = state.images.filter((it) => it.result);
  if (!done.length) {
    setStatus('No results to export.', 'error');
    return;
  }
  await busy('Building ZIP…', async () => {
    const entries = done.map((it) => ({ row: it.result.row, overlay: it.overlay }));
    const { zip, xlsxError } = await buildZip(entries, scaleBar(), state.params.ppm);
    download(zip, 'neurite_results.zip', 'application/zip');
    setStatus(xlsxError
      ? `Exported CSV and overlays (Excel export failed: ${xlsxError.message}).`
      : `Exported CSV, XLSX and ${done.length} overlay${done.length === 1 ? '' : 's'} as neurite_results.zip.`);
  });
}

/* ---------- threshold preview ---------- */

function previewKey(it) {
  return it.id + '|' + it.roi.join(';') + '|' + JSON.stringify(detectionParams());
}

let previewTimer = 0;
let previewRunning = false;
let previewAgain = false;
function schedulePreview() {
  if (state.tab !== 'preview') return;
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, 150);
}
async function runPreview() {
  if (previewRunning) {
    previewAgain = true;
    return;
  }
  previewRunning = true;
  try {
    do {
      previewAgain = false;
      const it = selected();
      if (!it || state.tab !== 'preview') break;
      const key = previewKey(it);
      if (state.preview && state.preview.key === key) break;
      $('viewInfo').textContent = 'Computing threshold preview…';
      const r = await call('preview', { id: it.id, roi: it.roi, params: detectionParams() });
      state.preview = {
        key,
        canvas: toCanvas(r.rgba, it.width, it.height, state.preview && state.preview.canvas),
        info: r,
      };
      refreshView();
    } while (previewAgain);
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    previewRunning = false;
  }
}

/* ---------- rendering ---------- */

let viewer;
const composed = document.createElement('canvas');

function selected() {
  if (state.ref && state.ref.id === state.sel) return state.ref;
  return state.images.find((it) => it.id === state.sel) || null;
}

function isRef(it) {
  return !!state.ref && it === state.ref;
}

function setStatus(text, kind) {
  const el = $('status');
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

async function busy(text, fn) {
  if (state.busy) return;
  state.busy = true;
  setStatus(text, 'busy');
  updateButtons();
  try {
    await fn();
  } catch (err) {
    setStatus(err && err.message ? err.message : String(err), 'error');
  } finally {
    state.busy = false;
    renderAll();
  }
}

function updateButtons() {
  const hasResults = state.images.some((it) => it.result);
  const stale = state.images.some((it) => it.stale || !it.result);
  $('btnRun').disabled = state.busy || !state.images.length;
  $('btnExport').disabled = state.busy || !hasResults || stale;
  $('btnExport').title = hasResults && stale ? 'Results are out of date: run the analysis again before exporting.' : '';
  $('btnDemo').disabled = state.busy;
  const it = selected();
  $('btnAuto').disabled = state.busy || !it || !isRef(it);
  $('btnResetRoi').disabled = state.busy || !it || isRef(it);
  $('btnApplyAll').disabled = state.busy || !state.ref || !state.images.length;
}

function renderLists() {
  $('refName').textContent = state.ref ? `${state.ref.name} (${state.ref.width}×${state.ref.height})` : 'None loaded';
  $('clearRef').hidden = !state.ref;
  const list = $('imgList');
  list.textContent = '';
  for (const it of state.images) {
    const li = document.createElement('li');
    if (it.id === state.sel) li.className = 'sel';
    const name = document.createElement('button');
    name.type = 'button';
    name.className = 'link';
    name.textContent = it.name;
    name.title = `${it.width}×${it.height}` + (it.custom ? ', ROI adjusted individually' : '');
    name.addEventListener('click', () => select(it.id));
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = it.custom ? 'own ROI' : '';
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'icon';
    rm.textContent = '×';
    rm.title = 'Remove ' + it.name;
    rm.setAttribute('aria-label', 'Remove ' + it.name);
    rm.addEventListener('click', () => removeImage(it.id));
    li.append(name, tag, rm);
    list.appendChild(li);
  }
  $('clearImgs').hidden = !state.images.length;

  const sel = $('itemSel');
  sel.textContent = '';
  const add = (it, label) => {
    const o = document.createElement('option');
    o.value = it.id;
    o.textContent = label;
    sel.appendChild(o);
  };
  if (state.ref) add(state.ref, 'Reference: ' + state.ref.name);
  state.images.forEach((it, i) => add(it, `Image ${i + 1}: ${it.name}`));
  if (state.sel) sel.value = state.sel;
  sel.disabled = !sel.options.length;
}

function renderResults() {
  const body = $('resultsBody');
  body.textContent = '';
  const rows = state.images.filter((it) => it.result);
  for (const it of rows) {
    const tr = document.createElement('tr');
    if (it.stale) tr.className = 'stale';
    if (it.id === state.sel) tr.classList.add('sel');
    tr.addEventListener('click', () => {
      state.tab = 'overlay';
      select(it.id);
    });
    for (const col of COLUMNS) {
      const td = document.createElement('td');
      td.textContent = cellText(col, it.result.row);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  $('resultsEmpty').hidden = rows.length > 0;
  const stale = rows.some((it) => it.stale);
  $('staleNote').hidden = !stale;
  updateButtons();
}

function updateReadout() {
  const it = selected();
  $('readout').textContent = it && it.roi ? readoutText(Pipeline.roiReadout(it.roi), state.params.ppm) : '';
}

function refreshView() {
  const it = selected();
  document.querySelectorAll('.tab').forEach((b) => {
    const on = b.dataset.tab === state.tab;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  const info = $('viewInfo');
  if (!it) {
    viewer.setRoi(null, false);
    viewer.setSource(null, 0, 0, 'Load a reference image and images to analyse, or try the demo images.');
    info.textContent = '';
    updateReadout();
    return;
  }
  if (state.tab === 'roi') {
    viewer.setRoi(it.roi, true);
    viewer.setSource(it.canvas, it.width, it.height);
    info.textContent = isRef(it)
      ? 'Drag a corner or the whole shape. The reference ROI is copied to every image that has not been adjusted individually.'
      : (it.custom ? 'This image has its own ROI.' : 'Using the reference ROI.') + ' Drag a corner or the whole shape to adjust it for this image only.';
  } else if (state.tab === 'preview') {
    viewer.setRoi(it.roi, false);
    const pv = state.preview && state.preview.key === previewKey(it) ? state.preview : null;
    if (pv) {
      viewer.setSource(pv.canvas, it.width, it.height);
      const r = pv.info;
      const thr = r.otsu === null
        ? `Threshold ${r.thr} (manual)`
        : `Threshold ${r.thr} (Otsu ${r.otsu} ${state.params.otsuOffset < 0 ? '−' : '+'} ${Math.abs(state.params.otsuOffset)})`;
      info.textContent = `${thr}. ${r.kept} of ${r.components} components kept by min area. White: kept; dark red: removed as too small.`;
    } else {
      viewer.setSource(it.canvas, it.width, it.height, 'Computing threshold preview…');
      runPreview();
    }
  } else {
    viewer.setRoi(null, false);
    if (it.overlay && it.result) {
      let src = it.overlay;
      if (state.params.sbOn) {
        composed.width = it.width;
        composed.height = it.height;
        const ctx = composed.getContext('2d');
        ctx.drawImage(it.overlay, 0, 0);
        drawScaleBar(ctx, it.width, it.height, scaleBar(), state.params.ppm);
        src = composed;
      }
      viewer.setSource(src, it.width, it.height);
      const row = it.result.row;
      info.textContent = `${row.num} neurite segment${row.num === 1 ? '' : 's'}, threshold ${it.result.thr}` +
        (it.stale ? '. Out of date: parameters or ROI changed since this was computed.' : '.');
    } else {
      viewer.setSource(it.canvas, it.width, it.height,
        isRef(it) ? 'The reference image only defines the ROI. Add it to the images to analyse it.' : 'Run the analysis to see the overlay.');
      info.textContent = '';
    }
  }
  updateReadout();
}

function renderAll() {
  renderLists();
  renderResults();
  refreshView();
  updateButtons();
}

function select(id) {
  state.sel = id;
  renderAll();
}

/* ---------- wiring ---------- */

function onRoiChange(roi, done) {
  const it = selected();
  if (!it) return;
  it.roi = roi.map((p) => p.slice());
  if (isRef(it)) {
    propagateRefRoi(false);
  } else {
    it.custom = true;
    if (it.result) it.stale = true;
  }
  updateReadout();
  if (done) {
    renderLists();
    renderResults();
  }
}

function dropZone(id, input, handler) {
  const zone = $(id);
  const pick = $(input);
  zone.addEventListener('click', (e) => {
    if (e.target.closest('button') && e.target.closest('button') !== zone) return;
    pick.click();
  });
  zone.addEventListener('keydown', (e) => {
    if (e.target === zone && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      pick.click();
    }
  });
  pick.addEventListener('change', () => {
    const files = Array.from(pick.files);
    pick.value = '';
    handler(files);
  });
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => {
    e.preventDefault();
    zone.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (e) => {
    e.preventDefault();
    zone.classList.remove('over');
  }));
  zone.addEventListener('drop', (e) => handler(Array.from(e.dataTransfer.files)));
}

function init() {
  const head = $('resultsHead');
  for (const col of COLUMNS) {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.title = col.tip;
    th.scope = 'col';
    head.appendChild(th);
  }

  buildParamControls();
  showParams();

  viewer = new Viewer($('view'));
  viewer.onRoiChange = onRoiChange;

  dropZone('dropRef', 'fileRef', (files) => files[0] && setReference(files[0]));
  dropZone('dropImgs', 'fileImgs', addImages);
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());

  $('btnDemo').addEventListener('click', loadDemo);
  $('btnRun').addEventListener('click', runAnalysis);
  $('btnExport').addEventListener('click', exportZip);
  $('btnDefaults').addEventListener('click', () => {
    state.params = Object.assign({}, DEFAULTS);
    saveParams();
    showParams();
    markStale();
    scheduleRepaint();
    renderAll();
  });
  $('clearRef').addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!state.ref) return;
    const id = state.ref.id;
    state.ref = null;
    await call('remove', { id });
    if (state.sel === id) state.sel = state.images[0] ? state.images[0].id : null;
    renderAll();
  });
  $('clearImgs').addEventListener('click', async (e) => {
    e.stopPropagation();
    for (const it of state.images.splice(0)) await call('remove', { id: it.id });
    state.sel = state.ref ? state.ref.id : null;
    renderAll();
  });

  $('itemSel').addEventListener('change', () => select($('itemSel').value));
  document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
    state.tab = b.dataset.tab;
    refreshView();
  }));
  $('zoomIn').addEventListener('click', () => viewer.zoomBy(1.25));
  $('zoomOut').addEventListener('click', () => viewer.zoomBy(0.8));
  $('zoomFit').addEventListener('click', () => viewer.fit());

  $('btnAuto').addEventListener('click', () => busy('Detecting scratch gap…', async () => {
    const r = await call('autoRoi', { id: state.ref.id });
    state.ref.roi = r.roi;
    propagateRefRoi(false);
    setStatus(r.found ? 'Scratch gap detected.' : 'No scratch gap detected; ROI set to the full image.');
  }));
  $('btnResetRoi').addEventListener('click', () => {
    const it = selected();
    if (!it || isRef(it)) return;
    const roi = refRoiFor(it);
    if (String(roi) !== String(it.roi) && it.result) it.stale = true;
    it.roi = roi;
    it.custom = false;
    renderAll();
  });
  $('btnApplyAll').addEventListener('click', () => {
    propagateRefRoi(true);
    renderAll();
  });

  renderAll();
  setStatus('Loading image engine (OpenCV.js, about 10 MB, first visit only)…', 'busy');
  call('init').then(
    () => setStatus('Ready. Images are processed on this device and are never uploaded.'),
    (err) => setStatus('Image engine failed to start: ' + err.message, 'error')
  );
}

init();
