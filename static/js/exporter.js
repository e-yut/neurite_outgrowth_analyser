// CSV / XLSX / overlay PNG export, bundled into one ZIP. Uses the vendored fflate and SheetJS globals.
import { COLUMNS, csvValue } from './format.js';

const KEYS = COLUMNS.map((c) => c.key);

function csvQuote(s) {
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// Same layout as csv.DictWriter in the desktop app (comma separated, CRLF line endings).
export function buildCsv(rows) {
  const lines = [KEYS.join(',')];
  for (const r of rows) lines.push(KEYS.map((k) => csvQuote(csvValue(k, r))).join(','));
  return lines.join('\r\n') + '\r\n';
}

export function buildXlsx(rows) {
  const XLSX = globalThis.XLSX;
  const aoa = [KEYS].concat(rows.map((r) => KEYS.map((k) => r[k])));
  const ws = XLSX.utils.aoa_to_sheet(aoa);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
  return new Uint8Array(XLSX.write(wb, { type: 'array', bookType: 'xlsx' }));
}

// Scale bar with the geometry of export() in the desktop app. The label is drawn with a canvas
// font, so (unlike cv2.putText) it can show "µ".
export function drawScaleBar(ctx, w, h, sb, ppm) {
  if (!sb.on) return;
  const lengthPx = Math.trunc(sb.lengthUm * ppm);
  if (!(lengthPx > 0)) return;
  const m = 10;
  const bottom = sb.position.indexOf('Bottom') === 0;
  const right = sb.position.indexOf('Right') > 0;
  const x1 = right ? w - m - lengthPx : m;
  const y1 = bottom ? h - m : m + 10;
  ctx.save();
  ctx.fillStyle = sb.color;
  ctx.fillRect(x1, y1 - 2, lengthPx + 1, 5);
  ctx.font = '22px Arial, Helvetica, sans-serif';
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.fillText(sb.label, x1, bottom ? y1 - 10 : y1 + 25);
  ctx.restore();
}

function canvasToPng(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(async (blob) => {
      if (!blob) return reject(new Error('Could not encode overlay image'));
      resolve(new Uint8Array(await blob.arrayBuffer()));
    }, 'image/png');
  });
}

function overlayName(file, used) {
  const dot = file.lastIndexOf('.');
  const stem = dot > 0 ? file.slice(0, dot) : file;
  let name = `overlay_${stem}.png`;
  for (let i = 2; used.has(name.toLowerCase()); i++) name = `overlay_${stem}_${i}.png`;
  used.add(name.toLowerCase());
  return name;
}

// entries: [{ row, overlay: HTMLCanvasElement }], sb: scale-bar settings.
export async function buildZip(entries, sb, ppm) {
  const fflate = globalThis.fflate;
  const rows = entries.map((e) => e.row);
  const files = {
    'neurite_results.csv': [fflate.strToU8(buildCsv(rows)), { level: 6 }],
  };
  let xlsxError = null;
  try {
    files['neurite_results.xlsx'] = [buildXlsx(rows), { level: 0 }];
  } catch (err) {
    xlsxError = err; // as on the desktop: not fatal, the CSV and overlays are still saved
  }
  const used = new Set();
  for (const e of entries) {
    const c = document.createElement('canvas');
    c.width = e.overlay.width;
    c.height = e.overlay.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(e.overlay, 0, 0);
    drawScaleBar(ctx, c.width, c.height, sb, ppm);
    files[overlayName(e.row.file, used)] = [await canvasToPng(c), { level: 0 }];
  }
  return { zip: fflate.zipSync(files), xlsxError };
}

export function download(bytes, name, type) {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
