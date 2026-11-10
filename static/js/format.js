// Number formatting that matches Python, so tables and CSV files read the same as the desktop app.

// Python's format(x, '.<d>f'): exact decimal rounding, ties to even (toFixed sends ties upwards).
export function pyFixed(x, d) {
  if (!Number.isFinite(x)) return String(x);
  const neg = x < 0 || Object.is(x, -0);
  const a = Math.abs(x);
  let out = a.toFixed(d);
  if (a < 1e21) {
    const exact = a.toFixed(Math.min(100, d + 70));
    const dot = exact.indexOf('.');
    const tail = exact.slice(dot + 1 + d);
    if (/^50*$/.test(tail)) {
      const truncated = d > 0 ? exact.slice(0, dot + 1 + d) : exact.slice(0, dot);
      const last = truncated.charCodeAt(truncated.length - 1) - 48;
      if (last % 2 === 0) out = truncated;
    }
  }
  return (neg ? '-' : '') + out;
}

// Python's repr(float): shortest round-trip digits, always with ".0" or an exponent.
export function pyRepr(x) {
  if (Number.isNaN(x)) return 'nan';
  if (x === Infinity) return 'inf';
  if (x === -Infinity) return '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const sign = x < 0 ? '-' : '';
  const [mant, expStr] = Math.abs(x).toExponential().split('e');
  const exp = parseInt(expStr, 10);
  const digits = mant.replace('.', '');
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? digits[0] + '.' + digits.slice(1) : digits;
    const e = Math.abs(exp);
    return sign + m + 'e' + (exp < 0 ? '-' : '+') + (e < 10 ? '0' + e : e);
  }
  if (exp < 0) return sign + '0.' + '0'.repeat(-exp - 1) + digits;
  if (digits.length <= exp + 1) return sign + digits + '0'.repeat(exp + 1 - digits.length) + '.0';
  return sign + digits.slice(0, exp + 1) + '.' + digits.slice(exp + 1);
}

// _format_readout_text, plus the micron equivalents.
export function readoutText(r, ppm) {
  const px = `T:${pyFixed(r.top, 1)}px R:${pyFixed(r.right, 1)}px ` +
    `B:${pyFixed(r.bottom, 1)}px L:${pyFixed(r.left, 1)}px | A:${pyFixed(r.area, 0)}px²`;
  if (!(ppm > 0)) return px;
  const um = `T:${pyFixed(r.top / ppm, 1)}µm R:${pyFixed(r.right / ppm, 1)}µm ` +
    `B:${pyFixed(r.bottom / ppm, 1)}µm L:${pyFixed(r.left / ppm, 1)}µm | A:${pyFixed(r.area / (ppm * ppm), 0)}µm²`;
  return px + '\n' + um;
}

export const COLUMNS = [
  { key: 'file', label: 'File', tip: 'Name of the image file.' },
  { key: 'num', label: 'Neurite Count', tip: 'Total number of neurite segments detected.' },
  { key: 'tot_px', label: 'Total Neurite Length (px)', tip: 'Sum of the lengths of all detected neurites in pixels.' },
  { key: 'tot_um', label: 'Total Neurite Length (µm)', tip: 'Sum of the lengths of all detected neurites in microns.', fixed: 2 },
  { key: 'max_px', label: 'Max Neurite Length (px)', tip: 'Length of the longest detected neurite segment in pixels.' },
  { key: 'max_um', label: 'Max Neurite Length (µm)', tip: 'Length of the longest detected neurite segment in microns.', fixed: 2 },
  { key: 'avg_px', label: 'Average Neurite Length (px)', tip: 'Mean length of detected neurites in pixels.', fixed: 1 },
  { key: 'avg_um', label: 'Average Neurite Length (µm)', tip: 'Mean length of detected neurites in microns.', fixed: 2 },
  { key: 'med_px', label: 'Median Neurite Length (px)', tip: 'Median length of detected neurites in pixels.', fixed: 1 },
  { key: 'med_um', label: 'Median Neurite Length (µm)', tip: 'Median length of detected neurites in microns.', fixed: 2 },
  { key: 'roi_area_px2', label: 'ROI Area (px²)', tip: 'Area of the region of interest in square pixels.' },
  { key: 'roi_area_um2', label: 'ROI Area (µm²)', tip: 'Area of the region of interest in square microns.', fixed: 0 },
];

// Keys that are Python ints in the desktop app; everything else numeric is a float.
const INT_KEYS = new Set(['num', 'tot_px', 'max_px', 'roi_area_px2']);

export function cellText(col, row) {
  const v = row[col.key];
  if (col.key === 'file') return v;
  return col.fixed === undefined ? String(v) : pyFixed(v, col.fixed);
}

// The value as Python's csv module would write it.
export function csvValue(key, row) {
  const v = row[key];
  if (key === 'file') return v;
  if (INT_KEYS.has(key)) return String(v);
  // sum([]) is the int 0 in Python, so an image without neurites writes "0" here.
  if (key === 'tot_um' && row.num === 0) return '0';
  return pyRepr(v);
}
