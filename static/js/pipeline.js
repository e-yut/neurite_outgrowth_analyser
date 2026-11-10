/*
 * Neurite Outgrowth Analyser - web edition
 * Image-processing pipeline, a line-by-line port of NeuriteAnalysis_v.6.7.1.py to OpenCV.js.
 *
 * Classic script (no imports) so it can be loaded with importScripts() in the worker.
 * Usage: const pipeline = NeuritePipeline.create(cv);
 *
 * Images are passed as { id, width, height, rgba } where rgba is a Uint8Array/Uint8ClampedArray.
 * ROIs are four [x, y] integer points: top-left, top-right, bottom-right, bottom-left.
 *
 * The quirks of the desktop script are reproduced on purpose (see README).
 */
(function (root) {
  'use strict';

  /* ---------- pure helpers (no OpenCV) ---------- */

  // numpy's pairwise summation for float64, so that means follow np.mean.
  function pairwiseSum(a, start, n) {
    if (n < 8) {
      let res = 0;
      for (let i = 0; i < n; i++) res += a[start + i];
      return res;
    }
    if (n <= 128) {
      const r = [0, 0, 0, 0, 0, 0, 0, 0];
      for (let j = 0; j < 8; j++) r[j] = a[start + j];
      let i = 8;
      for (; i < n - (n % 8); i += 8) {
        for (let j = 0; j < 8; j++) r[j] += a[start + i + j];
      }
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[start + i];
      return res;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2);
  }

  function mean(a) {
    return pairwiseSum(a, 0, a.length) / a.length;
  }

  function median(a) {
    const s = Array.from(a).sort((x, y) => x - y);
    const n = s.length;
    const mid = n >> 1;
    return n % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }

  // _compute_readout: side lengths and shoelace area of the quadrilateral.
  function roiReadout(pts) {
    const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
    let s = 0;
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      s += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
    }
    return {
      top: dist(pts[0], pts[1]),
      right: dist(pts[1], pts[2]),
      bottom: dist(pts[3], pts[2]),
      left: dist(pts[0], pts[3]),
      area: Math.abs(s) / 2,
    };
  }

  // The statistics block of run_analysis. lengthsPx is already filtered by min length.
  function summarise(file, lengthsPx, roi, ppm) {
    const lengthsUm = lengthsPx.map((lp) => lp / ppm);
    const has = lengthsPx.length > 0;
    let totPx = 0;
    let totUm = 0;
    let comp = 0;
    let maxPx = 0;
    let maxUm = 0;
    for (let i = 0; i < lengthsPx.length; i++) {
      totPx += lengthsPx[i];
      // Python's sum() of floats is Neumaier-compensated since 3.12; reproduce it bit for bit.
      const x = lengthsUm[i];
      const t = totUm + x;
      comp += Math.abs(totUm) >= Math.abs(x) ? (totUm - t) + x : (x - t) + totUm;
      totUm = t;
      if (lengthsPx[i] > maxPx) maxPx = lengthsPx[i];
      if (lengthsUm[i] > maxUm) maxUm = lengthsUm[i];
    }
    if (Number.isFinite(comp)) totUm += comp;
    const areaPx2 = roiReadout(roi).area;
    return {
      file,
      num: lengthsUm.length,
      tot_px: totPx,
      tot_um: totUm,
      max_px: maxPx,
      max_um: maxUm,
      avg_px: has ? mean(lengthsPx) : 0,
      avg_um: has ? mean(lengthsUm) : 0,
      med_px: has ? median(lengthsPx) : 0,
      med_um: has ? median(lengthsUm) : 0,
      roi_area_px2: Math.trunc(areaPx2),
      roi_area_um2: areaPx2 / (ppm * ppm),
    };
  }

  /* ---------- OpenCV-backed pipeline ---------- */

  function create(cv) {
    const CC_STAT_AREA = typeof cv.CC_STAT_AREA === 'number' ? cv.CC_STAT_AREA : 4;
    const white = () => new cv.Scalar(255, 255, 255, 255);

    // One-entry cache of the enhanced image (everything before thresholding), so moving the
    // threshold sliders does not redo the top-hat. Pure memoisation: same code path either way.
    let cache = null;

    function clearCache() {
      if (cache) {
        cache.blur.delete();
        cache = null;
      }
    }

    function grayMat(img) {
      const src = new cv.Mat(img.height, img.width, cv.CV_8UC4);
      src.data.set(img.rgba);
      const gray = new cv.Mat();
      // Same weights as cv2.COLOR_BGR2GRAY; only the channel order of the source differs.
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      src.delete();
      return gray;
    }

    // gray -> ROI mask -> equalizeHist -> invert -> top-hat -> Gaussian blur 5x5
    function enhance(img, roi, topHat) {
      const key = img.id + '|' + roi.join(';') + '|' + topHat;
      if (cache && cache.key === key) return cache.blur;
      clearCache();

      const gray = grayMat(img);
      const mask = cv.Mat.zeros(gray.rows, gray.cols, cv.CV_8UC1);
      const pts = cv.matFromArray(4, 1, cv.CV_32SC2, [].concat.apply([], roi));
      cv.fillConvexPoly(mask, pts, white(), cv.LINE_8, 0);
      const masked = new cv.Mat();
      cv.bitwise_and(gray, mask, masked);

      const eq = new cv.Mat();
      cv.equalizeHist(masked, eq);
      const inv = new cv.Mat();
      cv.bitwise_not(eq, inv);
      const ker = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(topHat, topHat));
      const tht = new cv.Mat();
      cv.morphologyEx(inv, tht, cv.MORPH_TOPHAT, ker);
      const blur = new cv.Mat();
      cv.GaussianBlur(tht, blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

      [gray, mask, pts, masked, eq, inv, ker, tht].forEach((m) => m.delete());
      cache = { key, blur };
      return blur;
    }

    function binarise(blur, p) {
      const bw = new cv.Mat();
      let otsu = null;
      let thr;
      if (p.manualThresh > 0) {
        thr = p.manualThresh;
      } else {
        const tmp = new cv.Mat();
        otsu = cv.threshold(blur, tmp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
        tmp.delete();
        thr = Math.trunc(Math.max(0, Math.min(255, otsu + p.otsuOffset)));
      }
      cv.threshold(blur, bw, thr, 255, cv.THRESH_BINARY);
      return { bw, thr, otsu };
    }

    // connectedComponentsWithStats, keep components with area >= minArea
    function filterArea(bw, minArea) {
      const labels = new cv.Mat();
      const stats = new cv.Mat();
      const cents = new cv.Mat();
      const n = cv.connectedComponentsWithStats(bw, labels, stats, cents, 8, cv.CV_32S);
      const clean = cv.Mat.zeros(bw.rows, bw.cols, cv.CV_8UC1);
      const keep = new Uint8Array(n);
      let kept = 0;
      const s = stats.data32S;
      for (let i = 1; i < n; i++) {
        if (s[i * 5 + CC_STAT_AREA] >= minArea) {
          keep[i] = 255;
          kept++;
        }
      }
      const lab = labels.data32S;
      const out = clean.data;
      for (let k = 0; k < out.length; k++) out[k] = keep[lab[k]];
      labels.delete();
      stats.delete();
      cents.delete();
      return { clean, components: n - 1, kept };
    }

    // Morphological skeleton exactly as in the script.
    function skeleton(clean) {
      const sk = cv.Mat.zeros(clean.rows, clean.cols, cv.CV_8UC1);
      const m = clean.clone();
      const elem = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3, 3));
      const er = new cv.Mat();
      const di = new cv.Mat();
      const df = new cv.Mat();
      for (;;) {
        cv.erode(m, er, elem);
        cv.dilate(er, di, elem);
        cv.subtract(m, di, df);
        cv.bitwise_or(sk, df, sk);
        er.copyTo(m);
        if (cv.countNonZero(m) === 0) break;
      }
      [m, elem, er, di, df].forEach((x) => x.delete());
      return sk;
    }

    function segment(img, roi, p) {
      const blur = enhance(img, roi, p.topHat);
      const { bw, thr, otsu } = binarise(blur, p);
      const { clean, components, kept } = filterArea(bw, p.minArea);
      return { bw, clean, thr, otsu, components, kept };
    }

    // Returns the neurite lengths (px, >= minLength) and the skeleton as a byte array.
    function analyse(img, roi, p) {
      const seg = segment(img, roi, p);
      const sk = skeleton(seg.clean);
      const contours = new cv.MatVector();
      const hier = new cv.Mat();
      cv.findContours(sk, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
      const lengths = [];
      const total = contours.size();
      for (let i = 0; i < total; i++) {
        const c = contours.get(i);
        const len = c.rows;
        c.delete();
        if (len >= p.minLength) lengths.push(len);
      }
      const skBytes = sk.data.slice();
      [seg.bw, seg.clean, sk, contours, hier].forEach((x) => x.delete());
      return {
        lengths,
        skeleton: skBytes,
        thr: seg.thr,
        otsu: seg.otsu,
        components: seg.components,
        kept: seg.kept,
        contours: total,
      };
    }

    // Threshold preview: the binary image the analysis will use.
    // White = kept, dark red = thresholded but discarded by the min-area filter.
    function preview(img, roi, p) {
      const seg = segment(img, roi, p);
      const n = img.width * img.height;
      const out = new Uint8ClampedArray(n * 4);
      const bw = seg.bw.data;
      const clean = seg.clean.data;
      for (let i = 0, o = 0; i < n; i++, o += 4) {
        if (clean[i]) {
          out[o] = out[o + 1] = out[o + 2] = 255;
        } else if (bw[i]) {
          out[o] = 150;
          out[o + 1] = 40;
          out[o + 2] = 40;
        }
        out[o + 3] = 255;
      }
      seg.bw.delete();
      seg.clean.delete();
      return { rgba: out, thr: seg.thr, otsu: seg.otsu, components: seg.components, kept: seg.kept };
    }

    // Overlay image: base per mode, dilated skeleton painted in the overlay colour.
    function overlay(img, skBytes, mode, color, thickness) {
      const w = img.width;
      const h = img.height;
      const n = w * h;
      const out = new Uint8ClampedArray(n * 4);

      if (mode === 'Color') {
        out.set(img.rgba);
      } else {
        const gray = grayMat(img);
        let src = gray;
        if (mode !== 'Grayscale') {
          src = new cv.Mat();
          cv.threshold(gray, src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
        }
        const g = src.data;
        for (let i = 0, o = 0; i < n; i++, o += 4) out[o] = out[o + 1] = out[o + 2] = g[i];
        if (src !== gray) src.delete();
        gray.delete();
      }

      let maskBytes = skBytes;
      if (thickness > 1) {
        const sk = new cv.Mat(h, w, cv.CV_8UC1);
        sk.data.set(skBytes);
        const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(thickness, thickness));
        const dil = new cv.Mat();
        cv.dilate(sk, dil, kern);
        maskBytes = dil.data.slice();
        [sk, kern, dil].forEach((x) => x.delete());
      }
      for (let i = 0, o = 0; i < n; i++, o += 4) {
        if (maskBytes[i]) {
          out[o] = color[0];
          out[o + 1] = color[1];
          out[o + 2] = color[2];
        }
        out[o + 3] = 255;
      }
      return out;
    }

    // Scratch-gap detection from the reference image, as in load_images / preview_reference.
    function autoRoi(img) {
      const w = img.width;
      const h = img.height;
      const gray = grayMat(img);
      const edges = new cv.Mat();
      cv.Canny(gray, edges, 50, 150, 3, false);
      const rs = new cv.Mat(h, 1, cv.CV_64FC1);
      {
        const e = edges.data;
        const r = rs.data64F;
        for (let y = 0; y < h; y++) {
          let sum = 0;
          const off = y * w;
          for (let x = 0; x < w; x++) sum += e[off + x];
          r[y] = sum;
        }
      }
      // Literal port of cv2.GaussianBlur(rs.reshape(-1, 1), (51, 1), 0).
      // ksize is (width, height), so the 51-tap kernel runs across the single column.
      const smMat = new cv.Mat();
      cv.GaussianBlur(rs, smMat, new cv.Size(51, 1), 0, 0, cv.BORDER_DEFAULT);
      const sm = Float64Array.from(smMat.data64F);
      [gray, edges, rs, smMat].forEach((x) => x.delete());

      let mn = Infinity;
      let mx = -Infinity;
      for (let y = 0; y < h; y++) {
        if (sm[y] < mn) mn = sm[y];
        if (sm[y] > mx) mx = sm[y];
      }
      let bestStart = -1;
      let bestLen = 0;
      let runStart = -1;
      for (let y = 0; y <= h; y++) {
        const below = y < h && (sm[y] - mn) / (mx - mn + 1e-9) < 0.1;
        if (below) {
          if (runStart < 0) runStart = y;
        } else if (runStart >= 0) {
          if (y - runStart > bestLen) {
            bestLen = y - runStart;
            bestStart = runStart;
          }
          runStart = -1;
        }
      }
      const found = bestLen > 0;
      const y1 = found ? bestStart : 0;
      const y2 = found ? bestStart + bestLen - 1 : h;
      return { roi: [[0, y1], [w, y1], [w, y2], [0, y2]], found };
    }

    return { enhance, segment, analyse, preview, overlay, autoRoi, clearCache };
  }

  root.NeuritePipeline = { create, summarise, roiReadout, mean, median, pairwiseSum };
})(typeof self !== 'undefined' ? self : this);
