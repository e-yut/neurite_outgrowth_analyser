/*
 * Neurite Outgrowth Analyser - web edition
 * Web Worker: owns OpenCV.js and all pixel data handed to it. Nothing here touches the network
 * apart from loading the vendored scripts that sit next to it.
 */
'use strict';

importScripts(
  '../vendor/opencv/opencv.js',
  '../vendor/pako/pako.min.js',
  '../vendor/utif/UTIF.js',
  'pipeline.js'
);

let pipeline = null;
const images = new Map(); // id -> { id, width, height, rgba }
const skeletons = new Map(); // id -> Uint8Array

// OpenCV.js builds differ: `cv` may be a factory, a promise, or a module that is still starting.
// The 4.9.0 module is a "thenable" that resolves with itself, so it must never be awaited or
// returned from an async function (that would loop forever). Poll for readiness instead.
function loadCv() {
  return new Promise((resolve, reject) => {
    let c = self.cv;
    if (typeof c === 'function') c = c();
    const started = Date.now();
    const adopt = (m) => {
      if (m && m.Mat) c = m;
    };
    if (c && typeof c.then === 'function' && !c.Mat) {
      try {
        c.then(adopt);
      } catch (e) {
        /* fall through to polling */
      }
    }
    const tick = () => {
      if (c && c.Mat) {
        self.cv = c;
        pipeline = NeuritePipeline.create(c);
        resolve();
      } else if (Date.now() - started > 120000) {
        reject(new Error('OpenCV.js did not start'));
      } else {
        setTimeout(tick, 50);
      }
    };
    tick();
  });
}

const ready = loadCv();

function getImage(id) {
  const img = images.get(id);
  if (!img) throw new Error('Image is no longer loaded');
  return img;
}

function drop(id) {
  images.delete(id);
  skeletons.delete(id);
  pipeline.clearCache();
}

const ops = {
  init() {
    return { ok: true };
  },

  // TIFF files are decoded here because browsers cannot decode them.
  decodeTiff({ id, buffer }) {
    const ifds = UTIF.decode(buffer);
    if (!ifds.length) throw new Error('No image found in TIFF file');
    // Use the largest page (some microscopes store a thumbnail as well).
    let ifd = ifds[0];
    let best = 0;
    for (const d of ifds) {
      const w = d.t256 ? d.t256[0] : 0;
      const h = d.t257 ? d.t257[0] : 0;
      if (w * h > best) {
        best = w * h;
        ifd = d;
      }
    }
    UTIF.decodeImage(buffer, ifd);
    const rgba = UTIF.toRGBA8(ifd);
    const width = ifd.width;
    const height = ifd.height;
    if (!width || !height || rgba.length < width * height * 4) {
      throw new Error('Unsupported TIFF format');
    }
    // cv2.imread drops alpha; do the same so the pipeline sees opaque pixels.
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    images.set(id, { id, width, height, rgba });
    const copy = new Uint8ClampedArray(rgba);
    return { result: { width, height, rgba: copy }, transfer: [copy.buffer] };
  },

  store({ id, width, height, rgba }) {
    images.set(id, { id, width, height, rgba: new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength) });
    return { ok: true };
  },

  remove({ id }) {
    drop(id);
    return { ok: true };
  },

  autoRoi({ id }) {
    return pipeline.autoRoi(getImage(id));
  },

  preview({ id, roi, params }) {
    const r = pipeline.preview(getImage(id), roi, params);
    return { result: r, transfer: [r.rgba.buffer] };
  },

  analyse({ id, roi, params, overlay }) {
    const img = getImage(id);
    const r = pipeline.analyse(img, roi, params);
    skeletons.set(id, r.skeleton);
    const rgba = pipeline.overlay(img, r.skeleton, overlay.mode, overlay.color, overlay.thickness);
    return {
      result: {
        lengths: r.lengths,
        thr: r.thr,
        otsu: r.otsu,
        components: r.components,
        kept: r.kept,
        contours: r.contours,
        overlay: rgba,
      },
      transfer: [rgba.buffer],
    };
  },

  // Repaint an overlay from the cached skeleton (colour, thickness or mode changed).
  overlay({ id, overlay }) {
    const sk = skeletons.get(id);
    if (!sk) return { overlay: null };
    const rgba = pipeline.overlay(getImage(id), sk, overlay.mode, overlay.color, overlay.thickness);
    return { result: { overlay: rgba }, transfer: [rgba.buffer] };
  },
};

self.onmessage = async (ev) => {
  const { seq, op, args } = ev.data;
  try {
    await ready;
    if (!ops[op]) throw new Error('Unknown operation: ' + op);
    const out = ops[op](args || {});
    if (out && out.transfer) self.postMessage({ seq, ok: true, result: out.result }, out.transfer);
    else self.postMessage({ seq, ok: true, result: out });
  } catch (err) {
    let message = err && err.message ? err.message : String(err);
    // OpenCV.js throws raw pointers for C++ exceptions.
    if (typeof err === 'number' && self.cv && self.cv.exceptionFromPtr) {
      try {
        message = self.cv.exceptionFromPtr(err).msg;
      } catch (e) {
        message = 'OpenCV error ' + err;
      }
    }
    self.postMessage({ seq, ok: false, error: message });
  }
};
