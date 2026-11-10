// Image viewer with zoom/pan and an editable quadrilateral ROI (four corners + whole-shape drag).

const HANDLE = 7; // handle half-size in CSS pixels
const HIT = 12;

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.source = null;
    this.w = 0;
    this.h = 0;
    this.roi = null;
    this.editable = false;
    this.scale = 1;
    this.ox = 0;
    this.oy = 0;
    this.message = '';
    this.onRoiChange = null; // (roi, done) => void
    this.drag = null;
    this._fitted = false;

    canvas.addEventListener('pointerdown', (e) => this._down(e));
    canvas.addEventListener('pointermove', (e) => this._move(e));
    canvas.addEventListener('pointerup', (e) => this._up(e));
    canvas.addEventListener('pointercancel', (e) => this._up(e));
    canvas.addEventListener('wheel', (e) => this._wheel(e), { passive: false });
    canvas.addEventListener('dblclick', () => this.fit());
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._resize();
  }

  // source: anything drawImage accepts (or null). Keeps the view when the size is unchanged.
  setSource(source, w, h, message) {
    const sameSize = this.source && source && this.w === w && this.h === h;
    this.source = source;
    this.message = message || '';
    if (source) {
      this.w = w;
      this.h = h;
      if (!sameSize || !this._fitted) this.fit();
    }
    this.draw();
  }

  setRoi(roi, editable) {
    this.roi = roi;
    this.editable = !!editable;
    this.draw();
  }

  fit() {
    const cw = this.canvas.clientWidth;
    const ch = this.canvas.clientHeight;
    if (!this.source || !cw || !ch) return;
    const pad = 16;
    this.scale = Math.min((cw - 2 * pad) / this.w, (ch - 2 * pad) / this.h);
    if (!(this.scale > 0)) this.scale = 1;
    this.ox = (cw - this.w * this.scale) / 2;
    this.oy = (ch - this.h * this.scale) / 2;
    this._fitted = true;
    this.draw();
  }

  zoomBy(factor, cx, cy) {
    if (!this.source) return;
    if (cx === undefined) {
      cx = this.canvas.clientWidth / 2;
      cy = this.canvas.clientHeight / 2;
    }
    const s = Math.max(0.02, Math.min(40, this.scale * factor));
    const ix = (cx - this.ox) / this.scale;
    const iy = (cy - this.oy) / this.scale;
    this.scale = s;
    this.ox = cx - ix * s;
    this.oy = cy - iy * s;
    this.draw();
  }

  _resize() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const cw = c.clientWidth;
    const ch = c.clientHeight;
    if (!cw || !ch) return;
    c.width = Math.round(cw * dpr);
    c.height = Math.round(ch * dpr);
    if (this.source && !this.drag) this.fit();
    else this.draw();
  }

  _pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _toImage(p) {
    return [(p[0] - this.ox) / this.scale, (p[1] - this.oy) / this.scale];
  }

  _toScreen(p) {
    return [p[0] * this.scale + this.ox, p[1] * this.scale + this.oy];
  }

  _hitHandle(p) {
    if (!this.roi || !this.editable) return -1;
    let best = -1;
    let bestD = HIT;
    this.roi.forEach((pt, i) => {
      const s = this._toScreen(pt);
      const d = Math.hypot(s[0] - p[0], s[1] - p[1]);
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  }

  _inside(ip) {
    if (!this.roi || !this.editable) return false;
    let inside = false;
    const r = this.roi;
    for (let i = 0, j = 3; i < 4; j = i++) {
      if ((r[i][1] > ip[1]) !== (r[j][1] > ip[1]) &&
        ip[0] < ((r[j][0] - r[i][0]) * (ip[1] - r[i][1])) / (r[j][1] - r[i][1]) + r[i][0]) {
        inside = !inside;
      }
    }
    return inside;
  }

  _down(e) {
    if (!this.source || e.button > 1) return;
    const p = this._pos(e);
    const corner = e.button === 0 ? this._hitHandle(p) : -1;
    if (corner >= 0) {
      this.drag = { type: 'corner', corner };
    } else if (e.button === 0 && !e.shiftKey && this._inside(this._toImage(p))) {
      this.drag = { type: 'shape', start: this._toImage(p), orig: this.roi.map((q) => q.slice()) };
    } else {
      this.drag = { type: 'pan', start: p, ox: this.ox, oy: this.oy };
    }
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch (err) {
      /* capture is a convenience only */
    }
    e.preventDefault();
  }

  _move(e) {
    const p = this._pos(e);
    const d = this.drag;
    if (!d) {
      const c = this._hitHandle(p) >= 0 ? 'crosshair' : this._inside(this._toImage(p)) ? 'move' : 'grab';
      this.canvas.style.cursor = this.source ? c : 'default';
      return;
    }
    if (d.type === 'pan') {
      this.ox = d.ox + p[0] - d.start[0];
      this.oy = d.oy + p[1] - d.start[1];
      this.draw();
      return;
    }
    const ip = this._toImage(p);
    if (d.type === 'corner') {
      this.roi[d.corner] = [
        Math.max(0, Math.min(this.w, Math.round(ip[0]))),
        Math.max(0, Math.min(this.h, Math.round(ip[1]))),
      ];
    } else {
      // Rigid move: clamp the offset so every corner stays inside the image.
      let dx = Math.round(ip[0] - d.start[0]);
      let dy = Math.round(ip[1] - d.start[1]);
      for (const q of d.orig) {
        dx = Math.max(-q[0], Math.min(this.w - q[0], dx));
        dy = Math.max(-q[1], Math.min(this.h - q[1], dy));
      }
      this.roi = d.orig.map((q) => [q[0] + dx, q[1] + dy]);
    }
    d.moved = true;
    this.draw();
    if (this.onRoiChange) this.onRoiChange(this.roi, false);
  }

  _up(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    if (d.moved && this.onRoiChange) this.onRoiChange(this.roi, true);
  }

  _wheel(e) {
    if (!this.source) return;
    e.preventDefault();
    const p = this._pos(e);
    this.zoomBy(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)), p[0], p[1]);
  }

  draw() {
    const c = this.canvas;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = c.clientWidth;
    const ch = c.clientHeight;

    if (this.source) {
      ctx.imageSmoothingEnabled = this.scale < 1;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.source, this.ox, this.oy, this.w * this.scale, this.h * this.scale);
    }

    if (this.source && this.roi) {
      const pts = this.roi.map((q) => this._toScreen(q));
      ctx.beginPath();
      pts.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
      ctx.closePath();
      if (this.editable) {
        ctx.fillStyle = 'rgba(255, 214, 10, 0.10)';
        ctx.fill();
      }
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
      ctx.setLineDash([]);
      ctx.stroke();
      ctx.strokeStyle = '#ffd60a';
      ctx.setLineDash(this.editable ? [] : [6, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      if (this.editable) {
        for (const q of pts) {
          ctx.fillStyle = '#ffd60a';
          ctx.strokeStyle = '#111';
          ctx.lineWidth = 1.5;
          ctx.fillRect(q[0] - HANDLE, q[1] - HANDLE, 2 * HANDLE, 2 * HANDLE);
          ctx.strokeRect(q[0] - HANDLE, q[1] - HANDLE, 2 * HANDLE, 2 * HANDLE);
        }
      }
    }

    if (this.message) {
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const tw = ctx.measureText(this.message).width + 28;
      ctx.fillStyle = 'rgba(15, 20, 28, 0.82)';
      ctx.fillRect((cw - tw) / 2, ch / 2 - 20, tw, 40);
      ctx.fillStyle = '#e8edf3';
      ctx.fillText(this.message, cw / 2, ch / 2);
    }
  }
}
