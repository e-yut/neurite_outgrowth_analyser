#!/usr/bin/env python3
"""
Fixed and cleaned version of neurite_analysis_gui_v6.7
Fixes: import order, missing imports, indentation, CSV/Excel export, ROI area bookkeeping,
       preview_threshold method placement, overlay export file naming, and robustness.
"""

import json
import os
import math
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd  # required for Excel export

# --------------------------
# Small helper: ToolTip class
# --------------------------
class ToolTip:
    """Simple tooltip for showing descriptive text on hover."""
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.text = None

    def show(self, text, x, y):
        if self.tipwindow or not text:
            return
        self.text = text
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        self.tipwindow.wm_geometry(f"+{x+20}+{y+10}")
        label = tk.Label(self.tipwindow, text=text, background="#ffffe0",
                         relief="solid", borderwidth=1,
                         font=("tahoma", "8", "normal"), justify="left")
        label.pack(ipadx=4, ipady=2)

    def hide(self):
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None
        self.text = None

# --------------------------
# DraggablePolygon (ROI)
# --------------------------
class DraggablePolygon:
    """Draggable quadrilateral ROI with corner handles and whole-polygon drag."""
    def __init__(self, canvas, points, callback):
        self.canvas = canvas
        self.callback = callback
        self.points = points[:]  # list of (x,y)
        self.handle_radius = 15
        self.handles = []
        self.poly = None
        self._draw()
        canvas.tag_bind('handle', '<B1-Motion>', self._move_handle)
        canvas.tag_bind('poly', '<ButtonPress-1>', self._start_polygon)
        canvas.tag_bind('poly', '<B1-Motion>', self._move_polygon)

    def _draw(self):
        # Delete previous items
        for h in self.handles:
            try:
                self.canvas.delete(h)
            except Exception:
                pass
        if self.poly:
            try:
                self.canvas.delete(self.poly)
            except Exception:
                pass

        coords = sum(self.points, ())
        self.poly = self.canvas.create_polygon(
            *coords, outline='red', fill='', width=2, tags=('poly',)
        )

        self.handles = []
        for idx, (x, y) in enumerate(self.points):
            r = self.handle_radius
            h = self.canvas.create_oval(
                x-r, y-r, x+r, y+r,
                fill='blue', tags=('handle', str(idx))
            )
            self.handles.append(h)

    def _move_handle(self, event):
        # Keep within canvas area
        try:
            cw = int(self.canvas['width'])
            ch = int(self.canvas['height'])
        except Exception:
            cw, ch = 10000, 10000
        x = max(0, min(event.x, cw))
        y = max(0, min(event.y, ch))
        idx = int(self.canvas.gettags(tk.CURRENT)[1])
        self.points[idx] = (x, y)
        r = self.handle_radius
        self.canvas.coords(self.handles[idx], x-r, y-r, x+r, y+r)
        self.canvas.coords(self.poly, *sum(self.points, ()))
        self.callback(self.points)

    def _start_polygon(self, event):
        self._lastx = event.x
        self._lasty = event.y

    def _move_polygon(self, event):
        dx = event.x - self._lastx
        dy = event.y - self._lasty
        try:
            cw = int(self.canvas['width'])
            ch = int(self.canvas['height'])
        except Exception:
            cw, ch = 10000, 10000
        new_points = []
        for x, y in self.points:
            nx = max(0, min(int(x + dx), cw))
            ny = max(0, min(int(y + dy), ch))
            new_points.append((nx, ny))
        self.points = new_points
        self._lastx = event.x
        self._lasty = event.y
        self._draw()
        self.callback(self.points)

# --------------------------
# Main application
# --------------------------
class NeuriteApp:
    def __init__(self, master):
        self.master = master
        master.title("Neurite Outgrowth Analysis v6.7.1 (by Eylan Yutuc)")

        # state
        self.ref_path = ""
        self.img_paths = []
        self.roi_dict = {}
        self.ref_roi = None
        self.results = []

        # settings
        self.overlay_color = (0, 0, 255)
        self.overlay_mode = tk.StringVar(value="Color")
        self.px_per_um = tk.DoubleVar(value=0.287)
        self.overlay_thickness = tk.IntVar(value=3)

        self.add_scalebar = tk.BooleanVar(value=False)
        self.scalebar_length_um = tk.DoubleVar(value=1000.0)
        self.scalebar_position = tk.StringVar(value="Bottom Right")
        self.scalebar_label = tk.StringVar(value="1000 µm")
        self.scalebar_color = (255, 255, 255)

        # GUI controls
        ctrl = tk.Frame(master); ctrl.pack(fill='x', padx=5, pady=5)
        tk.Label(ctrl, text="Reference (0 h image):").grid(row=0, column=0, sticky='w')
        self.ref_entry = tk.Entry(ctrl, width=40); self.ref_entry.grid(row=0, column=1)
        tk.Button(ctrl, text="Browse", command=self.load_reference).grid(row=0, column=2)
        tk.Button(ctrl, text="Adjust Reference ROI", command=self.preview_reference).grid(row=0, column=3, padx=5)

        tk.Label(ctrl, text="Images to analyse:").grid(row=1, column=0, sticky='w')
        self.img_entry = tk.Entry(ctrl, width=40); self.img_entry.grid(row=1, column=1)
        tk.Button(ctrl, text="Browse", command=self.load_images).grid(row=1, column=2)

        tk.Label(ctrl, text="Output folder:").grid(row=2, column=0, sticky='w')
        self.out_entry = tk.Entry(ctrl, width=40); self.out_entry.grid(row=2, column=1)
        tk.Button(ctrl, text="Browse", command=self.load_output).grid(row=2, column=2)

        tk.Label(ctrl, text="Overlay mode:").grid(row=3, column=0, sticky='w')
        tk.OptionMenu(ctrl, self.overlay_mode, "Color", "Grayscale", "B/W").grid(row=3, column=1, sticky='w')

        tk.Label(ctrl, text="Pixels per µm:").grid(row=4, column=0, sticky='w')
        tk.Entry(ctrl, textvariable=self.px_per_um, width=10).grid(row=4, column=1, sticky='w')

        tk.Label(ctrl, text="Overlay thickness (px):").grid(row=5, column=0, sticky='w')
        tk.Scale(ctrl, from_=1, to=10, orient='horizontal',
                 variable=self.overlay_thickness).grid(row=5, column=1, sticky='w')
        tk.Button(ctrl, text="Overlay Colour", command=self.choose_color).grid(row=5, column=2, sticky='w')

        # Scale bar
        sbf = tk.LabelFrame(master, text="Scale Bar Settings"); sbf.pack(fill='x', padx=5, pady=5)
        tk.Checkbutton(sbf, text="Add scale bar", variable=self.add_scalebar).grid(row=0, column=0, sticky='w')
        tk.Label(sbf, text="Length (µm):").grid(row=1, column=0, sticky='w')
        tk.Entry(sbf, textvariable=self.scalebar_length_um, width=10).grid(row=1, column=1, sticky='w')
        tk.Label(sbf, text="Position:").grid(row=1, column=2, sticky='w')
        tk.OptionMenu(sbf, self.scalebar_position,
                      "Bottom Right", "Bottom Left", "Top Right", "Top Left").grid(row=1, column=3, sticky='w')
        tk.Label(sbf, text="Label:").grid(row=2, column=0, sticky='w')
        tk.Entry(sbf, textvariable=self.scalebar_label, width=10).grid(row=2, column=1, sticky='w')
        tk.Button(sbf, text="Bar Colour", command=self.choose_scalebar_color).grid(row=2, column=2, sticky='w')

        # Detection params
        param = tk.LabelFrame(master, text="Detection Parameters"); param.pack(fill='x', padx=5, pady=5)
        self.top_hat = tk.IntVar(value=15)
        self.otsu_off = tk.DoubleVar(value=0.0)
        self.manual_thresh = tk.IntVar(value=100)
        self.min_area = tk.IntVar(value=150)
        self.min_length = tk.IntVar(value=50)

        tk.Label(param, text="Top-hat size (px):").grid(row=0, column=0, sticky='w')
        tk.Scale(param, from_=3, to=51, resolution=2, orient='horizontal', variable=self.top_hat).grid(row=0, column=1, sticky='ew')
        tk.Label(param, text="Structuring element for top-hat.").grid(row=1, column=1, sticky='w')

        tk.Label(param, text="Otsu offset:").grid(row=2, column=0, sticky='w')
        tk.Scale(param, from_=-50, to=50, orient='horizontal', variable=self.otsu_off).grid(row=2, column=1, sticky='ew')
        tk.Label(param, text="Adjust Otsu threshold.").grid(row=3, column=1, sticky='w')

        tk.Label(param, text="Manual thresh (0=auto):").grid(row=4, column=0, sticky='w')
        tk.Scale(param, from_=0, to=255, orient='horizontal', variable=self.manual_thresh).grid(row=4, column=1, sticky='ew')
        tk.Label(param, text="0 uses Otsu+offset.").grid(row=5, column=1, sticky='w')

        tk.Label(param, text="Min area (px):").grid(row=6, column=0, sticky='w')
        tk.Scale(param, from_=0, to=5000, resolution=10, orient='horizontal', variable=self.min_area).grid(row=6, column=1, sticky='ew')
        tk.Label(param, text="Discard small components.").grid(row=7, column=1, sticky='w')

        tk.Label(param, text="Min length (px):").grid(row=8, column=0, sticky='w')
        tk.Scale(param, from_=0, to=200, orient='horizontal', variable=self.min_length).grid(row=8, column=1, sticky='ew')
        tk.Label(param, text="Discard short skeletons.").grid(row=9, column=1, sticky='w')

        # Actions
        btnf = tk.Frame(master); btnf.pack(fill='x', padx=5, pady=5)
        tk.Button(btnf, text="Preview/Adjust ROI", command=self.preview).pack(side='left', padx=5)
        tk.Button(btnf, text="Run Analysis", command=self.run_analysis).pack(side='left', padx=5)
        tk.Button(btnf, text="Export CSV & Overlays", command=self.export).pack(side='left', padx=5)

        # Results table with descriptive headers
        cols = ('file', 'num', 'tot_px', 'tot_um', 'max_px', 'max_um', 'avg_px', 'avg_um', 'med_px', 'med_um')
        display_cols = ('File', 'Neurite Count', 'Total Neurite Length (px)', 'Total Neurite Length (µm)',
                        'Max Neurite Length (px)', 'Max Neurite Length (µm)',
                        'Average Neurite Length (px)', 'Average Neurite Length (µm)',
                        'Median Neurite Length (px)', 'Median Neurite Length (µm)')
        column_tooltips = {
            'file': 'Name of the image file.',
            'num': 'Total number of neurite segments detected.',
            'tot_px': 'Sum of the lengths of all detected neurites in pixels.',
            'tot_um': 'Sum of the lengths of all detected neurites in microns.',
            'max_px': 'Length of the longest detected neurite segment in pixels.',
            'max_um': 'Length of the longest detected neurite segment in microns.',
            'avg_px': 'Mean length of detected neurites in pixels.',
            'avg_um': 'Mean length of detected neurites in microns.',
            'med_px': 'Median length of detected neurites in pixels.',
            'med_um': 'Median length of detected neurites in microns.'
        }
        self.tree = ttk.Treeview(master, columns=cols, show='headings')
        for col, display_col in zip(cols, display_cols):
            self.tree.heading(col, text=display_col)
            self.tree.column(col, width=120 if col != 'file' else 220, anchor='center')

        # header tooltips
        self.tooltip = ToolTip(self.tree)
        self.current_column = None

        def on_motion(event):
            region = self.tree.identify("region", event.x, event.y)
            if region == "heading":
                column = self.tree.identify_column(event.x)  # '#1'...
                try:
                    col_index = int(column.replace('#', '')) - 1
                except Exception:
                    self.tooltip.hide()
                    self.current_column = None
                    return
                if 0 <= col_index < len(cols):
                    col_key = cols[col_index]
                    if self.current_column != col_key:
                        self.tooltip.hide()
                        tip = column_tooltips.get(col_key, "")
                        self.tooltip.show(tip, event.x_root, event.y_root)
                        self.current_column = col_key
                else:
                    self.tooltip.hide()
                    self.current_column = None
            else:
                self.tooltip.hide()
                self.current_column = None

        self.tree.bind('<Motion>', on_motion)
        self.tree.bind('<Leave>', lambda e: self.tooltip.hide())
        self.tree.pack(fill='both', expand=True, padx=5, pady=5)

    # --------------------------
    # Small UI helpers
    # --------------------------
    def choose_color(self):
        clr = colorchooser.askcolor()[1]
        if clr:
            h = clr.lstrip('#')
            self.overlay_color = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    def choose_scalebar_color(self):
        clr = colorchooser.askcolor()[1]
        if clr:
            h = clr.lstrip('#')
            self.scalebar_color = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    def load_reference(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.tif;*.tiff")])
        if p:
            self.ref_path = p
            self.ref_entry.delete(0, tk.END)
            self.ref_entry.insert(0, p)
            self.ref_roi = None

    def load_images(self):
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png;*.jpg;*.tif;*.tiff")])
        if not paths:
            return
        self.img_paths = list(paths)
        self.img_entry.delete(0, tk.END)
        self.img_entry.insert(0, ";".join(self.img_paths))

        # attempt to auto-derive ROI from reference if available
        default_roi = None
        if self.ref_path and os.path.exists(self.ref_path):
            try:
                ref = cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE)
                h, w = ref.shape
                edges = cv2.Canny(ref, 50, 150)
                rs = edges.sum(axis=1).astype(float)
                sm = cv2.GaussianBlur(rs.reshape(-1, 1), (51, 1), 0).flatten()
                norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)
                gap = np.where(norm < 0.1)[0]
                if gap.size:
                    groups = np.split(gap, np.where(np.diff(gap) != 1)[0] + 1)
                    lg = max(groups, key=len)
                    y1, y2 = lg[0], lg[-1]
                else:
                    y1, y2 = 0, h
                default_roi = [(0, y1), (w, y1), (w, y2), (0, y2)]
            except Exception:
                default_roi = None

        for p in self.img_paths:
            if self.ref_roi:
                self.roi_dict[p] = self.ref_roi[:]
            elif default_roi:
                self.roi_dict[p] = default_roi[:]
            else:
                # full image fallback
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                h0, w0 = img.shape
                self.roi_dict[p] = [(0, 0), (w0, 0), (w0, h0), (0, h0)]

    def load_output(self):
        d = filedialog.askdirectory()
        if d:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, d)

    # compute readout: side lengths and polygon area
    def _compute_readout(self, pts):
        top = math.dist(pts[0], pts[1])
        right = math.dist(pts[1], pts[2])
        bottom = math.dist(pts[3], pts[2])
        left = math.dist(pts[0], pts[3])
        area = abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2
        return {'top': top, 'right': right, 'bottom': bottom, 'left': left}, area

    def _format_readout_text(self, lengths, area):
        return (f"T:{lengths['top']:.1f}px R:{lengths['right']:.1f}px "
                f"B:{lengths['bottom']:.1f}px L:{lengths['left']:.1f}px | A:{area:.0f}px²")

    # --------------------------
    # Preview reference ROI
    # --------------------------
    def preview_reference(self):
        if not self.ref_path or not os.path.exists(self.ref_path):
            messagebox.showerror("Error", "No reference image loaded")
            return
        win = tk.Toplevel(self.master)
        win.title("Adjust Reference ROI")
        canvas = tk.Canvas(win, bg='white')
        hbar = tk.Scrollbar(win, orient='horizontal', command=canvas.xview)
        vbar = tk.Scrollbar(win, orient='vertical', command=canvas.yview)
        canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        hbar.grid(row=1, column=0, sticky='ew')
        vbar.grid(row=0, column=1, sticky='ns')
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        img_cv = cv2.imread(self.ref_path)
        h0, w0 = img_cv.shape[:2]
        zoom = tk.DoubleVar(value=25.0)

        readout_lbl = tk.Label(win, text="")
        readout_lbl.grid(row=2, column=0, sticky='w')

        def update_measurements():
            if not self.ref_roi:
                return
            lengths, area = self._compute_readout(self.ref_roi)
            readout_lbl.config(text=self._format_readout_text(lengths, area))

        def draw_ref(_=None):
            canvas.delete("all")
            z = zoom.get() / 100.0
            w_vis, h_vis = max(1, int(w0 * z)), max(1, int(h0 * z))
            canvas.config(width=w_vis, height=h_vis, scrollregion=(0, 0, w_vis, h_vis))
            pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(pil.resize((w_vis, h_vis), Image.LANCZOS))
            canvas.create_image(0, 0, anchor='nw', image=imgtk)
            canvas.image = imgtk

            if self.ref_roi:
                pts = [(int(x * z), int(y * z)) for x, y in self.ref_roi]
            else:
                # derive ROI from edges if possible
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                rs = edges.sum(axis=1).astype(float)
                sm = cv2.GaussianBlur(rs.reshape(-1, 1), (51, 1), 0).flatten()
                norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-9)
                gap = np.where(norm < 0.1)[0]
                if gap.size:
                    groups = np.split(gap, np.where(np.diff(gap) != 1)[0] + 1)
                    lg = max(groups, key=len)
                    y1, y2 = lg[0], lg[-1]
                else:
                    y1, y2 = 0, h0
                pts = [(0, int(y1 * z)), (int(w0 * z), int(y1 * z)),
                       (int(w0 * z), int(y2 * z)), (0, int(y2 * z))]

            def on_roi_change(new_pts):
                # convert scaled points back to image coords and store
                self.ref_roi = [(int(x / z), int(y / z)) for x, y in new_pts]
                for p in self.img_paths:
                    self.roi_dict[p] = self.ref_roi[:]

            DraggablePolygon(canvas, pts, on_roi_change)

        tk.Scale(win, from_=1, to=100, orient='horizontal', label='Zoom %', variable=zoom, command=draw_ref).grid(row=3, column=0, sticky='ew')
        tk.Button(win, text="Update Measurements", command=update_measurements).grid(row=4, column=0, pady=5)

        draw_ref()

    # --------------------------
    # Preview per-image ROIs
    # --------------------------
    def preview(self):
        if not self.img_paths:
            messagebox.showerror("Error", "No images to preview")
            return
        win = tk.Toplevel(self.master)
        win.title("Adjust ROI per Image")
        canvas = tk.Canvas(win, bg='white')
        hbar = tk.Scrollbar(win, orient='horizontal', command=canvas.xview)
        vbar = tk.Scrollbar(win, orient='vertical', command=canvas.yview)
        canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        hbar.grid(row=1, column=0, sticky='ew')
        vbar.grid(row=0, column=1, sticky='ns')
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor='nw')

        zoom = tk.DoubleVar(value=25.0)
        readouts = {}

        def load_all(_=None):
            for w in inner.winfo_children():
                w.destroy()
            z = zoom.get() / 100.0
            cols = min(3, max(1, len(self.img_paths)))
            for idx, p in enumerate(self.img_paths):
                rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                h0, w0 = rgb.shape[:2]
                ph = ImageTk.PhotoImage(Image.fromarray(rgb).resize((max(1, int(w0 * z)), max(1, int(h0 * z))), Image.LANCZOS))
                frm = tk.Frame(inner, bd=2, relief='groove')
                frm.grid(row=idx // cols, column=idx % cols, padx=5, pady=5)
                lbl = tk.Label(frm, image=ph); lbl.image = ph; lbl.pack()
                tk.Label(frm, text=os.path.basename(p)).pack()
                c = tk.Canvas(frm, width=max(1, int(w0 * z)), height=max(1, int(h0 * z))); c.pack()
                c.create_image(0, 0, anchor='nw', image=ph); c.image = ph

                read_lbl = tk.Label(frm, text="")
                read_lbl.pack()
                readouts[p] = read_lbl

                pts = [(int(x * z), int(y * z)) for x, y in self.roi_dict[p]]

                def on_roi_change(new_pts, path=p):
                    self.roi_dict[path] = [(int(x / z), int(y / z)) for x, y in new_pts]

                DraggablePolygon(c, pts, on_roi_change)

            inner.update_idletasks()
            canvas.config(scrollregion=canvas.bbox('all'))

        def update_measurements():
            for p, lbl in readouts.items():
                lengths, area = self._compute_readout(self.roi_dict[p])
                lbl.config(text=self._format_readout_text(lengths, area))

        tk.Scale(win, from_=1, to=100, orient='horizontal', label='Zoom %', variable=zoom, command=load_all).grid(row=2, column=0, sticky='ew')
        tk.Button(win, text="Update Measurements", command=update_measurements).grid(row=3, column=0, pady=5)
        load_all()

    # --------------------------
    # Threshold preview method (single-image interactive preview)
    # --------------------------
    def preview_threshold(self):
        if not self.img_paths:
            messagebox.showerror("Error", "No images loaded")
            return
        p = self.img_paths[0]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        tht = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (self.top_hat.get(),) * 2))
        blur = cv2.GaussianBlur(tht, (5, 5), 0)
        if self.manual_thresh.get() > 0:
            _, bw = cv2.threshold(blur, self.manual_thresh.get(), 255, cv2.THRESH_BINARY)
        else:
            otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tv = int(max(0, min(255, otsu_val + self.otsu_off.get())))
            _, bw = cv2.threshold(blur, tv, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold Preview", bw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --------------------------
    # Run analysis
    # --------------------------
    def run_analysis(self):
        if not self.img_paths:
            messagebox.showerror("Error", "No images loaded")
            return
        ppm = self.px_per_um.get()
        if ppm <= 0:
            messagebox.showerror("Error", "Pixels per µm must be positive")
            return

        self.results.clear()
        self.tree.delete(*self.tree.get_children())
        thickness = self.overlay_thickness.get()

        for p in self.img_paths:
            pts = np.array(self.roi_dict[p], dtype=np.int32)
            img = cv2.imread(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            mask = np.zeros_like(gray)
            cv2.fillConvexPoly(mask, pts, 255)
            roi = cv2.bitwise_and(gray, mask)

            eq = cv2.equalizeHist(roi)
            inv = cv2.bitwise_not(eq)
            ker = cv2.getStructuringElement(cv2.MORPH_RECT, (self.top_hat.get(),) * 2)
            tht = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, ker)
            blur = cv2.GaussianBlur(tht, (5, 5), 0)

            if self.manual_thresh.get() > 0:
                _, bw = cv2.threshold(blur, self.manual_thresh.get(), 255, cv2.THRESH_BINARY)
            else:
                otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                tv = int(max(0, min(255, otsu_val + self.otsu_off.get())))
                _, bw = cv2.threshold(blur, tv, 255, cv2.THRESH_BINARY)

            n_lbl, lab, stat, _ = cv2.connectedComponentsWithStats(bw)
            clean = np.zeros_like(bw)
            for i in range(1, n_lbl):
                if stat[i, cv2.CC_STAT_AREA] >= self.min_area.get():
                    clean[lab == i] = 255

            sk = np.zeros_like(clean)
            m = clean.copy()
            elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # skeletonise-like thinning via morphological approach (same approach as before)
            while True:
                er = cv2.erode(m, elem)
                di = cv2.dilate(er, elem)
                df = cv2.subtract(m, di)
                sk = cv2.bitwise_or(sk, df)
                m = er.copy()
                if cv2.countNonZero(m) == 0:
                    break

            cnts = cv2.findContours(sk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            lengths_px = [len(c) for c in cnts if len(c) >= self.min_length.get()]
            lengths_um = [lp / ppm for lp in lengths_px]

            num = len(lengths_um)
            total_px = sum(lengths_px); total_um = sum(lengths_um)
            max_px = max(lengths_px) if lengths_px else 0
            max_um = max(lengths_um) if lengths_um else 0.0
            avg_px = float(np.mean(lengths_px)) if lengths_px else 0.0
            avg_um = float(np.mean(lengths_um)) if lengths_px else 0.0
            med_px = float(np.median(lengths_px)) if lengths_px else 0.0
            med_um = float(np.median(lengths_um)) if lengths_um else 0.0

            mode = self.overlay_mode.get()
            if mode == "Color":
                base = img.copy()
            elif mode == "Grayscale":
                base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                _, b2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                base = cv2.cvtColor(b2, cv2.COLOR_GRAY2BGR)

            if thickness > 1:
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
                mask_sk = cv2.dilate(sk, kern)
            else:
                mask_sk = sk

            overlay = base.copy()
            ys, xs = np.nonzero(mask_sk)
            overlay[ys, xs] = self.overlay_color

            fn = os.path.basename(p)

            # compute ROI area (pixels) and add to record
            lengths, area_px2 = self._compute_readout(list(self.roi_dict[p]))
            area_um2 = area_px2 / (ppm ** 2)

            r = {
                'file': fn, 'num': num,
                'tot_px': total_px, 'tot_um': total_um,
                'max_px': max_px, 'max_um': max_um,
                'avg_px': avg_px, 'avg_um': avg_um,
                'med_px': med_px, 'med_um': med_um,
                'roi_area_px2': int(area_px2), 'roi_area_um2': float(area_um2),
                'overlay': overlay
            }

            self.results.append(r)
            self.tree.insert("", "end", values=(
                fn, num,
                total_px, f"{total_um:.2f}",
                max_px, f"{max_um:.2f}",
                f"{avg_px:.1f}", f"{avg_um:.2f}",
                f"{med_px:.1f}", f"{med_um:.2f}"
            ))

        messagebox.showinfo("Done", f"Processed {len(self.results)} images")

    # --------------------------
    # Export results (CSV + Excel + overlay images)
    # --------------------------
    def export(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export")
            return
        out = self.out_entry.get()
        if not out:
            messagebox.showerror("Error", "No output folder set")
            return
        os.makedirs(out, exist_ok=True)

        # CSV export
        csv_path = os.path.join(out, "neurite_results.csv")
        csv_fieldnames = [
            'file', 'num',
            'tot_px', 'tot_um', 'max_px', 'max_um',
            'avg_px', 'avg_um', 'med_px', 'med_um',
            'roi_area_px2', 'roi_area_um2'
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            for r in self.results:
                # copy only fields that are present (overlay excluded)
                outrow = {k: r.get(k, "") for k in csv_fieldnames}
                writer.writerow(outrow)

        # Excel export via pandas (exclude overlay column)
        try:
            excel_path = os.path.join(out, "neurite_results.xlsx")
            df = pd.DataFrame(self.results)
            if 'overlay' in df.columns:
                df = df.drop(columns=['overlay'])
            df.to_excel(excel_path, index=False)
        except Exception as e:
            # not fatal; warn user but continue saving CSV and overlays
            messagebox.showwarning("Excel export failed", f"Could not write Excel file: {e}")

        # write overlay images (PNG)
        for r in self.results:
            img = r['overlay'].copy()
            if self.add_scalebar.get():
                ppm = self.px_per_um.get()
                length_px = int(self.scalebar_length_um.get() * ppm)
                h, w = img.shape[:2]
                m = 10
                pos = self.scalebar_position.get()
                if pos == "Bottom Right":
                    x1, y1 = w - m - length_px, h - m
                elif pos == "Bottom Left":
                    x1, y1 = m, h - m
                elif pos == "Top Right":
                    x1, y1 = w - m - length_px, m + 10
                else:
                    x1, y1 = m, m + 10
                x2, y2 = x1 + length_px, y1
                cv2.line(img, (x1, y1), (x2, y2), self.scalebar_color, thickness=5)
                cv2.putText(img, self.scalebar_label.get(),
                            (x1, y1 - 10 if "Bottom" in pos else y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.scalebar_color, 2,
                            lineType=cv2.LINE_AA)

            # ensure extension
            outname = os.path.join(out, f"overlay_{r['file']}")
            base, ext = os.path.splitext(outname)
            if ext == "":
                outname = base + ".png"
            cv2.imwrite(outname, img)

        messagebox.showinfo("Exported", f"Saved CSV and overlays to {out}")


# --------------------------
# Run application & optional config
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()

    # Optionally load config.json (safe: wrap in try/except)
    cfg_path = "config.json"
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                config = json.load(f)
            if config.get("dark_mode"):
                # use root to set palette
                try:
                    root.tk_setPalette(background="#2e2e2e", foreground="white",
                                       activeBackground="#3e3e3e", activeForeground="white")
                except Exception:
                    pass
        except Exception:
            # do not block startup on bad config
            pass

    app = NeuriteApp(root)
    root.mainloop()
