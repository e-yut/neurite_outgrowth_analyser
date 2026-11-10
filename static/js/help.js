// Neurite Outgrowth Analyser - web edition: help dialog, guided tutorial and theme toggle.
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  /* ---------- light / dark theme ---------- */

  var THEME_KEY = 'neurite-outgrowth-theme';
  var themeBtn = $('themeToggle');

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    // Show the mode the button switches to.
    themeBtn.textContent = theme === 'dark' ? '\u2600\uFE0F' : '\uD83C\uDF19';
    var next = theme === 'dark' ? 'light' : 'dark';
    themeBtn.title = 'Switch to ' + next + ' mode';
    themeBtn.setAttribute('aria-label', 'Switch to ' + next + ' mode');
  }

  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');

  themeBtn.addEventListener('click', function () {
    var theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* ignore */ }
  });

  /* ---------- help dialog ---------- */

  var dialog = $('helpDialog');

  $('btnHelp').addEventListener('click', function () {
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  });
  $('btnHelpClose').addEventListener('click', function () { dialog.close ? dialog.close() : dialog.removeAttribute('open'); });
  dialog.addEventListener('click', function (ev) {
    // Click on the backdrop closes the dialog.
    if (ev.target === dialog) dialog.close();
  });
  $('btnTourFromHelp').addEventListener('click', function () {
    if (dialog.close) dialog.close(); else dialog.removeAttribute('open');
    startTour();
  });

  /* ---------- guided tutorial ---------- */

  var STEPS = [
    { sel: '#btnDemo', title: 'Demo images',
      text: 'Loads the two bundled example images (a 0 h reference and a 120 h time point) so you can try the app without your own data.' },
    { sel: '#dropRef', title: 'Reference image (0 h)',
      text: 'Drop or pick the image taken right after the scratch. The cell-free gap is detected automatically and becomes the region of interest (ROI) used for every image. TIFF, PNG and JPG are supported.' },
    { sel: '#dropImgs', title: 'Images to analyse',
      text: 'Drop or pick the later time-point images to measure. Each one starts with a copy of the reference ROI, which you can then adjust individually.' },
    { sel: '#sliders', title: 'Detection parameters',
      text: 'The five parameters of the measurement pipeline: top-hat size (enhances thin bright structures), Otsu offset (shifts the automatic threshold), manual threshold (0 = automatic Otsu + offset), minimum component area, and minimum skeleton length. Hover each slider for details.' },
    { sel: '#ppm', title: 'Pixels per micron',
      text: 'Your image calibration. Lengths in pixels are divided by this to give microns; ROI areas use its square.' },
    { sel: '#secOverlay', title: 'Overlay appearance',
      text: 'How detected neurites are painted on the exported overlay: background mode (Color, Grayscale or B/W), neurite colour and line thickness. Purely visual — the measurements never change.' },
    { sel: '#secScalebar', title: 'Scale bar',
      text: 'Optionally draw a scale bar on the exported overlay PNGs: physical length in microns, corner, label text and colour.' },
    { sel: '#btnRun', title: 'Run analysis',
      text: 'Processes every loaded image with the current ROI and parameters and fills the results table. Re-run after changing the ROI or any detection parameter.' },
    { sel: '#btnExport', title: 'Export results',
      text: 'Downloads one ZIP containing results.csv, results.xlsx and an overlay PNG for each analysed image.' },
    { sel: '#itemSel', title: 'Image selector',
      text: 'Chooses which loaded image the viewer shows. You can also click an image in the list on the left or a row of the results table.' },
    { sel: '.tabs', title: 'Viewer tabs',
      text: 'ROI shows the editable region of interest. Threshold preview shows, live, the binary image the pipeline will measure — ideal for tuning parameters. Overlay shows the analysed result.' },
    { sel: '.zoom', title: 'Zoom',
      text: 'Zoom the viewer in and out or fit the whole image. The mouse wheel over the image zooms too.' },
    { sel: '.roi-tools', title: 'ROI tools',
      text: 'Auto-detect gap re-runs the automatic scratch detection on the reference. Use reference ROI resets the shown image to the reference ROI. Apply reference ROI to all copies it to every image. The readout shows the side lengths and area live.' },
    { sel: '.stage', title: 'Image viewer',
      text: 'In the ROI tab, drag any corner handle to reshape the quadrilateral, drag inside it to move the whole shape, and zoom with the mouse wheel.' },
    { sel: '.results', title: 'Results table',
      text: 'One row per analysed image: neurite count, total/max/mean/median lengths in px and µm, and the ROI area. Hover a column header for its definition. Rows turn grey when settings change after they were computed.' },
    { sel: '#themeToggle', title: 'Light / dark mode',
      text: 'Switches the page between a light and a dark theme. Your choice is remembered.' },
    { sel: '#btnHelp', title: 'Help',
      text: 'Opens the step-by-step guide with full explanations of every stage. That\u2019s the end of the tour \u2014 enjoy!' },
  ];

  var tour = { on: false, i: 0, hi: null, pop: null };

  function startTour() {
    if (tour.on) return;
    tour.on = true;
    tour.i = 0;

    tour.hi = document.createElement('div');
    tour.hi.id = 'tourHighlight';
    tour.pop = document.createElement('div');
    tour.pop.id = 'tourPop';
    tour.pop.setAttribute('role', 'dialog');
    document.body.appendChild(tour.hi);
    document.body.appendChild(tour.pop);

    window.addEventListener('resize', place);
    document.addEventListener('keydown', onKey, true);
    showStep(0);
  }

  function endTour() {
    if (!tour.on) return;
    tour.on = false;
    tour.hi.remove();
    tour.pop.remove();
    window.removeEventListener('resize', place);
    document.removeEventListener('keydown', onKey, true);
  }

  function onKey(ev) {
    if (ev.key === 'Escape') { ev.stopPropagation(); endTour(); }
    else if (ev.key === 'ArrowRight' || ev.key === 'Enter') { ev.preventDefault(); next(1); }
    else if (ev.key === 'ArrowLeft') { ev.preventDefault(); next(-1); }
  }

  function next(d) {
    var i = tour.i + d;
    if (i < 0) return;
    if (i >= STEPS.length) { endTour(); return; }
    showStep(i);
  }

  function target() {
    return document.querySelector(STEPS[tour.i].sel);
  }

  function showStep(i) {
    tour.i = i;
    var s = STEPS[i];
    var el = target();
    if (!el) { next(1); return; }

    tour.pop.innerHTML =
      '<h3></h3><p></p>' +
      '<div class="tour-foot">' +
      '<span class="count">' + (i + 1) + ' / ' + STEPS.length + '</span>' +
      '<button type="button" class="skip" title="End the tutorial">Skip</button>' +
      '<button type="button" class="back" title="Previous step"' + (i === 0 ? ' disabled' : '') + '>Back</button>' +
      '<button type="button" class="next" title="Next step">' + (i === STEPS.length - 1 ? 'Finish' : 'Next') + '</button>' +
      '</div>';
    tour.pop.querySelector('h3').textContent = s.title;
    tour.pop.querySelector('p').textContent = s.text;
    tour.pop.querySelector('.skip').addEventListener('click', endTour);
    tour.pop.querySelector('.back').addEventListener('click', function () { next(-1); });
    tour.pop.querySelector('.next').addEventListener('click', function () { next(1); });

    el.scrollIntoView({ block: 'center', inline: 'nearest' });
    requestAnimationFrame(place);
  }

  function place() {
    if (!tour.on) return;
    var el = target();
    if (!el) return;
    var r = el.getBoundingClientRect();
    var pad = 6;
    tour.hi.style.top = (r.top - pad) + 'px';
    tour.hi.style.left = (r.left - pad) + 'px';
    tour.hi.style.width = (r.width + 2 * pad) + 'px';
    tour.hi.style.height = (r.height + 2 * pad) + 'px';

    // Place the popover below the target when there is room, otherwise above
    // or beside it, and clamp it to the viewport.
    var pw = tour.pop.offsetWidth, ph = tour.pop.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var top, left;
    if (r.bottom + pad + ph + 8 <= vh) top = r.bottom + pad + 8;
    else if (r.top - pad - ph - 8 >= 0) top = r.top - pad - ph - 8;
    else top = Math.max(8, Math.min(vh - ph - 8, r.top));
    if (top >= r.top - pad && top <= r.bottom + pad && (r.bottom + pad + ph + 8 > vh) && (r.top - pad - ph - 8 < 0)) {
      // Popover vertically overlaps a tall target: put it beside the target.
      left = r.right + pad + 8 + pw <= vw ? r.right + pad + 8 : r.left - pad - pw - 8;
      left = Math.max(8, Math.min(vw - pw - 8, left));
    } else {
      left = Math.max(8, Math.min(vw - pw - 8, r.left));
    }
    tour.pop.style.top = top + 'px';
    tour.pop.style.left = left + 'px';
  }

  $('btnTour').addEventListener('click', startTour);
})();
