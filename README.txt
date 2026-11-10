Neurite Outgrowth Analysis v6.7.1
=================================

Web edition
-----------
Live: https://e-yut.github.io/neurite_outgrowth_analyser/

A browser-only edition of the analyser lives on the "e-yut-web" branch
(index.html plus the static/ folder). It needs no Python, no installation and
no server: images are decoded and analysed inside the browser and never leave
your machine. It reproduces the v6.7.1 pipeline step for step with OpenCV.js
(WebAssembly) running in a Web Worker.

How to use it: load the 0 h reference image (the scratch gap is detected
automatically and becomes the ROI), load the images to analyse, adjust the ROI
by dragging its corners or the whole shape (mouse wheel zooms), set the
parameters, press "Run analysis", then "Export ZIP" to download the CSV, XLSX
and overlay PNGs. "Load demo images" loads the two demo TIFFs from this
repository.

Additions over the desktop app: a live threshold preview that uses the real
analysis pipeline, a per-image ROI with side-length and area readout,
parameters remembered in the browser, and a single ZIP download in place of
the output-folder picker.

Parity: with the same ROI and parameters the web edition gives the same
numbers as the v6.7.1 pipeline. This was checked on both demo images against
OpenCV 5.0.0 in Python 3.13 for four parameter/ROI combinations; every column
matched to the last digit. The known quirks of v6.7.1 are kept on purpose
(for example, neurite length is the contour point count of the skeleton,
which roughly doubles the length of a one-pixel-wide line).

Differences from the desktop app: overlays are always saved as PNG; the
scale-bar label uses a browser font; JPEG and 16-bit images may decode
slightly differently in a browser than in OpenCV; for multi-page TIFFs the
largest page is used.

Third-party libraries (vendored in static/vendor, each with its licence):
OpenCV.js 4.9.0 (Apache-2.0), UTIF.js 3.1.0 (MIT), pako 1.0.11 (MIT),
SheetJS CE 0.20.3 (Apache-2.0), fflate 0.8.2 (MIT).

Creator
-------
Eylan Yutuc

Purpose
-------
Neurite Outgrowth Analysis v6.7.1 is a desktop graphical user interface (GUI)
application for analysing neurite-like structures in microscopy images. It is
designed to support semi-automated image processing, region-of-interest (ROI)
selection, neurite-length quantification, visual overlay generation, and export
of tabular results.

The application is implemented in Python and uses Tkinter for the GUI, OpenCV
for image processing, Pillow for image display, NumPy for numerical operations,
and pandas for Excel export.

This script is intended as a practical research-analysis tool. Results should
be inspected visually and validated against representative images before being
used for formal reporting, publication, or statistical analysis.

Version
-------
Version: 6.7.1
Application title: Neurite Outgrowth Analysis v6.7.1 (by Eylan Yutuc)
Script filename: NeuriteAnalysis_v.6.7.1.py

Main features
-------------
1. Load one reference image, typically a 0-hour or baseline image.
2. Load one or more images for analysis.
3. Select an output folder for exported files.
4. Adjust a reference ROI using a draggable quadrilateral.
5. Adjust ROIs independently for each analysis image.
6. View ROI side lengths and polygon area.
7. Select the overlay display mode:
   - Color
   - Grayscale
   - B/W
8. Set the image calibration as pixels per micrometre.
9. Configure overlay colour and line thickness.
10. Configure image-processing and detection parameters.
11. Preview thresholding for the first loaded image.
12. Detect neurite-like structures using enhancement, thresholding,
    connected-component filtering, and skeletonisation-like processing.
13. Calculate neurite counts and length statistics in pixels and micrometres.
14. Calculate ROI area in square pixels and square micrometres.
15. Export results as CSV and Excel files.
16. Export processed overlay images as PNG files.
17. Optionally add a calibrated scale bar to exported overlays.
18. Optionally load a config.json file at startup to enable dark mode.

Important interpretation
------------------------
The script detects image structures according to intensity, morphology, size,
and skeletonisation settings. It does not identify neurites biologically; it
identifies image objects that satisfy the configured computational criteria.
Therefore, the quality of the output depends on image quality, staining,
background, magnification, calibration, ROI placement, and parameter selection.

Always review the exported overlays alongside the numerical results.

Requirements
------------
The following software is required:

- Python 3.8 or later is recommended.
- Tkinter, normally included with standard Python installations.
- Pillow.
- OpenCV with Python bindings.
- NumPy.
- pandas.
- An Excel writer engine supported by pandas, usually openpyxl, for .xlsx export.

Python packages
---------------
Install the required Python packages with:

    python -m pip install pillow opencv-python numpy pandas openpyxl

On some Linux distributions, Tkinter must be installed separately. For Debian,
Ubuntu, or WSL-based environments, the package is commonly named:

    sudo apt install python3-tk

The exact package name may differ between operating systems.

Input image formats
-------------------
The file-selection dialogs are configured for the following image extensions:

- PNG: .png
- JPEG: .jpg
- TIFF: .tif
- TIFF: .tiff

Images are read by OpenCV. Images should be readable by OpenCV and should have
sufficient contrast between neurite-like structures and the background.

Supported channels and display behaviour
----------------------------------------
The analysis converts each loaded image to grayscale before processing. The
original colour image is retained for colour-mode overlays. Grayscale and B/W
modes create alternate bases for the exported overlay image.

The script is not designed to preserve quantitative information from arbitrary
multichannel microscopy files. If a file contains multiple channels, confirm
which channel OpenCV reads and whether that channel is appropriate for analysis.

Starting the application
------------------------
Open a terminal in the directory containing the script and run:

    python NeuriteAnalysis_v.6.7.1.py

On systems where Python 3 is invoked as python3, use:

    python3 NeuriteAnalysis_v.6.7.1.py

A Tkinter window should open. If the GUI does not start, check that Python,
Tkinter, and the required packages are installed correctly.

Basic workflow
--------------
1. Start the script.
2. Load a reference image using the "Browse" button beside "Reference (0 h
   image)". The reference image is optional for running the analysis, but it
   can be useful for establishing a consistent ROI.
3. Load the images to analyse using the "Browse" button beside "Images to
   analyse". Multiple files can be selected.
4. Choose an output folder.
5. Enter the image calibration in the "Pixels per µm" field.
6. Configure the overlay and detection settings.
7. Use "Adjust Reference ROI" if a reference image is loaded and a common ROI
   should be defined.
8. Use "Preview/Adjust ROI" to inspect or modify the ROI assigned to each image.
9. Optionally use the threshold preview to inspect segmentation behaviour.
10. Click "Run Analysis".
11. Inspect the results table.
12. Click "Export CSV & Overlays" to save the numerical results and overlay
    images.

Reference image and ROI behaviour
---------------------------------
The reference image is labelled as a "0 h image" in the interface, but the
script does not require that the image actually represents zero hours. It is a
user-defined reference image.

When images are loaded, the script attempts to derive a default rectangular ROI
from horizontal image-edge information in the reference image. If that process
is unsuccessful, it falls back to the full image. If a reference ROI has been
manually adjusted, it is assigned to the loaded analysis images.

The ROI is represented as a quadrilateral with four draggable corner handles.
The entire polygon can also be dragged. Coordinates are kept within the canvas
bounds during interaction.

The reference ROI window provides:

- A zoom control.
- A draggable quadrilateral.
- Side-length readouts for the top, right, bottom, and left edges.
- Polygon area in square pixels.

The per-image ROI window provides the same general ROI-adjustment workflow for
each loaded image. Individual ROIs can therefore be modified when fields of
view, image dimensions, or sample positioning differ between files.

ROI area calculation
--------------------
The ROI area is calculated from the four polygon vertices using the shoelace
formula. The stored values are:

- roi_area_px2: ROI area in square pixels.
- roi_area_um2: ROI area in square micrometres.

The conversion uses the configured calibration:

    area_um2 = area_px2 / (pixels_per_um ** 2)

Pixels-per-micrometre calibration
----------------------------------
The "Pixels per µm" value is used to convert lengths and areas. The script uses:

    length_um = length_px / pixels_per_um

and:

    area_um2 = area_px2 / (pixels_per_um ** 2)

The default value is 0.287 pixels per micrometre. This is only a default and
must be replaced with the correct calibration for the imaging system and image
set. Entering the inverse quantity, micrometres per pixel, will produce
incorrect results.

Use a calibration determined from the relevant microscope acquisition settings,
scale bar, objective, camera configuration, or image metadata. Use one
consistent calibration for all images in a comparison unless images were
acquired at different scales.

Overlay settings
----------------
Overlay mode controls the base image used for exported overlays:

Color
  Uses the original colour image as the base.

Grayscale
  Converts the image to grayscale and then creates a three-channel image for
  coloured overlay drawing.

B/W
  Applies Otsu thresholding to the grayscale image and uses the resulting
  binary image as the base.

The overlay colour is selected with the "Overlay Colour" button. The overlay
thickness controls the dilation applied to the detected skeleton before it is
drawn. A thicker value produces a more visible overlay but may obscure nearby
structures.

Detection parameters
--------------------
Top-hat size (px)
  Controls the rectangular structuring element used for morphological top-hat
  enhancement. The GUI permits odd values from 3 to 51, with a default of 15.
  Larger values can enhance structures at different spatial scales but may
  alter background and object separation.

Otsu offset
  Adjusts the automatically determined Otsu threshold. The value is added to
  the Otsu threshold and constrained to the valid 0-255 range. Positive values
  generally make detection more selective; negative values generally make it
  more permissive, although the effect depends on the image.

Manual thresh (0=auto)
  The displayed label indicates that 0 uses automatic thresholding. In the
  current GUI, the default is 100 and values above zero are used directly as a
  manual binary threshold. Set this value to 0 to use Otsu thresholding plus
  the Otsu offset.

Min area (px)
  Removes connected components smaller than the specified pixel area. The
  default is 150 pixels.

Min length (px)
  Removes detected contour objects whose contour length is shorter than the
  specified threshold. The default is 50 pixels.

Threshold preview
-----------------
The script contains a preview_threshold method for the first loaded analysis
image. The preview applies the same broad thresholding sequence used by the
analysis and displays the binary result in an OpenCV window.

If the interface does not expose a dedicated threshold-preview button in the
version being used, the method can be connected to a Tkinter button or invoked
programmatically from a development environment.

The preview uses the first image in the loaded image list and does not replace
analysis. It is intended to help assess whether the selected top-hat, manual
threshold, and Otsu-offset settings are reasonable.

Image-processing pipeline
--------------------------
For each image, the analysis follows these main stages:

1. Read the image using OpenCV.
2. Convert the image to grayscale.
3. Create a binary mask from the quadrilateral ROI.
4. Restrict the grayscale image to the ROI.
5. Apply histogram equalisation.
6. Invert the equalised image.
7. Apply morphological top-hat enhancement using a rectangular kernel.
8. Apply Gaussian blur with a 5 x 5 kernel.
9. Apply either the manual threshold or Otsu threshold plus offset.
10. Label connected components.
11. Remove components below the minimum-area threshold.
12. Generate a skeletonisation-like representation using repeated erosion,
    dilation, subtraction, and accumulation.
13. Find external contours in the resulting skeleton image.
14. Exclude contour objects shorter than the minimum-length threshold.
15. Calculate summary statistics from the remaining contour lengths.
16. Draw the resulting detection mask on the selected overlay base.

The skeletonisation stage is described in the source code as a
"skeletonise-like thinning via morphological approach". It is not the same as
all specialised one-pixel skeletonisation algorithms. The resulting contour
lengths should therefore be interpreted as image-analysis measurements defined
by this implementation.

Measurements generated
----------------------
For every analysed image, the script calculates:

- Neurite count: number of detected contour objects meeting the minimum-length
  criterion.
- Total neurite length in pixels.
- Total neurite length in micrometres.
- Maximum neurite length in pixels.
- Maximum neurite length in micrometres.
- Average neurite length in pixels.
- Average neurite length in micrometres.
- Median neurite length in pixels.
- Median neurite length in micrometres.
- ROI area in square pixels.
- ROI area in square micrometres.

If no objects pass the filtering criteria, the numerical summary values are set
to zero rather than causing an error.

Results table
-------------
The GUI displays the following columns:

- File
- Neurite Count
- Total Neurite Length (px)
- Total Neurite Length (µm)
- Max Neurite Length (px)
- Max Neurite Length (µm)
- Average Neurite Length (px)
- Average Neurite Length (µm)
- Median Neurite Length (px)
- Median Neurite Length (µm)

Hovering over a table heading displays a tooltip describing the corresponding
measurement.

Exported files
--------------
When export is successful, the selected output folder receives the following
file types.

neurite_results.csv
  A comma-separated file containing one row per analysed image. The exported
  fields are:

  - file
  - num
  - tot_px
  - tot_um
  - max_px
  - max_um
  - avg_px
  - avg_um
  - med_px
  - med_um
  - roi_area_px2
  - roi_area_um2

neurite_results.xlsx
  An Excel workbook containing the result records, excluding the in-memory
  overlay image arrays. Excel export depends on a working pandas Excel writer
  engine such as openpyxl. If Excel export fails, the script displays a warning
  but continues with CSV and overlay export.

overlay_<original_filename>.png
  An overlay image for each analysed file. The overlay is written as a PNG.
  If the original file has an extension, the generated name generally retains
  that extension after the overlay prefix; if no extension is present, .png is
  appended. Review filenames when working with unusual input names.

Scale bars
----------
Scale bars are added only during overlay export and only when "Add scale bar"
is enabled.

Available positions are:

- Bottom Right
- Bottom Left
- Top Right
- Top Left

The scale-bar pixel length is calculated as:

    scale_bar_length_px = scale_bar_length_um * pixels_per_um

The displayed label is user-defined and defaults to "1000 µm". The label is not
automatically recalculated if the length is changed, so update the label to
match the selected length.

The scale-bar colour is selected independently from the neurite overlay colour.

Configuration file
------------------
At startup, the script checks for a file named config.json in the current
working directory. If the file exists and contains:

    {
        "dark_mode": true
    }

the script attempts to apply a dark Tkinter palette. Invalid or unreadable
configuration files do not prevent the application from starting.

A minimal example config.json is:

    {
        "dark_mode": false
    }

The configuration file is optional. The script does not require it for normal
operation.

Recommended analysis practice
-----------------------------
1. Analyse a representative pilot image set before processing a full experiment.
2. Confirm that the pixels-per-micrometre value is correct.
3. Inspect the reference ROI and all per-image ROIs.
4. Use threshold preview or equivalent inspection to evaluate segmentation.
5. Compare detected structures against the original images.
6. Inspect exported overlays for false positives and missed neurites.
7. Keep processing parameters constant across images in a planned comparison,
   unless there is a documented reason to change them.
8. Record the script version, image set, calibration, ROI settings, and all
   detection parameters.
9. Retain the raw images and exported overlays together with the CSV/XLSX files.
10. Apply the same biological inclusion and exclusion rules independently of
    the software output.

Reproducibility checklist
-------------------------
For each analysis, record:

- Script version.
- Python version.
- Operating system.
- Package versions.
- Microscope and objective.
- Image acquisition settings.
- Image file format and bit depth.
- Pixels-per-micrometre calibration.
- Reference image identity.
- ROI coordinates or ROI-selection procedure.
- Overlay mode and colour.
- Overlay thickness.
- Top-hat size.
- Otsu offset.
- Manual threshold.
- Minimum connected-component area.
- Minimum contour length.
- Whether scale bars were included.
- Output filenames and analysis date.

Potential limitations
---------------------
- The application assumes that the selected image-processing settings are
  suitable for the image set.
- Thresholding may be sensitive to uneven illumination, background variation,
  saturation, noise, staining intensity, and image compression.
- Connected-component filtering can remove genuine short or faint structures.
- Large artefacts may be retained if they satisfy the area criterion.
- The computed contour length is an image-derived measurement and may not equal
  the true biological neurite length.
- Branches, crossings, merged structures, and fragmented neurites may affect
  the number and length measurements.
- The scale bar is a visual annotation and does not independently verify image
  calibration.
- Different image dimensions or acquisition scales require careful ROI and
  calibration review.
- Excel export requires an appropriate writer engine.
- The OpenCV threshold preview opens a separate window and may require user
  interaction to close it.
- Tkinter and OpenCV GUI behaviour can vary between operating systems and
  headless environments.
- The code should be tested on the target system before deployment to a wider
  research group.

Troubleshooting
---------------
The GUI does not start
  Confirm that Python and Tkinter are installed. Then test the dependencies:

      python -c "import tkinter, cv2, numpy, pandas; from PIL import Image"

A package import fails
  Install or update the missing package with pip. For example:

      python -m pip install --upgrade pillow opencv-python numpy pandas openpyxl

No images are available in the dialog
  Confirm that the files use one of the supported extensions and that the files
  are readable. Copying the files to a local working directory can help rule
  out network or permission problems.

The analysis reports no neurites
  Inspect the ROI, lower the manual threshold or set it to 0 for automatic
  thresholding, reduce the minimum area or minimum length, and review the
  top-hat size. Confirm that the image contains sufficient contrast.

Too many false positives are detected
  Increase the manual threshold, adjust the Otsu offset, increase the minimum
  area or minimum length, revise the ROI, and inspect the source image for
  background artefacts.

Measurements are in the wrong physical units
  Verify that the entered value is pixels per micrometre, not micrometres per
  pixel. Confirm that the calibration applies to every image.

The overlay is difficult to see
  Change the overlay mode or colour and increase the overlay thickness. Avoid
  using a colour that is similar to the underlying image.

Excel export fails
  Install an Excel writer such as openpyxl:

      python -m pip install openpyxl

  CSV and overlay export may still succeed even if the Excel file cannot be
  written.

The scale bar appears incorrect
  Check the pixels-per-micrometre value, scale-bar length, image dimensions,
  position, and label. Remember that the label is entered manually.

Source-code structure
--------------------
The script is organised into the following main components:

- Imports and dependency setup.
- ToolTip: hover tooltips for GUI elements.
- DraggablePolygon: interactive quadrilateral ROI drawing and movement.
- NeuriteApp: primary application state, interface, analysis, and export logic.
- GUI helper methods for colour selection and file loading.
- ROI preview and measurement methods.
- Threshold preview method.
- Analysis method.
- CSV, Excel, and overlay export method.
- Main application startup and optional configuration loading.

Creator and attribution
-----------------------
Created by Eylan Yutuc.

If this script is adapted, redistributed, or incorporated into another
workflow, retain creator attribution where appropriate and document any
changes that could affect image processing, measurements, exports, or
interpretation.

Suggested citation wording
--------------------------
Yutuc, E. Neurite Outgrowth Analysis v6.7.1. Python graphical image-analysis
software for neurite-like structure quantification. Version 6.7.1.

License
-------
No explicit licence is declared in the supplied script. Before redistribution,
modification, or incorporation into a larger software package, add an
appropriate licence and clarify the permitted use, modification, and
redistribution terms.

Disclaimer
----------
This software is provided for research and educational use. The creator makes
no guarantee that the generated measurements are suitable for a particular
experimental, clinical, regulatory, diagnostic, or publication purpose.
Users are responsible for validating the workflow, calibration, segmentation,
quality-control procedures, and interpretation of results for their own data.
