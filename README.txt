VISUALCHECK v1 - offline box checker (Python version)
=====================================================
(Easier: use VisualCheck_v1_Windows.zip - no Python needed.)
Works 100% offline. Nothing is uploaded. No internet is used when checking.

SETUP (one time)
1. Install Python 3.12 or 3.13 (64-bit) from python.org.
   During install tick "Add python.exe to PATH".
2. Double-click setup.bat (needs internet once, only to download the
   free Python parts: PyMuPDF, OpenCV, NumPy, zxing-cpp, Pillow).

USE
1. Double-click run.bat.
2. "Masters folder": master PDFs (one page per flavour) or images.
   Images of one product: put them together in one sub-folder.
3. "Photos folder": box photos (JPG, PNG, HEIC, WEBP, BMP, TIFF) or PDF.
4. Press "Run check". The report opens in your browser.
   Reports are saved in the "reports" folder.

TIPS
- Flavour names: after the first run, open flavour_names.csv in the
  masters folder with Excel, type the flavour names (e.g. WTML N1), save.
  Next run the report shows these names.
- Take photos flat, from above, with good light. Several boxes per photo is OK.
- The barcode is only checked when it is visible in the photo (back side).
- SAME / DIFFERENT for the barcode is reliable. The red "rough difference"
  marks are only a hint (light, shadows and folds also make red marks).
  Always check small text by eye.
