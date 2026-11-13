# -*- coding: utf-8 -*-
"""
Visual Check - offline box vs master artwork checker.

Everything runs on this computer. Nothing is uploaded, no internet is used.
  * masters folder : master PDF files (one PDF per product, one page per flavour)
  * photos folder  : phone photos of printed boxes (JPG / PNG)
  * result         : an HTML report in the "reports" folder, opened in your browser
"""
import os
import sys
import csv
import json
import html
import time
import hashlib
import pathlib
import threading
import traceback
import webbrowser
from datetime import datetime

import numpy as np
import cv2
import zxingcpp

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

try:
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 400_000_000
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        HEIF_OK = True
    except Exception:
        HEIF_OK = False
except ImportError:
    Image = None
    HEIF_OK = False


def _data_dir():
    """Folder for settings, cache and reports (next to the app if possible)."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    try:
        test = os.path.join(base, ".write_test")
        with open(test, "w") as f:
            f.write("x")
        os.remove(test)
        return base
    except Exception:
        alt = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "VisualCheck")
        os.makedirs(alt, exist_ok=True)
        return alt


APP_DIR = _data_dir()
APP_NAME = "VisualCheck v1"
RES_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
ICON_FILE = os.path.join(RES_DIR, "VisualCheck.ico")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
CACHE_FILE = os.path.join(APP_DIR, "cache", "barcodes.json")
NAMES_FILE = "flavour_names.csv"          # created inside the masters folder

VIS_DPI = 150            # master render size used for picture matching
PHOTO_MAX = 2400         # photos are shrunk to this long side for matching
BARCODE_DPIS = (200, 300, 400, 600)
MAX_RENDER_SIDE = 9000
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif")
PHOTO_PDF_DPI = 200      # a PDF in the photos folder: each page is one photo
MASTER_IMG_SIDE = 1800   # an image in the masters folder is scaled to this long side
EAN = zxingcpp.BarcodeFormat.EAN13

cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))


# ----------------------------------------------------------------- helpers
def log_print(msg):
    try:
        if sys.stdout is not None:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(str(msg).encode(enc, "replace").decode(enc, "replace"), flush=True)
    except Exception:
        pass


def imread_any(path):
    """cv2.imread that also works with non-English Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def load_image(path):
    """Read any common image (also HEIC), turned the right way up (EXIF)."""
    if Image is not None:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return imread_any(path)


def is_img(fn):
    return fn.lower().endswith(IMG_EXT)


def is_pdf(fn):
    return fn.lower().endswith(".pdf")


def imwrite_any(path, img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)


def ean_ok(code):
    if len(code) != 13 or not code.isdigit():
        return False
    d = [int(c) for c in code]
    s = sum(d[i] * (3 if i % 2 else 1) for i in range(12))
    return (10 - s % 10) % 10 == d[12]


def fmt_ean(code):
    if code and len(code) == 13:
        return "%s %s %s" % (code[0], code[1:7], code[7:])
    return code or "-"


def quad_of(res):
    p = res.position
    return np.float32([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                       [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]])


def read_codes(gray, fixed=False):
    kw = dict(formats=EAN, try_rotate=True, try_downscale=True)
    out = []
    try:
        res = zxingcpp.read_barcodes(gray, **kw)
        if fixed:
            res = list(res) + list(zxingcpp.read_barcodes(
                gray, binarizer=zxingcpp.Binarizer.FixedThreshold, **kw))
    except Exception:
        return out
    for r in res:
        if r.valid and ean_ok(r.text):
            out.append((r.text, quad_of(r)))
    return out


def render(page, dpi, gray=False):
    side = max(page.rect.width, page.rect.height) * dpi / 72.0
    if side > MAX_RENDER_SIDE:
        dpi = dpi * MAX_RENDER_SIDE / side
    z = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY if gray else fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    if gray:
        return img[:, :, 0].copy(), dpi
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dpi


def file_key(path):
    st = os.stat(path)
    return hashlib.sha1(("%s|%d|%d" % (os.path.basename(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


# ----------------------------------------------------------------- masters
class Line:
    """One master PDF = one product line."""
    pass


def scan_pdf_barcodes(path, log):
    """Return list of {page, barcode, quad(at VIS_DPI)} for every page."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        found = None
        for dpi in BARCODE_DPIS:
            g, real_dpi = render(page, dpi, gray=True)
            codes = read_codes(g)
            if codes:
                text, q = codes[0]
                found = {"barcode": text, "quad": (q * (VIS_DPI / real_dpi)).tolist()}
                break
        pages.append({"page": i + 1, **(found or {"barcode": None, "quad": None})})
    doc.close()
    return pages


def load_names(masters_dir):
    names = {}
    p = os.path.join(masters_dir, NAMES_FILE)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    k = (row.get("pdf_file", "").strip(), str(row.get("page", "")).strip())
                    if row.get("flavour_name", "").strip():
                        names[k] = row["flavour_name"].strip()
        except Exception:
            pass
    return names


def save_names(masters_dir, lines, names):
    p = os.path.join(masters_dir, NAMES_FILE)
    try:
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pdf_file", "page", "barcode", "flavour_name"])
            for ln in lines:
                for it in ln.items:
                    w.writerow([ln.file, it["page"], it["barcode"] or "", names.get((ln.file, str(it["page"])), "")])
    except PermissionError:
        pass  # file open in Excel - fine, just skip


def master_image_item(path, cache, page_no):
    """An image file used as one master page."""
    img = load_image(path)
    if img is None:
        return None
    key = "img:" + file_key(path)
    sc = MASTER_IMG_SIDE / float(max(img.shape[:2]))
    if key in cache:
        bc = cache[key]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        groups = photo_barcodes(gray)
        bc = {"barcode": None, "quad": None}
        if groups:
            g = max(groups, key=lambda g: len(g["votes"]))
            bc = {"barcode": max(set(g["votes"]), key=g["votes"].count), "quad": g["q"].tolist()}
        cache[key] = bc
        save_json(CACHE_FILE, cache)
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    q = np.float32(bc["quad"]) * sc if bc["quad"] is not None else None
    return {"page": page_no, "barcode": bc["barcode"], "quad": q, "img": small}


def image_lines(masters_dir, cache, names, log):
    """Loose images: one line each. A sub-folder of images: one line (one image per flavour)."""
    groups = []
    for fn in sorted(os.listdir(masters_dir)):
        p = os.path.join(masters_dir, fn)
        if os.path.isfile(p) and is_img(fn):
            groups.append((fn, os.path.splitext(fn)[0], [p]))
        elif os.path.isdir(p) and not fn.startswith("."):
            imgs = [os.path.join(p, f) for f in sorted(os.listdir(p)) if is_img(f)]
            if imgs:
                groups.append((fn + "/", fn, imgs))
    out = []
    for key, title, paths in groups:
        ln = Line()
        ln.file, ln.title, ln.items, ln.skipped = key, title, [], []
        for i, p in enumerate(paths, 1):
            try:
                it = master_image_item(p, cache, i)
            except Exception as e:
                log("  !! cannot read master image %s (%s)" % (os.path.basename(p), e))
                continue
            if it is None:
                log("  !! cannot read master image %s" % os.path.basename(p))
                continue
            it["src"] = os.path.basename(p)
            it["name"] = names.get((key, str(i)), "") or (os.path.splitext(os.path.basename(p))[0] if len(paths) > 1 else "")
            ln.items.append(it)
        if ln.items:
            out.append(ln)
    return out


def index_masters(masters_dir, log=log_print):
    cache = load_json(CACHE_FILE, {})
    names = load_names(masters_dir)
    pdfs = sorted(f for f in os.listdir(masters_dir) if is_pdf(f))
    sift = cv2.SIFT_create(nfeatures=20000)
    lines = []
    pending = []
    for fn in pdfs:
        path = os.path.join(masters_dir, fn)
        key = file_key(path)
        if key in cache:
            pages = cache[key]
        else:
            log("  reading barcodes in %s ..." % fn)
            try:
                pages = scan_pdf_barcodes(path, log)
            except Exception as e:
                log("  !! cannot open %s (%s)" % (fn, e))
                continue
            cache[key] = pages
            save_json(CACHE_FILE, cache)
        doc = fitz.open(path)
        title = (doc.metadata or {}).get("title") or os.path.splitext(fn)[0]
        ln = Line()
        ln.file, ln.title, ln.items, ln.skipped = fn, title.strip(), [], []
        for p in pages:
            if not p["barcode"]:
                ln.skipped.append(p["page"])
                continue
            img, _ = render(doc[p["page"] - 1], VIS_DPI)
            ln.items.append({"page": p["page"], "barcode": p["barcode"],
                             "quad": np.float32(p["quad"]), "img": img,
                             "name": names.get((fn, str(p["page"])), "")})
        doc.close()
        if not ln.items:
            log("  !! %s: no barcode found on any page, skipped" % fn)
            continue
        pending.append(ln)
    pending += image_lines(masters_dir, cache, names, log)
    for ln in pending:
        h, w = ln.items[0]["img"].shape[:2]
        for it in ln.items:
            if it["img"].shape[:2] != (h, w):
                ih, iw = it["img"].shape[:2]
                if it.get("quad") is not None:
                    it["quad"] = it["quad"] * np.float32([w / float(iw), h / float(ih)])
                it["img"] = cv2.resize(it["img"], (w, h), interpolation=cv2.INTER_AREA)
        ln.size = (w, h)
        rep = cv2.cvtColor(ln.items[0]["img"], cv2.COLOR_BGR2GRAY)
        ln.kp, ln.desc = sift.detectAndCompute(rep, None)
        ln.kp_xy = np.float32([k.pt for k in ln.kp])
        ln.blur = [cv2.GaussianBlur(it["img"], (0, 0), 1.5) for it in ln.items]
        ln.lab = [cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32) for b in ln.blur]
        ln.mean = np.mean(np.stack([b.astype(np.float32) for b in ln.blur]), axis=0)
        if len(ln.items) > 1:
            std = np.std(np.stack(ln.lab), axis=0).max(axis=2)
            disc = (std > 20).astype(np.uint8)
        else:
            disc = np.zeros((h, w), np.uint8)
        # barcode area is compared separately (by the numbers), so leave it out
        ln.bc_mask = np.zeros((h, w), np.uint8)
        for it in ln.items:
            if it.get("quad") is None:
                continue
            x, y, bw, bh = cv2.boundingRect(it["quad"].astype(np.int32))
            m = int(0.35 * max(bw, bh))
            ln.bc_mask[max(0, y - m):y + bh + m, max(0, x - m):x + bw + m] = 1
        disc[ln.bc_mask > 0] = 0
        ln.disc = disc
        lines.append(ln)
        log("  %s: %d master page(s)%s" % (ln.title, len(ln.items),
            ("  (skipped page %s - no barcode / void page)" % ", ".join(map(str, ln.skipped))) if ln.skipped else ""))
    save_names(masters_dir, lines, names)
    return lines


def label(ln, it):
    if it.get("name"):
        return "%s - %s" % (ln.title, it["name"])
    if len(ln.items) > 1 or is_pdf(ln.file):
        return "%s - page %d" % (ln.title, it["page"])
    return ln.title


# ----------------------------------------------------------------- photos
def photo_barcodes(gray_full):
    """Read all EAN-13 in a photo, a few sizes, then vote per position."""
    reads = []
    for s in (1.0, 0.75, 0.5):
        g = gray_full if s == 1.0 else cv2.resize(gray_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for text, q in read_codes(g, fixed=(s == 0.5)):
            reads.append((text, q / s))
    groups = []
    for text, q in reads:
        c = q.mean(axis=0)
        size = np.linalg.norm(q[0] - q[2]) + 50
        for g in groups:
            if np.linalg.norm(g["c"] - c) < max(size, g["size"]) * 0.6:
                g["votes"].append(text)
                break
        else:
            groups.append({"c": c, "q": q, "size": size, "votes": [text]})
    return groups


def pick_code(votes, known, prefixes):
    def score(t):
        return (t in known, votes.count(t), prefixes.get(t[:7], 0))
    return max(set(votes), key=score)


def targeted_read(gray_full, quad_full):
    """Try again just around the place where the master has its barcode."""
    x, y, w, h = cv2.boundingRect(quad_full.astype(np.int32))
    m = int(0.6 * max(w, h))
    H, W = gray_full.shape
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    crop = gray_full[y0:y1, x0:x1]
    for z in (1.0, 1.5, 2.0, 0.7):
        c = crop if z == 1.0 else cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        codes = read_codes(c, fixed=True)
        if codes:
            return codes[0][0]
    return None


def find_instances(ln, pk_xy, pdesc, flann):
    """Find every copy of this product line in the photo (several boxes)."""
    if ln.desc is None or len(ln.kp) < 10 or pdesc is None:
        return []
    m = flann.knnMatch(pdesc, ln.desc, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 20:
        return []
    P = np.float32([pk_xy[a.queryIdx] for a in good])
    M = np.float32([ln.kp_xy[a.trainIdx] for a in good])
    alive = np.ones(len(good), bool)
    out, best = [], None
    for _ in range(15):
        idx = np.where(alive)[0]
        if len(idx) < 20:
            break
        H, inl = cv2.findHomography(M[idx], P[idx], cv2.RANSAC, 8.0, maxIters=5000)
        if H is None:
            break
        inl = inl.ravel() == 1
        if inl.sum() >= 12:   # polish the fit with the good matches only
            H2, _ = cv2.findHomography(M[idx[inl]], P[idx[inl]], cv2.LMEDS)
            if H2 is not None:
                pr = cv2.perspectiveTransform(M[idx].reshape(-1, 1, 2), H2).reshape(-1, 2)
                inl2 = np.linalg.norm(pr - P[idx], axis=1) < 6.0
                if inl2.sum() >= 0.9 * inl.sum():
                    H, inl = H2, inl2
        n = int(inl.sum())
        best = best or n
        if n < max(40, 0.15 * best):
            break
        ip, im = P[idx[inl]], M[idx[inl]]
        hull = cv2.convexHull(ip)
        for j in idx:
            if cv2.pointPolygonTest(hull, (float(P[j][0]), float(P[j][1])), True) > -20:
                alive[j] = False
        alive[idx[inl]] = False
        sx = np.linalg.svd(H[:2, :2], compute_uv=False)
        if sx.min() < 1e-3 or sx.max() / sx.min() > 4 or cv2.contourArea(hull) < 2000:
            continue
        out.append({"line": ln, "H": H, "n": n, "hull": hull, "mhull": cv2.convexHull(im)})
    return out


def poly_iou(a, b, shape):
    s = 600.0 / max(shape)
    ma = np.zeros((int(shape[0] * s) + 1, int(shape[1] * s) + 1), np.uint8)
    mb = ma.copy()
    cv2.fillConvexPoly(ma, (a.reshape(-1, 2) * s).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, (b.reshape(-1, 2) * s).astype(np.int32), 1)
    inter = np.logical_and(ma, mb).sum()
    return inter / max(1, min(ma.sum(), mb.sum()))


def sort_reading_order(insts):
    """Number boxes row by row, left to right."""
    if not insts:
        return insts
    info = []
    for c in insts:
        p = c["hull"].reshape(-1, 2)
        info.append((p[:, 1].mean(), p[:, 0].mean(), p[:, 1].max() - p[:, 1].min(), c))
    info.sort(key=lambda t: t[0])
    med_h = float(np.median([t[2] for t in info]))
    rows, cur = [], [info[0]]
    for t in info[1:]:
        if t[0] - cur[0][0] > 0.5 * med_h:
            rows.append(cur)
            cur = [t]
        else:
            cur.append(t)
    rows.append(cur)
    out = []
    for r in rows:
        out += [t[3] for t in sorted(r, key=lambda t: t[1])]
    return out


def local_scores(lab, msk, masters, shift=4):
    """Compare each area that differs between flavours, allowing a small local
    shift (boxes are never perfectly flat), and add up the best match."""
    m8 = cv2.dilate(msk.astype(np.uint8), np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8)
    H, W = msk.shape
    tot = np.zeros(len(masters))
    wsum = 0.0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 30:
            continue
        x0, y0, x1, y1 = max(0, x - shift), max(0, y - shift), min(W, x + w + shift), min(H, y + h + shift)
        if x1 - x0 <= 2 * shift or y1 - y0 <= 2 * shift:
            continue
        srch = lab[y0:y1, x0:x1]
        tx0, ty0, tx1, ty1 = x0 + shift, y0 + shift, x1 - shift, y1 - shift
        if tx1 - tx0 < 3 or ty1 - ty0 < 3:
            continue
        npx = float((tx1 - tx0) * (ty1 - ty0))
        for j, L in enumerate(masters):
            t = np.ascontiguousarray(L[ty0:ty1, tx0:tx1])
            r = cv2.matchTemplate(srch, t, cv2.TM_SQDIFF)
            tot[j] += np.sqrt(r.min() / npx) * area
        wsum += area
    if wsum == 0:
        return [float(np.mean(np.abs(lab[msk] - L[msk]).sum(axis=1))) for L in masters]
    return list(tot / wsum)


def classify(inst, small):
    ln, H = inst["line"], inst["H"]
    w, h = ln.size
    Hi = np.linalg.inv(H)
    warped = cv2.warpPerspective(small, Hi, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(small.shape[:2], 255, np.uint8), Hi, (w, h)) > 0
    seen = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(seen, inst["mhull"].reshape(-1, 2).astype(np.int32), 1)
    seen = (seen > 0) & valid
    wb = cv2.GaussianBlur(warped, (0, 0), 1.5).astype(np.float32)
    common = seen & (ln.disc == 0) & (ln.bc_mask == 0)
    gains = []
    if common.sum() > 500:
        sel = np.where(common.ravel())[0][::5]
        for c in range(3):
            x = wb[..., c].ravel()[sel]
            y = ln.mean[..., c].ravel()[sel]
            a, b = np.polyfit(x, y, 1)
            wb[..., c] = a * wb[..., c] + b
            gains.append((a, b))
    norm = np.clip(wb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    msk = seen & (ln.disc > 0)
    npx = int(msk.sum())
    if len(ln.items) == 1:
        scores = [0.0]
    elif npx == 0:
        scores = [999.0] * len(ln.items)
    else:
        scores = local_scores(lab, msk, ln.lab)
    order = list(np.argsort(scores))
    inst.update(scores=scores, order=order, npx=npx, seen=seen, gains=gains)
    if len(order) > 1:
        b1, b2 = scores[order[0]], scores[order[1]]
        margin = (b2 - b1) / max(b2, 1e-6)
        inst["sure"] = npx >= 60 and margin >= 0.12
    else:
        inst["sure"] = True
    inst["margin"] = margin if len(order) > 1 else 1.0
    return inst


def make_thumbs(inst, full, s, idx, out_dir, tag):
    """Aligned photo crop | master crop | rough difference."""
    ln = inst["line"]
    it = ln.items[idx]
    w, h = ln.size
    Hfull = np.diag([1.0 / s, 1.0 / s, 1.0]) @ inst["H"]
    warped = cv2.warpPerspective(full, np.linalg.inv(Hfull), (w, h), flags=cv2.INTER_AREA)
    if inst.get("gains"):
        wf = warped.astype(np.float32)
        for c, (a, b) in enumerate(inst["gains"]):
            wf[..., c] = a * wf[..., c] + b
        warped = np.clip(wf, 0, 255).astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(inst["mhull"].reshape(-1, 2).astype(np.int32))
    pad = 15
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    mas = it["img"]
    # rough difference: blur, colour distance, ignore thin edges (small misalignment)
    seen = cv2.erode(inst["seen"].astype(np.uint8), np.ones((25, 25), np.uint8))
    a = cv2.cvtColor(cv2.GaussianBlur(warped, (0, 0), 2.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(mas, (0, 0), 2.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.sqrt(((a - b) ** 2).sum(axis=2))
    bad = ((d > 45) & (seen > 0)).astype(np.uint8)
    bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bad)
    keep = np.zeros_like(bad)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= 60:
            keep[lab == k] = 1
    over = warped.copy()
    over[keep > 0] = (0.35 * over[keep > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(cv2.dilate(keep, np.ones((9, 9), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (0, 0, 255), 2)
    names = []
    for nm, im in (("photo", warped), ("master", mas), ("diff", over)):
        crop = im[y0:y1, x0:x1]
        if crop.size == 0:
            crop = im
        sc = min(1.0, 700.0 / max(crop.shape[:2]))
        if sc < 1:
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        fn = "%s_%s.jpg" % (tag, nm)
        imwrite_any(os.path.join(out_dir, fn), crop)
        names.append(fn)
    seen_area = max(1, int(seen.sum()))
    return names, 100.0 * keep.sum() / seen_area


def check_photo(full, name, lines, all_codes, out_dir, pno, log):
    cv2.setRNGSeed(12345)
    if full is None:
        log("  !! cannot read photo %s" % name)
        return None
    gray_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    s = min(1.0, PHOTO_MAX / float(max(full.shape[:2])))
    small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full.copy()
    gsmall = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1) barcodes
    groups = photo_barcodes(gray_full)
    prefixes = {}
    for g in groups:
        for t in g["votes"]:
            prefixes[t[:7]] = prefixes.get(t[:7], 0) + 1
    codes = []
    for g in groups:
        codes.append({"text": pick_code(g["votes"], all_codes, prefixes), "c_small": g["c"] * s, "used": False})

    # 2) find boxes by picture
    sift = cv2.SIFT_create(nfeatures=40000)
    pk, pdesc = sift.detectAndCompute(gsmall, None)
    pk_xy = np.float32([k.pt for k in pk]) if pk else np.zeros((0, 2), np.float32)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    cands = []
    for ln in lines:
        cands += find_instances(ln, pk_xy, pdesc, flann)
    cands.sort(key=lambda c: -c["n"])
    insts = []
    for c in cands:
        if all(poly_iou(c["hull"], k["hull"], small.shape[:2]) < 0.3 for k in insts):
            insts.append(c)
    insts = sort_reading_order(insts)

    # where each box should have its barcode (from the master)
    for inst in insts:
        classify(inst, small)
        ln = inst["line"]
        mq = ln.items[inst["order"][0]].get("quad")
        inst["has_q"] = mq is not None
        if mq is None:
            mq = np.float32([[0, 0], [ln.size[0], 0], [ln.size[0], ln.size[1]], [0, ln.size[1]]])
        inst["exp"] = cv2.perspectiveTransform(mq.reshape(-1, 1, 2), inst["H"]).reshape(-1, 2)
        inst["expc"] = inst["exp"].mean(axis=0)
        inst["code"] = None
    # give each barcode to the nearest box that expects a barcode there
    pairs = []
    for ii, inst in enumerate(insts):
        bsize = max(20.0, float(np.linalg.norm(inst["exp"][0] - inst["exp"][2])))
        for ci, c in enumerate(codes):
            dd = float(np.linalg.norm(c["c_small"] - inst["expc"]))
            inside = cv2.pointPolygonTest(inst["hull"], (float(c["c_small"][0]), float(c["c_small"][1])), True) > -30
            if (inst["has_q"] and dd < 1.5 * bsize) or inside:
                pairs.append((dd if inst["has_q"] else dd + 1e6, ii, ci))
    pairs.sort()
    for dd, ii, ci in pairs:
        if insts[ii]["code"] is None and not codes[ci]["used"]:
            insts[ii]["code"] = codes[ci]["text"]
            codes[ci]["used"] = True

    boxes = []
    for bi, inst in enumerate(insts):
        ln = inst["line"]
        vis = inst["order"][0]
        exp, expc = inst["exp"], inst["expc"]
        box_code = inst["code"]
        if box_code is None:
            inside = inst["has_q"] and cv2.pointPolygonTest(inst["hull"], (float(expc[0]), float(expc[1])), True) > -10
            if inside:
                box_code = targeted_read(gray_full, exp / s)
        boxes.append(decide(inst, box_code, lines))
        tag = "p%d_b%d" % (pno, bi + 1)
        idx = boxes[-1]["item_idx"] if boxes[-1]["line"] is ln else vis
        try:
            boxes[-1]["thumbs"], boxes[-1]["diff_pct"] = make_thumbs(inst, full, s, idx, out_dir, tag)
        except Exception:
            boxes[-1]["thumbs"], boxes[-1]["diff_pct"] = None, None
        boxes[-1]["hull"] = inst["hull"]

    # barcodes that are not on any recognised box
    for c in codes:
        if not c["used"]:
            boxes.append(decide(None, c["text"], lines, where=c["c_small"]))

    # overview picture with numbers
    ov = small.copy()
    for i, b in enumerate(boxes):
        col = {"SAME": (60, 170, 60), "DIFFERENT": (40, 40, 230)}.get(b["status"], (0, 160, 230))
        if b.get("hull") is not None:
            cv2.polylines(ov, [b["hull"].astype(np.int32)], True, col, 5)
            cxy = b["hull"].reshape(-1, 2).mean(axis=0)
        else:
            cxy = b["where"]
            cv2.circle(ov, tuple(int(v) for v in cxy), 40, col, 5)
        b["no"] = i + 1
        cv2.putText(ov, str(i + 1), (int(cxy[0]) - 25, int(cxy[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 14)
        cv2.putText(ov, str(i + 1), (int(cxy[0]) - 25, int(cxy[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 2.6, col, 6)
    sc = 1200.0 / max(ov.shape[:2])
    ov = cv2.resize(ov, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ovn = "p%d_overview.jpg" % pno
    imwrite_any(os.path.join(out_dir, ovn), ov)
    return {"file": name, "overview": ovn, "boxes": boxes}


def find_code(code, lines):
    if not code:
        return None, None
    for ln in lines:
        for i, it in enumerate(ln.items):
            if it["barcode"] == code:
                return ln, i
    return None, None


def decide(inst, box_code, lines, where=None):
    b = {"box_code": box_code, "where": where, "notes": []}
    code_ln, code_i = find_code(box_code, lines) if box_code else (None, None)
    if inst is None:
        # barcode only, no picture match
        b["hull"] = None
        if code_ln:
            b.update(line=code_ln, item_idx=code_i, by="barcode only",
                     status="SAME", master_code=box_code)
            b["notes"].append("Barcode is in the master. Box picture not matched (maybe only part of the box is in the photo).")
        else:
            b.update(line=None, item_idx=None, by="-", status="NOT FOUND", master_code=None)
            b["notes"].append("This barcode is not in any master, and the box picture was not recognised.")
        return b
    ln = inst["line"]
    vis = inst["order"][0]
    b["sure"] = inst["sure"]
    b["candidates"] = [label(ln, ln.items[k]) for k in inst["order"][:3]]
    if code_ln is not None:
        b.update(line=code_ln, item_idx=code_i, by="barcode", master_code=box_code)
        if code_ln is ln and code_i != vis and inst["sure"]:
            b["status"] = "DIFFERENT"
            b["notes"].append("Barcode belongs to %s, but the design looks like %s." %
                              (label(code_ln, code_ln.items[code_i]), label(ln, ln.items[vis])))
        elif code_ln is not ln:
            b["status"] = "DIFFERENT"
            b["notes"].append("Barcode belongs to %s, but the box looks like %s." %
                              (label(code_ln, code_ln.items[code_i]), label(ln, ln.items[vis])))
        else:
            b["status"] = "SAME"
        return b
    master_code = ln.items[vis]["barcode"]
    b.update(line=ln, item_idx=vis, by="picture", master_code=master_code)
    if box_code and not master_code:
        b["status"] = "SAME DESIGN"
        b["notes"].append("The master has no readable barcode, so the box barcode %s could not be compared." % fmt_ean(box_code))
    elif box_code:
        b["status"] = "DIFFERENT"
        b["notes"].append("Barcode on box %s is NOT the master barcode %s." % (fmt_ean(box_code), fmt_ean(master_code)))
    else:
        b["status"] = "SAME DESIGN"
        b["notes"].append("No barcode visible in this photo (front side?) - barcode not checked.")
    if not inst["sure"]:
        b["status"] = "NOT SURE" if b["status"] != "DIFFERENT" else "DIFFERENT"
        b["notes"].append("Picture match is not sure. Closest: %s." % " / ".join(b["candidates"][:2]))
    return b


# ----------------------------------------------------------------- report
CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#f4f4f4;color:#222}
h1{margin:0 0 6px} .small{color:#666;font-size:13px}
.sum{background:#fff;padding:12px 16px;border-radius:8px;margin:12px 0}
.photo{background:#fff;padding:14px;border-radius:8px;margin:18px 0}
.ov{max-width:100%;max-height:520px;border:1px solid #ccc}
.box{border-top:1px solid #ddd;padding:10px 0;display:flex;gap:16px;flex-wrap:wrap}
.info{min-width:300px;max-width:420px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;color:#fff;font-weight:bold}
.SAME{background:#2e9e44}.DIFFERENT{background:#d62828}.NOT{background:#e08b00}.DESIGN{background:#2e7bbf}
.thumbs{display:flex;gap:8px;flex-wrap:wrap}
.thumbs div{text-align:center;font-size:12px;color:#555}
.thumbs img{max-height:360px;max-width:300px;border:1px solid #ccc;background:#fff}
table{border-collapse:collapse} td{padding:2px 8px 2px 0;vertical-align:top}
.bad{color:#d62828;font-weight:bold}
"""


def badge(status):
    cls = {"SAME": "SAME", "DIFFERENT": "DIFFERENT", "SAME DESIGN": "DESIGN"}.get(status, "NOT")
    return '<span class="badge %s">%s</span>' % (cls, html.escape(status))


def write_report(out_dir, results, lines, masters_dir, photos_dir, secs):
    e = html.escape
    allb = [b for r in results for b in r["boxes"]]
    cnt = {}
    for b in allb:
        cnt[b["status"]] = cnt.get(b["status"], 0) + 1
    h = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>VisualCheck v1 report</title><style>%s</style></head><body>' % CSS]
    h.append("<h1>VisualCheck v1 report</h1>")
    h.append('<div class="small">%s &nbsp;|&nbsp; masters: %s &nbsp;|&nbsp; photos: %s &nbsp;|&nbsp; %.0f s &nbsp;|&nbsp; made offline on this computer</div>'
             % (datetime.now().strftime("%Y-%m-%d %H:%M"), e(masters_dir), e(photos_dir), secs))
    h.append('<div class="sum"><b>%d photos, %d boxes found.</b> &nbsp; ' % (len(results), len(allb)))
    h.append(" &nbsp; ".join("%s %d" % (badge(k), v) for k, v in sorted(cnt.items())))
    h.append('<div class="small" style="margin-top:8px">SAME = barcode and design match the master. '
             'SAME DESIGN = design matches, barcode not in the photo. DIFFERENT = barcode (or design) does not match. '
             'NOT SURE = check by eye. The red marks in "rough difference" are only a hint: light, shadows and folds also make red marks.</div>')
    h.append('<div class="small" style="margin-top:8px"><b>Masters used:</b><br>')
    for ln in lines:
        h.append("%s (%s): %d flavours%s<br>" % (e(ln.title), e(ln.file), len(ln.items),
                 (", skipped page %s" % ", ".join(map(str, ln.skipped))) if ln.skipped else ""))
    h.append("</div></div>")
    for r in results:
        h.append('<div class="photo"><h2>%s</h2><img class="ov" src="%s">' % (e(r["file"]), e(r["overview"])))
        if not r["boxes"]:
            h.append("<p>No box recognised in this photo.</p>")
        for b in r["boxes"]:
            ln, i = b["line"], b["item_idx"]
            item = label(ln, ln.items[i]) if ln is not None else "unknown"
            page = ("%s, page %d" % (ln.file, ln.items[i]["page"])) if ln is not None else "-"
            if ln is not None and ln.items[i].get("src"):
                page = "%s%s" % (ln.file if ln.file.endswith("/") else "", ln.items[i]["src"])
            same_code = b["box_code"] and b["master_code"] and b["box_code"] == b["master_code"]
            codecls = "" if (same_code or not b["box_code"]) else ' class="bad"'
            h.append('<div class="box"><div class="info"><h3>Box %d &nbsp;%s</h3><table>' % (b["no"], badge(b["status"])))
            h.append("<tr><td>Item</td><td><b>%s</b></td></tr>" % e(item))
            h.append("<tr><td>Master</td><td>%s</td></tr>" % e(page))
            h.append("<tr><td>Found by</td><td>%s</td></tr>" % e(b["by"]))
            h.append("<tr><td>Barcode on box</td><td%s>%s</td></tr>" % (codecls, e(fmt_ean(b["box_code"]) if b["box_code"] else "not visible / not read")))
            h.append("<tr><td>Barcode in master</td><td>%s</td></tr>" % e(fmt_ean(b["master_code"])))
            if b.get("diff_pct") is not None:
                h.append("<tr><td>Rough diff</td><td>%.1f%% of visible area marked</td></tr>" % b["diff_pct"])
            h.append("</table>")
            for n in b["notes"]:
                h.append("<p>%s</p>" % e(n))
            h.append("</div>")
            if b.get("thumbs"):
                t = b["thumbs"]
                h.append('<div class="thumbs"><div><img src="%s"><br>box photo (straightened)</div>'
                         '<div><img src="%s"><br>master</div><div><img src="%s"><br>rough difference</div></div>' % tuple(map(e, t)))
            h.append("</div>")
        h.append("</div>")
    h.append("</body></html>")
    p = os.path.join(out_dir, "report.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write("".join(h))
    return p


def list_photo_sources(photos_dir):
    """Every image is one photo; every PDF page is one photo."""
    out = []
    for fn in sorted(os.listdir(photos_dir), key=str.lower):
        p = os.path.join(photos_dir, fn)
        if not os.path.isfile(p):
            continue
        if is_img(fn):
            out.append((fn, lambda p=p: load_image(p)))
        elif is_pdf(fn):
            try:
                doc = fitz.open(p)
                n = len(doc)
                doc.close()
            except Exception:
                continue
            for i in range(n):
                def ld(p=p, i=i):
                    d = fitz.open(p)
                    side = max(d[i].rect.width, d[i].rect.height) * PHOTO_PDF_DPI / 72.0
                    dpi = PHOTO_PDF_DPI * min(1.0, 5000.0 / side)
                    img, _ = render(d[i], dpi)
                    d.close()
                    return img
                out.append(("%s (page %d)" % (fn, i + 1) if n > 1 else fn, ld))
    return out


def run_check(masters_dir, photos_dir, log=log_print, open_browser=True, progress=None):
    def prog(f, text):
        if progress:
            progress(f, text)
    t0 = time.time()
    prog(0.02, "Reading masters...")
    log("Reading masters in %s" % masters_dir)
    lines = index_masters(masters_dir, log)
    if not lines:
        log("No usable master found (PDF or image).")
        prog(0, "No usable master found.")
        return None, []
    all_codes = {it["barcode"] for ln in lines for it in ln.items if it["barcode"]}
    photos = list_photo_sources(photos_dir)
    if not photos:
        log("No photos found in %s" % photos_dir)
        prog(0, "No photos found.")
        return None, []
    out_dir = os.path.join(APP_DIR, "reports", datetime.now().strftime("report_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for pno, (name, loader) in enumerate(photos, 1):
        prog(0.1 + 0.85 * (pno - 1) / len(photos), "Checking photo %d of %d: %s" % (pno, len(photos), name))
        log("Checking photo %d/%d: %s" % (pno, len(photos), name))
        try:
            r = check_photo(loader(), name, lines, all_codes, out_dir, pno, log)
        except Exception:
            log("  !! error on this photo:\n" + traceback.format_exc())
            r = None
        if r:
            results.append(r)
            for b in r["boxes"]:
                ln, i = b["line"], b["item_idx"]
                log("  box %d: %-11s %s | box %s | master %s" % (
                    b["no"], b["status"], label(ln, ln.items[i]) if ln else "unknown",
                    b["box_code"] or "-", b["master_code"] or "-"))
    prog(0.97, "Writing report...")
    p = write_report(out_dir, results, lines, masters_dir, photos_dir, time.time() - t0)
    log("Done in %.0f s. Report: %s" % (time.time() - t0, p))
    allb = [b for r in results for b in r["boxes"]]
    cnt = {}
    for b in allb:
        cnt[b["status"]] = cnt.get(b["status"], 0) + 1
    summary = "%d photos, %d boxes: " % (len(results), len(allb)) + ", ".join("%s %d" % kv for kv in sorted(cnt.items()))
    prog(1.0, "Done! " + summary)
    if open_browser:
        webbrowser.open(pathlib.Path(p).resolve().as_uri())
    return p, results


# ----------------------------------------------------------------- window
def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    st = load_json(SETTINGS_FILE, {})
    for k in ("masters", "photos"):
        if not st.get(k) or not os.path.isdir(st[k]):
            st[k] = os.path.join(APP_DIR, k)
            os.makedirs(st[k], exist_ok=True)
    save_json(SETTINGS_FILE, st)

    root = tk.Tk()
    root.title(APP_NAME + "  -  offline box checker")
    try:
        if os.name == "nt":
            root.iconbitmap(default=ICON_FILE)
    except Exception:
        pass
    root.geometry("820x600")
    root.minsize(700, 520)
    root.configure(bg="#f4f6f8")
    F = "Segoe UI"
    big = (F, 13, "bold")
    vm, vp = tk.StringVar(value=st["masters"]), tk.StringVar(value=st["photos"])
    status = tk.StringVar(value="Ready. Choose the two folders, then press RUN CHECK.")
    last = {"report": st.get("last_report") if st.get("last_report") and os.path.exists(st.get("last_report")) else None}

    def open_path(p):
        if not p:
            return
        try:
            os.startfile(p)          # Windows
        except Exception:
            webbrowser.open(pathlib.Path(p).resolve().as_uri())

    def pick(var, key, title):
        d = filedialog.askdirectory(initialdir=var.get(), title=title)
        if d:
            d = os.path.normpath(d)
            var.set(d)
            st[key] = d
            save_json(SETTINGS_FILE, st)

    tk.Label(root, text=APP_NAME, font=(F, 20, "bold"), bg="#f4f6f8", fg="#1f3b57").pack(anchor="w", padx=20, pady=(16, 0))
    tk.Label(root, text="Compare printed boxes with the master files.  Works offline - nothing is uploaded.",
             font=(F, 10), bg="#f4f6f8", fg="#555").pack(anchor="w", padx=20)

    def folder_row(step, title, hint, var, key):
        fr = tk.Frame(root, bg="white", highlightbackground="#d8dde3", highlightthickness=1)
        fr.pack(fill="x", padx=20, pady=(12, 0))
        tk.Label(fr, text=step, font=(F, 18, "bold"), bg="white", fg="#2e7bbf", width=2).grid(row=0, column=0, rowspan=2, padx=(10, 4), pady=8)
        tk.Label(fr, text=title, font=(F, 11, "bold"), bg="white").grid(row=0, column=1, sticky="w")
        tk.Label(fr, text=hint, font=(F, 9), bg="white", fg="#666").grid(row=1, column=1, sticky="w")
        tk.Label(fr, textvariable=var, font=(F, 9), bg="white", fg="#1f3b57", anchor="w", wraplength=420,
                 justify="left").grid(row=2, column=1, sticky="w", pady=(0, 8))
        tk.Button(fr, text="Choose folder...", font=(F, 11), width=15, command=lambda: pick(var, key, title)).grid(row=0, column=2, rowspan=2, padx=8)
        tk.Button(fr, text="Open", font=(F, 10), width=6, command=lambda: open_path(var.get())).grid(row=0, column=3, rowspan=2, padx=(0, 10))
        fr.grid_columnconfigure(1, weight=1)

    folder_row("1", "Masters folder", "Master PDFs or images (one page / image per flavour)", vm, "masters")
    folder_row("2", "Photos folder", "Box photos: JPG, PNG, HEIC, WEBP, BMP, TIFF or PDF", vp, "photos")

    bf = tk.Frame(root, bg="#f4f6f8")
    bf.pack(fill="x", padx=20, pady=14)
    run_btn = tk.Button(bf, text="3   RUN CHECK", font=(F, 15, "bold"), bg="#2e9e44", fg="white",
                        activebackground="#25863a", activeforeground="white", width=18, height=1, relief="flat")
    run_btn.pack(side="left")
    open_btn = tk.Button(bf, text="Open report", font=big, width=14, bg="#2e7bbf", fg="white",
                         activebackground="#24679f", activeforeground="white", relief="flat",
                         disabledforeground="#eef3f8",
                         state="normal" if last["report"] else "disabled", command=lambda: open_path(last["report"]))
    if not last["report"]:
        open_btn.config(bg="#a9bccd")
    open_btn.pack(side="left", padx=12)
    tk.Button(bf, text="All reports", font=(F, 10), command=lambda: (os.makedirs(os.path.join(APP_DIR, "reports"), exist_ok=True),
              open_path(os.path.join(APP_DIR, "reports")))).pack(side="left")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("g.Horizontal.TProgressbar", troughcolor="#e3e7eb", background="#2e9e44", thickness=18)
    bar = ttk.Progressbar(root, style="g.Horizontal.TProgressbar", maximum=1000, mode="determinate")
    bar.pack(fill="x", padx=20)
    tk.Label(root, textvariable=status, font=(F, 10, "bold"), bg="#f4f6f8", fg="#1f3b57", anchor="w",
             wraplength=760, justify="left").pack(fill="x", padx=20, pady=(6, 4))

    detail = tk.Frame(root, bg="#f4f6f8")
    detail.pack(fill="both", expand=True, padx=20, pady=(0, 14))
    box = tk.Text(detail, height=8, font=("Consolas", 9), bg="white", relief="flat", wrap="none")
    sb = ttk.Scrollbar(detail, command=box.yview)
    box.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    box.pack(fill="both", expand=True)

    def log(msg):
        root.after(0, lambda: (box.insert("end", msg + "\n"), box.see("end")))

    def progress(f, text):
        root.after(0, lambda: (bar.configure(value=int(f * 1000)), status.set(text)))

    def work():
        try:
            p, _ = run_check(vm.get(), vp.get(), log, open_browser=False, progress=progress)
            if p:
                last["report"] = p
                st["last_report"] = p
                save_json(SETTINGS_FILE, st)
                root.after(0, lambda: (open_btn.config(state="normal", bg="#2e7bbf"), open_path(p)))
        except Exception:
            log(traceback.format_exc())
            progress(0, "Something went wrong - see the details below.")
        root.after(0, lambda: run_btn.config(state="normal", text="3   RUN CHECK"))

    def start():
        if not os.path.isdir(vm.get()) or not os.path.isdir(vp.get()):
            messagebox.showerror(APP_NAME, "Please choose the masters folder and the photos folder first.")
            return
        run_btn.config(state="disabled", text="Checking...")
        box.delete("1.0", "end")
        bar.configure(value=0)
        threading.Thread(target=work, daemon=True).start()

    run_btn.config(command=start)
    root.mainloop()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) >= 2:      # command line: app.py MASTERS PHOTOS [--no-browser]
        run_check(args[0], args[1], open_browser="--no-browser" not in sys.argv)
    else:
        gui()


if __name__ == "__main__":
    main()
