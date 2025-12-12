# --- Prevent proxy interception of localhost + fix bool schema crash in old gradio_client ---
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
try:
    from gradio_client import utils as _gc_utils
    _orig_get_type = getattr(_gc_utils, "get_type", None)
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)
    if _orig_get_type is not None and _orig_get_type is not _patched_get_type:
        _gc_utils.get_type = _patched_get_type
except Exception:
    pass
# --- Patch end ---

from typing import Tuple, Dict, Any
import io, tempfile, csv
import gradio as gr
import numpy as np
import cv2 as cv
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# ===================== Processing pipeline =====================
def describe_pipeline(img_bgr: np.ndarray):
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enh = cv.cvtColor(cv.merge([l2, a, b]), cv.COLOR_LAB2BGR)

    hsv = cv.cvtColor(enh, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(hsv)
    blue_mask = cv.inRange(hsv, (80, 25, 0), (150, 255, 255))
    has_blue = (np.count_nonzero(blue_mask) > 0)

    if has_blue:
        V_sub = V[blue_mask > 0]
        V_min = int(np.min(V_sub)) if V_sub.size else int(np.min(V))
    else:
        V_sub = V.reshape(-1)
        V_min = int(np.min(V))

    def robust_delta_from_V(V_sub_arr: np.ndarray, V_min_val: int, clamp=(25, 180)) -> int:
        V_sub_u8 = V_sub_arr.astype(np.uint8).reshape(-1, 1)
        ret, _ = cv.threshold(V_sub_u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        t = int(ret)
        delta_otsu = t - V_min_val
        if delta_otsu <= 0:
            med = float(np.median(V_sub_arr))
            mad = float(np.median(np.abs(V_sub_arr - med)))
            sigma = 1.4826 * mad
            delta_val = int(3.0 * sigma)
        else:
            delta_val = int(delta_otsu)
        lo, hi = clamp
        return int(np.clip(delta_val, lo, hi))

    delta = robust_delta_from_V(V_sub, V_min)
    keep_mask_bool = (blue_mask > 0) & (V <= V_min + delta)

    result = np.full_like(enh, 255)
    result[keep_mask_bool] = enh[keep_mask_bool]

    mask = (keep_mask_bool.astype(np.uint8) * 255)
    H_img, W_img = mask.shape
    min_area_ratio = 0.0003
    min_area = int(min_area_ratio * H_img * W_img)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        final = np.full_like(enh, 255)
        return final, result, keep_mask_bool

    comps = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < max(min_area, 15):
            continue
        comp_mask = (labels == i).astype(np.uint8) * 255
        ys_comp, xs_comp = np.where(labels == i)
        if xs_comp.size < 10:
            continue
        contours, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv.contourArea)
        rect = cv.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        L = max(rw, rh)
        T = max(1.0, min(rw, rh))
        aspect_rot = L / T
        perim = cv.arcLength(cnt, True)
        compactness = (4.0 * np.pi * area) / (perim * perim + 1e-6)

        pts = np.column_stack((xs_comp, ys_comp)).astype(np.float32)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eig(cov)
        evals = np.sort(np.clip(evals, 1e-12, None))
        elong = float(np.sqrt(evals[-1] / evals[0]))

        major = evecs[:, np.argmax(evals)]
        R = np.array([[major[0], -major[1]], [major[1], major[0]]], dtype=np.float32)
        rot = (pts @ R).astype(np.float32)
        xs_r = rot[:, 0]; ys_r = rot[:, 1]

        nbins = int(np.clip(L / 20.0, 8, 40))
        bins = np.linspace(xs_r.min(), xs_r.max(), nbins + 1)
        widths = []
        for k in range(nbins):
            m = (xs_r >= bins[k]) & (xs_r < bins[k + 1])
            if np.count_nonzero(m) < 6:
                continue
            widths.append(2.0 * np.median(np.abs(ys_r[m] - np.median(ys_r[m]))))
        if len(widths) == 0:
            width_cv = 1.0
        else:
            widths = np.array(widths, dtype=np.float32)
            Tmed = np.median(widths) + 1e-6
            width_cv = float(np.std(widths) / (Tmed))

        stripe_score = (1.00*np.log1p(aspect_rot) + 0.60*np.log1p(elong) - 0.80*np.log1p(width_cv + 1e-6) - 0.50*compactness)
        comps.append({"label": i, "score": float(stripe_score)})

    if len(comps) == 0:
        final = np.full_like(enh, 255)
        return final, result, keep_mask_bool

    scores = np.array([c["score"] for c in comps], dtype=np.float32)
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max - s_min < 1e-6:
        keep_ids = set([c["label"] for c in comps])
    else:
        scaled = ((scores - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        ret, _ = cv.threshold(scaled, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        p60 = float(np.percentile(scaled, 60))
        cutoff = max(int(ret), int(p60))
        keep_ids = set([c["label"] for c in comps if (((c["score"] - s_min) / (s_max - s_min) * 255.0) >= cutoff)])

    keep_mask_adapt = np.zeros_like(mask, dtype=np.uint8)
    for c in comps:
        if c["label"] in keep_ids:
            keep_mask_adapt[labels == c["label"]] = 255
    final = np.full_like(enh, 255)
    final[keep_mask_adapt > 0] = enh[keep_mask_adapt > 0]
    return final, result, keep_mask_bool

# ============= Sampling and Analysis =============
def sample_line_profile(img_gray: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], width: int = 7):
    x1, y1 = float(p1[0]), float(p1[1]); x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1; dy = y2 - y1
    length = int(np.hypot(dx, dy))
    if length < 2:
        return None, None
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    offsets = (np.arange(width) - (width - 1) / 2.0).astype(np.float32)
    t = np.linspace(0, length, length, endpoint=False).astype(np.float32)
    xs_line = x1 + ux * t; ys_line = y1 + uy * t
    map_x = xs_line[None, :] + nx * offsets[:, None]
    map_y = ys_line[None, :] + ny * offsets[:, None]
    sampled = cv.remap(img_gray, map_x.astype(np.float32), map_y.astype(np.float32),
                       interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    profile = sampled.mean(axis=0)
    return profile, length

def analyze_profile(profile: np.ndarray, peak_distance: int = 10, peak_prominence: int = 50):
    p = profile.astype(np.float32)
    pz = p - p.mean()
    peaks, _ = find_peaks(p, distance=peak_distance, prominence=peak_prominence)
    period_pix = np.mean(np.diff(peaks)) if len(peaks) > 1 else np.nan
    F = np.fft.rfft(pz); mag = np.abs(F)
    freqs = np.fft.rfftfreq(pz.size, d=1.0)
    if mag.size > 1:
        idx_main = 1 + np.argmax(mag[1:])
        main_freq = float(freqs[idx_main])
        main_period = (1.0 / main_freq) if main_freq > 1e-9 else np.inf
    else:
        main_freq, main_period = 0.0, np.inf
    return peaks, period_pix, freqs, mag, main_freq, main_period

# ========================= Helpers =========================
BOX = 1000

def ensure_min_size_1000(img: np.ndarray, box: int = BOX) -> np.ndarray:
    h, w = img.shape[:2]
    pad_bottom = max(0, box - h); pad_right = max(0, box - w)
    if pad_bottom or pad_right:
        img = cv.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    return img

def fig_to_rgb(plt_fig) -> np.ndarray:
    buf = io.BytesIO()
    plt_fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

def draw_line(img_bgr: np.ndarray, p1: Tuple[int,int], p2: Tuple[int,int], color=(0,0,255), thickness=1):
    out = img_bgr.copy()
    cv.line(out, p1, p2, color, thickness, cv.LINE_AA)
    return out

def overlay_rect_custom(img_bgr: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    x0 = int(np.clip(x0, 0, max(0, w_img-1)))
    y0 = int(np.clip(y0, 0, max(0, h_img-1)))
    w  = int(max(1, min(w,  w_img - x0)))
    h  = int(max(1, min(h,  h_img - y0)))
    vis = img_bgr.copy()
    cv.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 255, 255), 2)
    return cv.cvtColor(vis, cv.COLOR_BGR2RGB)

def crop_roi_rect(img_bgr: np.ndarray, x0: int, y0: int, w: int, h: int):
    h_img, w_img = img_bgr.shape[:2]
    x0 = int(np.clip(x0, 0, max(0, w_img-1)))
    y0 = int(np.clip(y0, 0, max(0, h_img-1)))
    w  = int(max(1, min(w,  w_img - x0)))
    h  = int(max(1, min(h,  h_img - y0)))
    return img_bgr[y0:y0+h, x0:x0+w].copy(), (x0, y0, x0+w, y0+h)

# =============== Per-page state ===============
def make_empty_page():
    return {
        "roi_x": 0, "roi_y": 0, "roi_w": 1000, "roi_h": 1000,
        "rect": None,
        "roi_orig": None,     # RGB ndarray
        "roi_proc": None,     # RGB ndarray
        "line_pts": [],       # [(x,y), (x,y)]
        "profile_img": None,  # RGB ndarray
        "spectrum_img": None, # RGB ndarray
        "age": None           # peak count
    }

def load_page_state(store: Dict[int, Dict[str, Any]], i: int):
    return store.get(i, make_empty_page())

# ============================== Gradio App ==============================
def build_demo():
    with gr.Blocks(title="Custom ROI + Enhancement (Selectable) + Multi-page Report + CSV", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "### 🎯 Custom ROI + Selectable Enhancement (Classical / U-net / CNN) + Profile/Spectrum + Multi-page PDF + CSV (age)\n"
            "• Switching images auto-saves/restores per-image state\n"
            "• Choose enhancement method; U-net/CNN are placeholder interfaces for future models\n"
        )

        # --- Top: files & index ---
        with gr.Row():
            files = gr.Files(label="Upload images (multiple)", file_types=["image"], file_count="multiple")
            idx = gr.Number(value=0, label="Current index (starts at 0)", interactive=True, precision=0)
            btn_prev = gr.Button("⬅ Previous", size="sm")
            btn_next = gr.Button("Next ➡", size="sm")

        # Internal states
        state_paths = gr.State([])         # List[str]
        state_imgs  = gr.State([])         # List[np.ndarray] BGR
        state_curr  = gr.State(None)       # current BGR
        perpage     = gr.State({})         # Dict[int, page_state]

        # --- Left: original & overlay preview ---
        with gr.Row():
            orig_view = gr.Image(label="Original (click to set ROI top-left)", interactive=True)
            over_view = gr.Image(label="Original + ROI preview", interactive=False)

        # --- ROI numeric inputs ---
        with gr.Row():
            roi_x = gr.Number(value=0, label="ROI X (top-left)", interactive=True, precision=0)
            roi_y = gr.Number(value=0, label="ROI Y (top-left)", interactive=True, precision=0)
            roi_w = gr.Number(value=1000, label="ROI Width (W)", interactive=True, precision=0)
            roi_h = gr.Number(value=1000, label="ROI Height (H)", interactive=True, precision=0)

        with gr.Row():
            btn_preview = gr.Button("👀 Preview ROI", variant="secondary")
            nudge = 10
            btn_up    = gr.Button(f"↑ Up {nudge}px", size="sm")
            btn_down  = gr.Button(f"↓ Down {nudge}px", size="sm")
            btn_left  = gr.Button(f"← Left {nudge}px", size="sm")
            btn_right = gr.Button(f"→ Right {nudge}px", size="sm")

        # --- Enhancement selection + run & refresh ---
        with gr.Row():
            enh_method = gr.Dropdown(choices=["Classical segmentation", "U-net", "CNN"], value="Classical segmentation",
                                     label="Enhancement method", interactive=True)
            btn_run_enh = gr.Button("⚙️ Run (with selected method)", variant="primary")
            btn_refresh = gr.Button("🧹 Refresh current page", variant="secondary")

        # --- Right: processed & original ROI views ---
        with gr.Row():
            roi_proc_view = gr.Image(label="Processed ROI (click twice here to draw a line)", interactive=True)
            roi_orig_view = gr.Image(label="Original ROI (with line)", interactive=False)

        # Parameters
        with gr.Accordion("Analysis parameters", open=False):
            thickness = gr.Slider(1, 99, value=7, step=2, label="Line width (pixels averaged along normal)")
            peak_distance = gr.Slider(1, 200, value=10, step=1, label="Min peak distance (px)")
            peak_prom = gr.Slider(0, 255, value=50, step=1, label="Peak prominence")

        with gr.Row():
            profile_img = gr.Image(label="Intensity profile", interactive=False)
            spectrum_img = gr.Image(label="Spectrum", interactive=False)

        # --- Export ---
        with gr.Row():
            btn_export_pdf = gr.Button("📄 Export current page PDF", variant="secondary")
            btn_export_all = gr.Button("🗂️ Export all PDFs (multi-page)", variant="secondary")
            btn_export_csv = gr.Button("📊 Export all CSV (image, age)", variant="secondary")
        with gr.Row():
            pdf_file = gr.File(label="Download (current PDF)", interactive=False)
            pdf_all  = gr.File(label="Download (all PDFs)", interactive=False)
            csv_all  = gr.File(label="Download CSV (image, age)", interactive=False)

        info = gr.Markdown("👋 Ready.")

        # ---------------- Handlers ----------------
        def on_files(files_list):
            paths = [f.name for f in files_list] if files_list else []
            imgs = []
            for p in paths:
                data = cv.imdecode(np.fromfile(p, dtype=np.uint8), cv.IMREAD_COLOR)
                if data is not None:
                    data = ensure_min_size_1000(data)
                    imgs.append(data)
            curr = imgs[0] if imgs else None

            pstore = {}
            if imgs:
                for i, im in enumerate(imgs):
                    h, w = im.shape[:2]
                    pg = make_empty_page()
                    pg["roi_w"] = min(1000, w)
                    pg["roi_h"] = min(1000, h)
                    pstore[i] = pg

            return (paths, imgs, curr, 0, pstore,
                    cv.cvtColor(curr, cv.COLOR_BGR2RGB) if curr is not None else None,
                    None,
                    pstore.get(0, make_empty_page())["roi_x"],
                    pstore.get(0, make_empty_page())["roi_y"],
                    pstore.get(0, make_empty_page())["roi_w"],
                    pstore.get(0, make_empty_page())["roi_h"],
                    None, None, None,
                    "✅ Images loaded. Set ROI on the original image, then choose enhancement and run.")

        files.upload(
            fn=on_files,
            inputs=[files],
            outputs=[state_paths, state_imgs, state_curr, idx, perpage,
                     orig_view, over_view,
                     roi_x, roi_y, roi_w, roi_h,
                     roi_proc_view, roi_orig_view, profile_img, info]
        )

        # Index change: restore page state
        def change_index(i, imgs, pstore):
            if not imgs:
                return None, None, None, 0, 0, 1000, 1000, None, None, None, "⚠️ No images."
            i = int(max(0, min(len(imgs) - 1, int(i))))
            cur = imgs[i]
            page = load_page_state(pstore, i)
            over = overlay_rect_custom(cur, page["roi_x"], page["roi_y"], page["roi_w"], page["roi_h"])
            return (cv.cvtColor(cur, cv.COLOR_BGR2RGB),
                    over,
                    cur,
                    page["roi_x"], page["roi_y"], page["roi_w"], page["roi_h"],
                    page["roi_proc"], page["roi_orig"], page["profile_img"],
                    f"📌 Current index {i}. Page state restored.")

        idx.change(
            fn=change_index,
            inputs=[idx, state_imgs, perpage],
            outputs=[orig_view, over_view, state_curr, roi_x, roi_y, roi_w, roi_h,
                     roi_proc_view, roi_orig_view, profile_img, info]
        )

        # Prev/Next: only change idx
        btn_prev.click(fn=lambda i, imgs: int((int(i)-1) % len(imgs)) if imgs else i,
                       inputs=[idx, state_imgs], outputs=[idx])
        btn_next.click(fn=lambda i, imgs: int((int(i)+1) % len(imgs)) if imgs else i,
                       inputs=[idx, state_imgs], outputs=[idx])

        # Click original: set ROI top-left
        def on_click_orig(evt: gr.SelectData, curr_bgr, w, h, i, pstore):
            if curr_bgr is None:
                return None, None, None, pstore, "⚠️ Please upload images first."
            x0, y0 = int(evt.index[0]), int(evt.index[1])
            vis = overlay_rect_custom(curr_bgr, x0, y0, int(w), int(h))
            page = load_page_state(pstore, int(i))
            page["roi_x"], page["roi_y"], page["roi_w"], page["roi_h"] = int(x0), int(y0), int(w), int(h)
            pstore[int(i)] = page
            return x0, y0, vis, pstore, f"🎯 ROI top-left set to ({x0}, {y0}), size=({int(w)}, {int(h)})."

        orig_view.select(
            fn=on_click_orig,
            inputs=[state_curr, roi_w, roi_h, idx, perpage],
            outputs=[roi_x, roi_y, over_view, perpage, info]
        )

        # Preview ROI
        def preview_roi(curr_bgr, x0, y0, w, h, i, pstore):
            if curr_bgr is None:
                return None, None, pstore, "⚠️ Please upload images first."
            vis = overlay_rect_custom(curr_bgr, int(x0), int(y0), int(w), int(h))
            roi_orig_bgr, rect = crop_roi_rect(curr_bgr, int(x0), int(y0), int(w), int(h))
            page = load_page_state(pstore, int(i))
            page["rect"] = rect
            page["roi_orig"] = cv.cvtColor(roi_orig_bgr, cv.COLOR_BGR2RGB)
            page["line_pts"] = []
            page["profile_img"] = None
            page["spectrum_img"] = None
            page["age"] = None
            pstore[int(i)] = page
            return vis, page["roi_orig"], pstore, "👀 ROI previewed. Choose enhancement then click Run."

        btn_preview.click(
            fn=preview_roi,
            inputs=[state_curr, roi_x, roi_y, roi_w, roi_h, idx, perpage],
            outputs=[over_view, roi_orig_view, perpage, info]
        )

        # Nudge ROI
        def nudge_roi(x0, y0, w, h, dx, dy, curr_bgr, i, pstore):
            if curr_bgr is None:
                return x0, y0, None, pstore, "⚠️ Please upload images first."
            x0 = int(x0) + dx; y0 = int(y0) + dy
            vis = overlay_rect_custom(curr_bgr, x0, y0, int(w), int(h))
            page = load_page_state(pstore, int(i))
            page["roi_x"], page["roi_y"], page["roi_w"], page["roi_h"] = x0, y0, int(w), int(h)
            pstore[int(i)] = page
            return x0, y0, vis, pstore, f"🔧 ROI moved to ({x0}, {y0})."

        btn_up.click(   fn=lambda x,y,w,h,img,i,ps: nudge_roi(x,y,w,h,0,-10,img,i,ps), inputs=[roi_x, roi_y, roi_w, roi_h, state_curr, idx, perpage], outputs=[roi_x, roi_y, over_view, perpage, info])
        btn_down.click( fn=lambda x,y,w,h,img,i,ps: nudge_roi(x,y,w,h,0, 10,img,i,ps), inputs=[roi_x, roi_y, roi_w, roi_h, state_curr, idx, perpage], outputs=[roi_x, roi_y, over_view, perpage, info])
        btn_left.click( fn=lambda x,y,w,h,img,i,ps: nudge_roi(x,y,w,h,-10,0,img,i,ps), inputs=[roi_x, roi_y, roi_w, roi_h, state_curr, idx, perpage], outputs=[roi_x, roi_y, over_view, perpage, info])
        btn_right.click(fn=lambda x,y,w,h,img,i,ps: nudge_roi(x,y,w,h, 10,0,img,i,ps), inputs=[roi_x, roi_y, roi_w, roi_h, state_curr, idx, perpage], outputs=[roi_x, roi_y, over_view, perpage, info])

        # Run (by method)
        def run_enhance(curr_bgr, x0, y0, w, h, i, pstore, method):
            if curr_bgr is None:
                return None, None, pstore, "⚠️ Please upload images first."
            roi_orig_bgr, rect = crop_roi_rect(curr_bgr, int(x0), int(y0), int(w), int(h))
            page = load_page_state(pstore, int(i))
            page["rect"] = rect
            page["roi_orig"] = cv.cvtColor(roi_orig_bgr, cv.COLOR_BGR2RGB)

            if method == "Classical segmentation":
                proc_roi_bgr, _, _ = describe_pipeline(roi_orig_bgr)
                page["roi_proc"] = cv.cvtColor(proc_roi_bgr, cv.COLOR_BGR2RGB)
                page["line_pts"] = []
                page["profile_img"] = None
                page["spectrum_img"] = None
                page["age"] = None
                pstore[int(i)] = page
                return page["roi_proc"], page["roi_orig"], pstore, "✅ Classical segmentation done. Now click twice on the processed ROI."

            elif method == "U-net":
                # Placeholder for future U-net model
                page["roi_proc"] = None
                page["line_pts"], page["profile_img"], page["spectrum_img"], page["age"] = [], None, None, None
                pstore[int(i)] = page
                return None, page["roi_orig"], pstore, "ℹ️ U-net interface placeholder (not implemented)."

            elif method == "CNN":
                # Placeholder for future CNN model
                page["roi_proc"] = None
                page["line_pts"], page["profile_img"], page["spectrum_img"], page["age"] = [], None, None, None
                pstore[int(i)] = page
                return None, page["roi_orig"], pstore, "ℹ️ CNN interface placeholder (not implemented)."

            else:
                pstore[int(i)] = page
                return None, page["roi_orig"], pstore, f"⚠️ Unknown method: {method}"

        btn_run_enh.click(
            fn=run_enhance,
            inputs=[state_curr, roi_x, roi_y, roi_w, roi_h, idx, perpage, enh_method],
            outputs=[roi_proc_view, roi_orig_view, perpage, info]
        )

        # Draw line on processed ROI -> profile + spectrum + age
        def on_click_roi(evt: gr.SelectData, roi_proc_img, roi_orig_img, thick, dmin, prom, i, pstore):
            if roi_proc_img is None:
                return roi_proc_img, roi_orig_img, None, None, pstore, "⚠️ No processed image available. Run 'Classical segmentation' first."
            x, y = int(evt.index[0]), int(evt.index[1])

            page = load_page_state(pstore, int(i))
            pts = list(page["line_pts"] or [])
            if len(pts) >= 2:
                pts = []
            pts.append((x, y))

            # One point: mark
            if len(pts) == 1:
                vis_proc = cv.cvtColor(roi_proc_img, cv.COLOR_RGB2BGR).copy()
                cv.circle(vis_proc, pts[0], 4, (0, 0, 255), -1, cv.LINE_AA)
                vis_proc = cv.cvtColor(vis_proc, cv.COLOR_BGR2RGB)
                page["roi_proc"] = vis_proc
                page["line_pts"] = pts
                pstore[int(i)] = page
                return vis_proc, roi_orig_img, None, None, pstore, f"🖱 Start: {pts[0]}. Click once more to set the end."

            # Two points: compute
            p1, p2 = pts[0], pts[1]
            proc_bgr = cv.cvtColor(roi_proc_img, cv.COLOR_RGB2BGR)
            orig_bgr = cv.cvtColor(roi_orig_img, cv.COLOR_RGB2BGR)

            gray_base = cv.cvtColor(proc_bgr, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
            eq = clahe.apply(gray_base)
            norm = cv.normalize(eq, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            profile, length = sample_line_profile(norm, p1, p2, width=int(thick))
            if profile is None or len(profile) == 0:
                return roi_proc_img, roi_orig_img, None, None, pstore, "⚠️ Segment too short or out of bounds. Please redraw."

            peaks, period_pix, freqs, mag, main_freq, main_period = analyze_profile(profile, int(dmin), int(prom))

            vis_proc2 = draw_line(proc_bgr, p1, p2, (0, 0, 255), 1)
            vis_orig2 = draw_line(orig_bgr, p1, p2, (0, 0, 255), 1)

            # Profile figure
            fig1 = plt.figure(figsize=(6, 3))
            plt.plot(profile, label='Intensity')
            if len(peaks) > 0:
                plt.plot(peaks, profile[peaks], 'rx', label='Peaks')
            ttl = f"Profile: peaks={len(peaks)}"
            if not np.isnan(period_pix):
                ttl += f" | avg period≈{period_pix:.1f}px"
            plt.title(ttl); plt.xlabel("Position (px)"); plt.ylabel("Intensity"); plt.legend()
            img_profile = fig_to_rgb(fig1); plt.close(fig1)

            # Spectrum figure
            fig2 = plt.figure(figsize=(6, 3))
            plt.plot(freqs, mag)
            plt.title(f"Spectrum: main freq≈{main_freq:.4f} cyc/px → period≈{main_period:.1f} px")
            plt.xlabel("Frequency (cycles/pixel)"); plt.ylabel("Magnitude")
            img_spec = fig_to_rgb(fig2); plt.close(fig2)

            vis_proc_rgb = cv.cvtColor(vis_proc2, cv.COLOR_BGR2RGB)
            vis_orig_rgb = cv.cvtColor(vis_orig2, cv.COLOR_BGR2RGB)

            msg = (f"✅ Segment {p1}→{p2} | length≈{length}px | "
                   f"peaks={len(peaks)} | avg period≈{period_pix:.1f}px | "
                   f"main≈{main_freq:.4f} cyc/px → {main_period:.1f}px")

            # Save & record age
            page["roi_proc"] = vis_proc_rgb
            page["roi_orig"] = vis_orig_rgb
            page["line_pts"] = pts
            page["profile_img"] = img_profile
            page["spectrum_img"] = img_spec
            page["age"] = int(len(peaks))
            pstore[int(i)] = page

            return vis_proc_rgb, vis_orig_rgb, img_profile, img_spec, pstore, msg

        roi_proc_view.select(
            fn=on_click_roi,
            inputs=[roi_proc_view, roi_orig_view, thickness, peak_distance, peak_prom, idx, perpage],
            outputs=[roi_proc_view, roi_orig_view, profile_img, spectrum_img, perpage, info]
        )

        # Refresh current page
        def refresh_page(i, imgs, pstore, curr_bgr):
            if not imgs or curr_bgr is None:
                return pstore, None, None, None, None, "⚠️ Nothing to refresh."
            i = int(i)
            newp = make_empty_page()
            h, w = curr_bgr.shape[:2]
            newp["roi_w"] = min(1000, w)
            newp["roi_h"] = min(1000, h)
            pstore[i] = newp
            over = overlay_rect_custom(curr_bgr, newp["roi_x"], newp["roi_y"], newp["roi_w"], newp["roi_h"])
            return pstore, over, None, None, None, f"🧹 Cleared page {i}."

        btn_refresh.click(
            fn=refresh_page,
            inputs=[idx, state_imgs, perpage, state_curr],
            outputs=[perpage, over_view, roi_proc_view, roi_orig_view, profile_img, info]
        )

        # Clear line only
        def reset_line(i, pstore):
            i = int(i)
            page = load_page_state(pstore, i)
            page["line_pts"] = []
            page["profile_img"] = None
            page["spectrum_img"] = None
            page["age"] = None
            pstore[i] = page
            return pstore, page["roi_proc"], page["roi_orig"], None, None, "🧹 Cleared line & curves."

        gr.Button("Clear line", size="sm").click(
            fn=reset_line,
            inputs=[idx, perpage],
            outputs=[perpage, roi_proc_view, roi_orig_view, profile_img, spectrum_img, info]
        )

        # -------- Export: current page PDF --------
        def export_pdf_current(i, pstore, thick, dmin, prom):
            page = load_page_state(pstore, int(i))
            if page["roi_orig"] is None or page["roi_proc"] is None:
                return None, "⚠️ This page has no processed image. Run 'Classical segmentation' first."

            fig = plt.figure(figsize=(8.27, 11.69))
            gs = fig.add_gridspec(6, 2, hspace=0.8, wspace=0.3)

            ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
            title = f"Page {int(i)} — ROI Enhancement & Line Profile Report"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ax_title.text(0.01, 0.75, title, fontsize=16, weight='bold')
            ax_title.text(0.01, 0.42, f"Generated at: {now}", fontsize=10)
            ax_title.text(0.01, 0.20, f"ROI rect: {page['rect']}\nLine points: {page['line_pts']}\n"
                                      f"Params — width={int(thick)}, peak_distance={int(dmin)}, peak_prom={int(prom)}\n"
                                      f"age(peaks)={page['age']}",
                          fontsize=10, va='top')

            ax1 = fig.add_subplot(gs[1:3, 0]); ax1.set_title("Processed ROI"); ax1.axis('off'); ax1.imshow(page["roi_proc"])
            ax2 = fig.add_subplot(gs[1:3, 1]); ax2.set_title("Original ROI (with line)"); ax2.axis('off'); ax2.imshow(page["roi_orig"])

            ax3 = fig.add_subplot(gs[3:5, :]); ax3.axis('off'); ax3.set_title("Intensity Profile")
            if page["profile_img"] is not None: ax3.imshow(page["profile_img"])
            else: ax3.text(0.5, 0.5, "No profile yet", ha='center', va='center')

            ax4 = fig.add_subplot(gs[5, :]); ax4.axis('off'); ax4.set_title("Spectrum")
            if page["spectrum_img"] is not None: ax4.imshow(page["spectrum_img"])
            else: ax4.text(0.5, 0.5, "No spectrum yet", ha='center', va='center')

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf_path = tmp.name
            plt.savefig(pdf_path, bbox_inches='tight'); plt.close(fig)
            return pdf_path, "✅ Current page PDF generated."

        btn_export_pdf.click(
            fn=export_pdf_current,
            inputs=[idx, perpage, thickness, peak_distance, peak_prom],
            outputs=[pdf_file, info]
        )

        # -------- Export: all pages PDF --------
        def export_pdf_all(pstore, n_imgs, thick, dmin, prom):
            if n_imgs is None or n_imgs == 0:
                return None, "⚠️ No images."

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            out_path = tmp.name

            with PdfPages(out_path) as pdf:
                for i in range(n_imgs):
                    page = load_page_state(pstore, i)
                    fig = plt.figure(figsize=(8.27, 11.69))
                    gs = fig.add_gridspec(6, 2, hspace=0.8, wspace=0.3)

                    ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
                    title = f"Page {i} — ROI Enhancement & Line Profile Report"
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ax_title.text(0.01, 0.75, title, fontsize=16, weight='bold')
                    ax_title.text(0.01, 0.42, f"Generated at: {now}", fontsize=10)
                    ax_title.text(0.01, 0.20, f"ROI rect: {page['rect']}\nLine points: {page['line_pts']}\n"
                                              f"Params — width={int(thick)}, peak_distance={int(dmin)}, peak_prom={int(prom)}\n"
                                              f"age(peaks)={page['age']}",
                                  fontsize=10, va='top')

                    ax1 = fig.add_subplot(gs[1:3, 0]); ax1.set_title("Processed ROI"); ax1.axis('off')
                    if page["roi_proc"] is not None: ax1.imshow(page["roi_proc"])
                    else: ax1.text(0.5,0.5,"Not processed",ha='center',va='center')

                    ax2 = fig.add_subplot(gs[1:3, 1]); ax2.set_title("Original ROI (with line)"); ax2.axis('off')
                    if page["roi_orig"] is not None: ax2.imshow(page["roi_orig"])
                    else: ax2.text(0.5,0.5,"No ROI preview",ha='center',va='center')

                    ax3 = fig.add_subplot(gs[3:5, :]); ax3.axis('off'); ax3.set_title("Intensity Profile")
                    if page["profile_img"] is not None: ax3.imshow(page["profile_img"])
                    else: ax3.text(0.5,0.5,"No profile",ha='center',va='center')

                    ax4 = fig.add_subplot(gs[5, :]); ax4.axis('off'); ax4.set_title("Spectrum")
                    if page["spectrum_img"] is not None: ax4.imshow(page["spectrum_img"])
                    else: ax4.text(0.5,0.5,"No spectrum",ha='center',va='center')

                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            return out_path, "✅ All pages PDF generated."

        btn_export_all.click(
            fn=lambda pstore, imgs, thick, dmin, prom: export_pdf_all(pstore, len(imgs) if imgs else 0, thick, dmin, prom),
            inputs=[perpage, state_imgs, thickness, peak_distance, peak_prom],
            outputs=[pdf_all, info]
        )

        # -------- Export: CSV (image, age) --------
        def export_csv_all(pstore, paths):
            if not paths:
                return None, "⚠️ No images."
            rows = []
            for i, p in enumerate(paths):
                page = load_page_state(pstore, i)
                age = page.get("age", None)
                if age is not None:
                    img_name = os.path.basename(p)
                    rows.append([img_name, int(age)])
            if not rows:
                return None, "⚠️ No pages with computed age (peaks)."

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            csv_path = tmp.name
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "age"])
                writer.writerows(rows)
            return csv_path, f"✅ Exported {len(rows)} rows to CSV."

        btn_export_csv.click(
            fn=export_csv_all,
            inputs=[perpage, state_paths],
            outputs=[csv_all, info]
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
