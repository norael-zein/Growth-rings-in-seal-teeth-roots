from visualization_integrate import visualization


# =============================================================================
# Prevent proxy interception of localhost and patch old gradio_client boolean
# schema crash by overriding get_type for boolean schemas.
# =============================================================================
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
# =============================================================================
# Patch end
# =============================================================================


# =============================================================================
# Standard library and third-party imports for IO, image processing, plotting,
# numerical computation, and UI construction.
# =============================================================================
from typing import Tuple, Dict, Any
import io, tempfile, csv, base64
import gradio as gr
import numpy as np
import cv2 as cv
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.compression"] = 0
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["image.interpolation"] = "nearest"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter


# =============================================================================
# PyTorch imports and utilities for CNN-based age regression inference.
# =============================================================================
import torch
import torch.nn as nn
from functools import lru_cache


# =============================================================================
# Squeeze-and-Excitation block used to recalibrate channel-wise feature responses.
# =============================================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =============================================================================
# CNN regression model definition for predicting age from a fixed-size ROI.
# Input shape:  [B, 3, 200, 400]
# Output shape: [B] (scalar age regression)
# =============================================================================
class BetterCNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 200x400 -> 100x200

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 100x200 -> 50x100

            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 4), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            SEBlock(channel=128, reduction=16)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 50))  # [B,128,50,100] -> [B,128,1,50]

        self.conv_head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)   # [B,128,50,100]
        x = self.pool(x)       # [B,128,1,50]
        x = x.squeeze(2)       # [B,128,50]
        x = self.conv_head(x)  # [B,64,50]
        x = self.fc(x)         # [B,1]
        return x.squeeze(1)    # [B]


# =============================================================================
# CNN model loading, caching, and ROI preprocessing to match training pipeline.
# =============================================================================
CNN_WEIGHTS_PATH = "cnn_age.pth"   # Set this to your weights file path (default: same directory)
CNN_DEVICE = "cpu"                # Use "cuda" if CUDA is available and configured

@lru_cache(maxsize=1)
def load_cnn_model_cached(weights_path: str = CNN_WEIGHTS_PATH, device_str: str = CNN_DEVICE):
    device = torch.device(device_str)
    model = BetterCNNRegressor().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def preprocess_roi_for_bettercnn(roi_rgb: np.ndarray) -> torch.Tensor:
    if roi_rgb is None:
        raise ValueError("roi_rgb is None")

    # Force model input size: W=400, H=200
    roi_rgb = cv.resize(roi_rgb, (400, 200), interpolation=cv.INTER_AREA)

    x = roi_rgb.astype(np.float32) / 255.0  # 0~1
    x = np.transpose(x, (2, 0, 1))          # HWC -> CHW
    x = torch.from_numpy(x).unsqueeze(0)    # 1x3x200x400
    return x.contiguous()


# =============================================================================
# General helper utilities for padding, drawing, image conversion, and encoding.
# =============================================================================
BOX = 1000

def ensure_min_size_1000(img: np.ndarray, box: int = BOX) -> np.ndarray:
    h, w = img.shape[:2]
    pad_bottom = max(0, box - h)
    pad_right = max(0, box - w)
    if pad_bottom or pad_right:
        img = cv.copyMakeBorder(
            img, 0, pad_bottom, 0, pad_right,
            cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return img


def fig_to_rgb(plt_fig) -> np.ndarray:
    buf = io.BytesIO()
    plt_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.50, dpi=300)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)


def draw_line(img_bgr: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
              color=(0, 0, 255), thickness=1):
    out = img_bgr.copy()
    cv.line(out, p1, p2, color, thickness, cv.LINE_AA)
    return out

def draw_red_x_bgr(img_bgr: np.ndarray, x: int, y: int, size: int = 7, thickness: int = 2):
    cv.line(img_bgr, (x - size, y - size), (x + size, y + size), (0, 0, 255), thickness, cv.LINE_AA)
    cv.line(img_bgr, (x - size, y + size), (x + size, y - size), (0, 0, 255), thickness, cv.LINE_AA)


def b64_image_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
        mime = "image/png" if ext == "png" else f"image/{ext}"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


# =============================================================================
# Dataset-like image enhancement using PIL to match training-time augmentation.
# =============================================================================
def enhance_pil_like_dataset(
    img: Image.Image,
    contrast_factor: float = 2.0,
    noise_filter_size: int = 1,
    sharpen_radius: float = 2.0,
    sharpen_percent: int = 180,
    sharpen_threshold: int = 5
) -> Image.Image:
    img = img.filter(ImageFilter.MedianFilter(size=int(noise_filter_size)))
    img = ImageEnhance.Contrast(img).enhance(float(contrast_factor))
    img = img.filter(ImageFilter.UnsharpMask(
        radius=float(sharpen_radius),
        percent=int(sharpen_percent),
        threshold=int(sharpen_threshold)
    ))
    return img


def enhance_bgr_like_dataset(
    curr_bgr: np.ndarray,
    contrast_factor: float,
    noise_filter_size: int,
    sharpen_radius: float,
    sharpen_percent: int,
    sharpen_threshold: int
) -> np.ndarray:
    rgb = cv.cvtColor(curr_bgr, cv.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil2 = enhance_pil_like_dataset(
        pil,
        contrast_factor=contrast_factor,
        noise_filter_size=noise_filter_size,
        sharpen_radius=sharpen_radius,
        sharpen_percent=sharpen_percent,
        sharpen_threshold=sharpen_threshold
    )
    rgb2 = np.array(pil2).astype(np.uint8)
    return cv.cvtColor(rgb2, cv.COLOR_RGB2BGR)


# =============================================================================
# Rotated ROI extraction by a user-defined line using a perspective transform.
# =============================================================================
def rotated_roi_from_line(
    img_bgr: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    roi_height_px: int,
    out_w: int = 400,
    out_h: int = 200,
    pad_value: int = 255
) -> np.ndarray:
    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])
    dx = x1 - x0
    dy = y1 - y0
    L = float(np.hypot(dx, dy))
    if L < 2.0:
        return None

    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    half_h = max(1.0, float(roi_height_px) / 2.0)

    src = np.array([
        [x0 - nx * half_h, y0 - ny * half_h],
        [x1 - nx * half_h, y1 - ny * half_h],
        [x1 + nx * half_h, y1 + ny * half_h],
        [x0 + nx * half_h, y0 + ny * half_h],
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(src, dst)

    roi_bgr = cv.warpPerspective(
        img_bgr, M, (out_w, out_h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(pad_value, pad_value, pad_value)
    )
    roi_rgb = cv.cvtColor(roi_bgr, cv.COLOR_BGR2RGB)
    return roi_rgb


# =============================================================================
# Intensity profile extraction and adaptive peak detection for ROI analysis.
# (UPDATED: follow the same principle as visualization())
# =============================================================================
def roi_center_band_profile(roi_gray_u8: np.ndarray, band: int = 3) -> np.ndarray:
    """
    - Sample points along ROI center horizontal line.
    - Average a small vertical band around that line.
    Returns: profile_u8, shape [W]
    """
    if roi_gray_u8 is None:
        raise ValueError("roi_gray_u8 is None")
    if roi_gray_u8.ndim != 2:
        raise ValueError(f"roi_gray_u8 must be 2D grayscale, got shape={roi_gray_u8.shape}")

    h, w = roi_gray_u8.shape
    if h == 0 or w == 0:
        return np.zeros((0,), dtype=np.uint8)

    band = int(max(1, band))
    if band % 2 == 0:
        band += 1
    half = band // 2

    y_center = h // 2
    xs = np.arange(w, dtype=np.int32)
    ys = np.full((w,), y_center, dtype=np.int32)

    profiles = []
    for o in range(-half, half + 1):
        yy = np.clip(ys + o, 0, h - 1)
        profiles.append(roi_gray_u8[yy, xs])

    profile = np.mean(np.stack(profiles, axis=0), axis=0).astype(np.uint8)
    return profile


def adaptive_peaks_like_visualization(profile_u8: np.ndarray):
    """
    Detect peaks on inverted profile (dark rings) with adaptive distance/prominence.
    Returns: peaks (np.ndarray), dbg (dict)
    """
    if profile_u8 is None:
        raise ValueError("profile_u8 is None")
    if profile_u8.ndim != 1:
        profile_u8 = np.ravel(profile_u8)

    p = profile_u8.astype(np.float32)
    if p.size == 0:
        return np.array([], dtype=np.int32), {
            "dyn": 0.0, "typical": 0.0, "min_dist": 0, "prom": 0.0,
            "noise": 0.0, "hmin_inv": 0.0, "cand_peaks": 0
        }

    dyn = float(np.percentile(p, 95) - np.percentile(p, 5))
    prom_lo = max(5.0, 0.30 * dyn)

    cand_peaks, _ = find_peaks(p, prominence=prom_lo, distance=2)

    if len(cand_peaks) >= 3:
        diffs = np.diff(cand_peaks)
        lo, hi = np.percentile(diffs, [20, 80])
        diffs_mid = diffs[(diffs >= lo) & (diffs <= hi)]
        if len(diffs_mid) > 0:
            typical = float(np.median(diffs_mid))
        else:
            typical = float(np.median(diffs))
    else:
        typical = 20.0

    L = int(len(p))
    scale = float(np.clip(300.0 / max(L, 1), 0.6, 1.4))
    min_dist = int(np.clip(0.9 * typical * scale, 20, 80))

    prom = max(15.0, 0.35 * dyn)

    inv = 255.0 - p

    d = np.diff(inv)
    if d.size:
        noise = float(1.4826 * np.median(np.abs(d - np.median(d))))
    else:
        noise = 0.0

    prom = max(prom, 4.0 * noise)
    hmin_inv = float(np.percentile(inv, 50))

    peaks, _ = find_peaks(
        inv,
        distance=min_dist,
        prominence=prom,
        height=hmin_inv,
        width=1
    )

    dbg = {
        "dyn": float(dyn),
        "typical": float(typical),
        "min_dist": int(min_dist),
        "prom": float(prom),
        "noise": float(noise),
        "hmin_inv": float(hmin_inv),
        "cand_peaks": int(len(cand_peaks)),
    }
    return peaks.astype(np.int32), dbg


# =============================================================================
# Per-page state management for multi-image workflows in the Gradio UI.
# =============================================================================
def make_empty_page():
    return {
        "line_pts": [],
        "orig_rgb": None,        # RGB (original)
        "image_name": "",        # image basename for report
        "orig_with_line": None,  # RGB
        "roi_preview": None,     # RGB (rotated crop)
        "profile_img": None,     # RGB
        "age": None,             # Age derived from peak count
        "cnn_age": None          # CNN predicted age
    }

def load_page_state(store: Dict[int, Dict[str, Any]], i: int):
    return store.get(i, make_empty_page())


# =============================================================================
# Gradio application builder
# (UPDATED: removed ROI/Preprocess params UI, but keep valley "x" markers)
# =============================================================================
def build_demo():
    TITLE_TXT = "#052453"
    MUTED_TXT = "#1c3e81"
    HEADER_BG_RGBA = "#1B5D79"

    PRIMARY_1 = TITLE_TXT
    PRIMARY_2 = "#1c3e81"
    ACCENT_TXT = "#a2143a"

    # ---- Fixed defaults (same as your sliders used to be) ----
    DEFAULT_OUT_W = 400
    DEFAULT_OUT_H = 200
    DEFAULT_ROI_HEIGHT_PX = 80

    DEFAULT_CONTRAST = 2.0
    DEFAULT_MEDIAN_SZ = 1
    DEFAULT_SHARPEN_RADIUS = 2.0
    DEFAULT_SHARPEN_PERCENT = 180
    DEFAULT_SHARPEN_THRESHOLD = 5
    DEFAULT_BAND = 3
    # ---------------------------------------------------------

    def popup_warn(msg: str):
        fn = getattr(gr, "Warning", None)
        if callable(fn):
            fn(msg)

    custom_css = f"""
        :root {{
          --primary-50:  {PRIMARY_1}10;
          --primary-100: {PRIMARY_1}22;
          --primary-200: {PRIMARY_1}33;
          --primary-300: {PRIMARY_2}55;
          --primary-400: {PRIMARY_2}77;
          --primary-500: {PRIMARY_1};
          --primary-600: {PRIMARY_2};
          --primary-700: {PRIMARY_2};
          --primary-800: {PRIMARY_1};
          --primary-900: {PRIMARY_1};
          --accent: {ACCENT_TXT};
          --muted:  {MUTED_TXT};
        }}

        button.primary {{
          background: {PRIMARY_1} !important;
          border-color: {PRIMARY_1} !important;
        }}
        button.primary:hover {{
          background: {PRIMARY_2} !important;
          border-color: {PRIMARY_2} !important;
        }}
        button.secondary {{
          color: {PRIMARY_1} !important;
          border-color: {PRIMARY_1}66 !important;
        }}

        a, a:hover {{ color: {PRIMARY_2}; }}

        .accent {{ color: {ACCENT_TXT}; font-weight: 700; }}
        .muted  {{ color: {MUTED_TXT}; }}

        .gr-label, label, .wrap label {{
          color: {PRIMARY_1} !important;
          font-weight: 600;
        }}

        #top_icon_row {{
          display: flex;
          align-items: center;
          justify-content: flex-start;
          gap: 12px;
          padding: 10px 12px;
          border-radius: 14px;
          border: 1px solid {PRIMARY_1}22;
          background: {HEADER_BG_RGBA} !important;
          margin-bottom: 8px;
        }}
        #top_icon_row img {{
          height: 44px;
          width: auto;
          display: block;
          filter: invert(1);
        }}
        #top_icon_row .title {{
          color: #ffffff;
          font-weight: 800;
          font-size: 18px;
          line-height: 1.2;
        }}
        #top_icon_row .subtitle {{
          color: {MUTED_TXT} !important;
          font-size: 13px;
          margin-top: 2px;
        }}

        #roi_preview_view img {{
          transform: translateY(45px) !important;
        }}
        #roi_preview_view .image-container,
        #roi_preview_view .image-frame,
        #roi_preview_view [data-testid="image"],
        #roi_preview_view .wrap {{
          padding-bottom: 45px !important;
          box-sizing: border-box !important;
          overflow: visible !important;
        }}
    """

    custom_theme = gr.themes.Soft().set(
        body_background_fill=f"linear-gradient(180deg, {PRIMARY_1}08, transparent)",
        block_background_fill="white",
        block_border_color=f"{PRIMARY_1}22",
        body_text_color=PRIMARY_1,
        body_text_color_subdued=MUTED_TXT,
        button_primary_background_fill=PRIMARY_1,
        button_primary_background_fill_hover=PRIMARY_2,
        button_primary_text_color="white",
        button_secondary_text_color=PRIMARY_1,
        button_secondary_border_color=f"{PRIMARY_1}66",
        input_border_color=f"{PRIMARY_1}33",
        input_border_color_focus=PRIMARY_2,
    )

    sea_uri = b64_image_data_uri("sea.png")
    if sea_uri:
        header_html = f"""
        <div id="top_icon_row">
          <img src="{sea_uri}" alt="sea icon" />
          <div>
            <div class="title">Rotated ROI by Line + Intensity + CNN + Report</div>
            <div class="subtitle"></div>
          </div>
        </div>
        """
    else:
        header_html = f"""
        <div id="top_icon_row">
          <div>
            <div class="title">Rotated ROI by Line + Intensity + CNN + Report</div>
            <div class="subtitle">(sea.png not found: place sea.png in the same directory as this script)</div>
          </div>
        </div>
        """

    with gr.Blocks(
        title="Rotated ROI by Line + Intensity + CNN + Report",
        theme=custom_theme,
        css=custom_css
    ) as demo:

        gr.HTML(header_html)
        gr.Markdown("### <span class='accent'>ROI processing\n")

        with gr.Row():
            files = gr.Files(label="Upload images (multiple)", file_types=["image"], file_count="multiple")
            idx = gr.Number(value=0, label="Current index (starts at 0)", interactive=True, precision=0)
            btn_prev = gr.Button("⬅ Previous", size="sm")
            btn_next = gr.Button("Next ➡", size="sm")

        state_paths = gr.State([])
        state_imgs  = gr.State([])
        state_curr  = gr.State(None)
        perpage     = gr.State({})

        with gr.Row():
            orig_view = gr.Image(
                label="Original (click TWO points)",
                interactive=False,
                sources=[]
            )
            orig_line_view = gr.Image(label="Original with line", interactive=False)

        # ✅ 已删除 ROI / Preprocess params 的 Accordion 和 sliders（UI 不再显示）

        with gr.Row():
            roi_preview_view = gr.Image(label="ROI preview (rotated crop)", interactive=False, elem_id="roi_preview_view")
            profile_img = gr.Image(label="Intensity (adaptive peaks)", interactive=False)

        with gr.Row():
            btn_cnn = gr.Button("🧠 CNN model predict", variant="primary")
            cnn_out = gr.Textbox(label="CNN predicted age", value="", interactive=False)
            btn_reset_roi = gr.Button("🔁 Re-select ROI", variant="secondary")

        with gr.Row():
            btn_export_pdf = gr.Button("📄 Export current page PDF", variant="secondary")
            btn_export_all = gr.Button("🗂️ Export all PDFs (multi-page)", variant="secondary")
            btn_export_csv = gr.Button("📊 Export all CSV (image, age)", variant="secondary")

        with gr.Row():
            pdf_file = gr.File(label="Download (current PDF)", interactive=False)
            pdf_all  = gr.File(label="Download (all PDFs)", interactive=False)
            csv_all  = gr.File(label="Download CSV (image, age)", interactive=False)

        info = gr.Markdown("<span class='muted'>👋 Ready.</span>")

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
                for i, _im in enumerate(imgs):
                    pstore[i] = make_empty_page()
                    if paths and i < len(paths):
                        pstore[i]["image_name"] = os.path.basename(paths[i])

            return (
                paths, imgs, curr, 0, pstore,
                cv.cvtColor(curr, cv.COLOR_BGR2RGB) if curr is not None else None,
                None, None, None, "",
                "<span class='accent'>✅ Images loaded.</span> <span class='muted'>Click TWO points on the original image.</span>"
            )

        files.upload(
            fn=on_files,
            inputs=[files],
            outputs=[
                state_paths, state_imgs, state_curr, idx, perpage,
                orig_view, orig_line_view, roi_preview_view, profile_img, cnn_out,
                info
            ]
        )

        def change_index(i, imgs, pstore):
            if not imgs:
                popup_warn("No images.")
                return None, None, None, None, "", None, 0, ""
            i = int(max(0, min(len(imgs) - 1, int(i))))
            cur = imgs[i]
            page = load_page_state(pstore, i)
            cnn_txt = "" if page.get("cnn_age", None) is None else str(int(page["cnn_age"]))
            return (
                cv.cvtColor(cur, cv.COLOR_BGR2RGB),
                page["orig_with_line"],
                page["roi_preview"],
                page["profile_img"],
                cnn_txt,
                cur, i,
                f"<span class='muted'>📌 Current index {i}. Restored state.</span>"
            )

        idx.change(
            fn=change_index,
            inputs=[idx, state_imgs, perpage],
            outputs=[orig_view, orig_line_view, roi_preview_view, profile_img, cnn_out, state_curr, idx, info]
        )

        btn_prev.click(
            fn=lambda i, imgs: int((int(i) - 1) % len(imgs)) if imgs else i,
            inputs=[idx, state_imgs],
            outputs=[idx]
        )
        btn_next.click(
            fn=lambda i, imgs: int((int(i) + 1) % len(imgs)) if imgs else i,
            inputs=[idx, state_imgs],
            outputs=[idx]
        )

        # ✅ 回调改为使用固定默认参数（UI删掉了，但谷底“叉叉”仍然会画：plt.plot(...,"rx") 不变）
        def on_click_orig_line(evt: gr.SelectData, curr_bgr, i, pstore, paths):
            if curr_bgr is None:
                popup_warn("Please upload images first.")
                return None, None, None, pstore, "", ""

            x, y = int(evt.index[0]), int(evt.index[1])
            page = load_page_state(pstore, int(i))

            # store original RGB (for report)
            page["orig_rgb"] = cv.cvtColor(curr_bgr, cv.COLOR_BGR2RGB)

            if paths and 0 <= int(i) < len(paths):
                page["image_name"] = os.path.basename(paths[int(i)])
            elif not page.get("image_name", ""):
                page["image_name"] = f"Page {int(i)}"

            pts = list(page["line_pts"] or [])
            if len(pts) >= 2:
                pts = []
            pts.append((x, y))

            if len(pts) == 1:
                vis = curr_bgr.copy()
                cv.circle(vis, pts[0], 4, (0, 0, 255), -1, cv.LINE_AA)
                vis_rgb = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
                page["line_pts"] = pts
                page["orig_with_line"] = vis_rgb
                page["roi_preview"] = None
                page["profile_img"] = None
                page["age"] = None
                page["cnn_age"] = None
                pstore[int(i)] = page
                return vis_rgb, None, None, pstore, "", f"<span class='muted'>🖱 Start: {pts[0]}. Click once more to set the end.</span>"

            p1, p2 = pts[0], pts[1]
            vis_bgr = draw_line(curr_bgr, p1, p2, (0, 0, 255), 2)
            vis_rgb = cv.cvtColor(vis_bgr, cv.COLOR_BGR2RGB)

            # --- NEW: CLAHE + normalize (like visualization()), keep the rest of the pipeline ---
            gray = cv.cvtColor(curr_bgr, cv.COLOR_BGR2GRAY)

            clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            eq = clahe.apply(gray)

            proc = cv.normalize(eq, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            # rotated_roi_from_line expects BGR, so convert back to 3 channels
            enh_bgr = cv.cvtColor(proc, cv.COLOR_GRAY2BGR)

            roi_rgb = rotated_roi_from_line(
                enh_bgr, p1, p2,
                roi_height_px=int(DEFAULT_ROI_HEIGHT_PX),
                out_w=int(DEFAULT_OUT_W),
                out_h=int(DEFAULT_OUT_H),
                pad_value=255
            )

            if roi_rgb is None:
                popup_warn("Segment too short. Please re-click two points.")
                page["line_pts"] = []
                page["orig_with_line"] = None
                page["roi_preview"] = None
                page["profile_img"] = None
                page["age"] = None
                page["cnn_age"] = None
                pstore[int(i)] = page
                return None, None, None, pstore, "", ""

            roi_gray = cv.cvtColor(roi_rgb, cv.COLOR_RGB2GRAY)
            roi_gray = cv.normalize(roi_gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            # --- EXACT ring detection + peak positions from visualization_integrate.py ---
            res = visualization(p1, p2, curr_bgr)  # uses CLAHE+normalize + line sampling + adaptive peaks on inverted

            profile_u8 = res["profile"].astype(np.uint8)      # along-line band profile (NOT ROI center)
            peaks = res["peaks"].astype(np.int32)             # indices into profile/xs/ys
            xs = res["xs"].astype(np.int32)                   # x coords along the original line
            ys = res["ys"].astype(np.int32)                   # y coords along the original line
            num_rings = int(res["num_rings"])

            # Draw red X on the ORIGINAL-with-line image at detected positions (xs[peaks], ys[peaks])
            vis_bgr2 = vis_bgr.copy()
            for pk in peaks:
                if 0 <= int(pk) < len(xs):
                    draw_red_x_bgr(vis_bgr2, int(xs[int(pk)]), int(ys[int(pk)]), size=7, thickness=2)
            vis_rgb = cv.cvtColor(vis_bgr2, cv.COLOR_BGR2RGB)

            # Plot INVERTED profile exactly like the peak logic (peaks are on inv)
            inv = 255.0 - profile_u8.astype(np.float32)

            fig = plt.figure(figsize=(6, 3), dpi=300)
            plt.plot(inv, label="Intensity (inverted, along line)")
            if len(peaks) > 0:
                plt.plot(peaks, inv[peaks], "rx", label="Peaks")
            plt.title(f"peaks={len(peaks)}")
            plt.xlabel("Index along line")
            plt.ylabel("Intensity (inverted)")
            plt.legend()
            plt.tight_layout(pad=2.0)

            img_profile = fig_to_rgb(fig)
            plt.close(fig)

            msg = (
                f"<span class='accent'>✅</span> <span class='muted'>Rotated ROI {int(DEFAULT_OUT_H)}×{int(DEFAULT_OUT_W)} | "
                f"roi_height_px={int(DEFAULT_ROI_HEIGHT_PX)} | rings={num_rings}</span>"
            )

            page["line_pts"] = pts
            page["orig_with_line"] = vis_rgb       # now includes crosses at detected ring positions
            page["roi_preview"] = roi_rgb          # keep rotated crop EXACTLY as before (for CNN etc)
            page["profile_img"] = img_profile
            page["age"] = num_rings                # intensity-age now matches visualization() exactly
            page["cnn_age"] = None
            pstore[int(i)] = page

            return vis_rgb, roi_rgb, img_profile, pstore, "", msg


        orig_view.select(
            fn=on_click_orig_line,
            inputs=[state_curr, idx, perpage, state_paths],
            outputs=[orig_line_view, roi_preview_view, profile_img, perpage, cnn_out, info]
        )

        def cnn_predict_current(i, pstore, progress=gr.Progress()):
            i = int(i)
            page = load_page_state(pstore, i)

            if page.get("roi_preview", None) is None:
                popup_warn("No ROI preview. Please select ROI first.")
                return pstore, "", ""

            progress(0.10, desc="Loading CNN model...")
            try:
                model, device = load_cnn_model_cached(CNN_WEIGHTS_PATH, CNN_DEVICE)
            except Exception as e:
                popup_warn(f"CNN load failed: {str(e)}")
                return pstore, "", ""

            progress(0.40, desc="Preprocessing ROI...")
            x = preprocess_roi_for_bettercnn(page["roi_preview"]).to(device)

            progress(0.75, desc="Running inference...")
            with torch.no_grad():
                y = model(x)
                pred = float(y.detach().cpu().view(-1)[0].item())

            progress(0.92, desc="Post-processing...")
            pred_int = int(round(pred))

            page["cnn_age"] = pred_int
            pstore[i] = page

            progress(1.0, desc="Done")
            return pstore, str(pred_int), "<span class='muted'>✅ CNN predicted and saved to page.</span>"

        btn_cnn.click(
            fn=cnn_predict_current,
            inputs=[idx, perpage],
            outputs=[perpage, cnn_out, info]
        )

        def reset_roi(i, pstore):
            i = int(i)
            page = load_page_state(pstore, i)
            page["line_pts"] = []
            page["orig_rgb"] = None
            page["orig_with_line"] = None
            page["roi_preview"] = None
            page["profile_img"] = None
            page["age"] = None
            page["cnn_age"] = None
            pstore[i] = page
            return pstore, None, None, None, "", "<span class='muted'>🔁 Cleared ROI. Click two points again.</span>"

        btn_reset_roi.click(
            fn=reset_roi,
            inputs=[idx, perpage],
            outputs=[perpage, orig_line_view, roi_preview_view, profile_img, cnn_out, info]
        )

        def export_pdf_current(i, pstore):
            page = load_page_state(pstore, int(i))
            if page["orig_with_line"] is None or page["roi_preview"] is None or page["profile_img"] is None:
                popup_warn("This page has no ROI results.")
                return None, ""

            fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
            gs = fig.add_gridspec(3, 2, height_ratios=[0.20, 1.0, 1.0], hspace=0.25, wspace=0.10)

            ax_title = fig.add_subplot(gs[0, :])
            ax_title.axis("off")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            img_name = page.get("image_name", f"Page {int(i)}")

            ax_title.text(0.01, 0.72, f"Page {int(i)} Rotated ROI & Intensity & CNN", fontsize=16, weight="bold")
            ax_title.text(0.01, 0.48, f"Image name:{img_name} | Generated at: {now}", fontsize=10)
            ax_title.text(
                0.01, 0.22,
                f"Line points: {page.get('line_pts', [])} | "
                f"intensity(peaks)={page.get('age', None)} | "
                f"age_perdict(years)={page.get('cnn_age', None)}",
                fontsize=10, va="top"
            )

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.set_title("Original")
            ax1.axis("off")
            img_orig = page.get("orig_rgb", None)
            if img_orig is None:
                img_orig = page["orig_with_line"]
            ax1.imshow(img_orig, interpolation="nearest")

            ax2 = fig.add_subplot(gs[1, 1])
            ax2.set_title("Original (with line)")
            ax2.axis("off")
            ax2.imshow(page["orig_with_line"], interpolation="nearest")

            ax3 = fig.add_subplot(gs[2, 0])
            ax3.set_title("ROI Preview (rotated crop)")
            ax3.axis("off")
            ax3.imshow(page["roi_preview"], interpolation="nearest")

            ax4 = fig.add_subplot(gs[2, 1])
            ax4.set_title("Intensity (adaptive peaks)")
            ax4.axis("off")
            ax4.imshow(page["profile_img"], interpolation="nearest")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf_path = tmp.name
            fig.savefig(pdf_path, dpi=300)
            plt.close(fig)
            return pdf_path, "<span class='muted'>✅ Current page PDF generated.</span>"

        btn_export_pdf.click(
            fn=export_pdf_current,
            inputs=[idx, perpage],
            outputs=[pdf_file, info]
        )

        def export_pdf_all(pstore, n_imgs):
            if n_imgs is None or n_imgs == 0:
                popup_warn("No images.")
                return None, ""

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            out_path = tmp.name

            with PdfPages(out_path) as pdf:
                for i in range(n_imgs):
                    page = load_page_state(pstore, i)
                    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
                    gs = fig.add_gridspec(3, 2, height_ratios=[0.20, 1.0, 1.0], hspace=0.25, wspace=0.10)

                    ax_title = fig.add_subplot(gs[0, :])
                    ax_title.axis("off")
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    img_name = page.get("image_name", f"Page {i}")

                    ax_title.text(0.01, 0.72, f"Page {i} Rotated ROI & Intensity & CNN", fontsize=16, weight="bold")
                    ax_title.text(0.01, 0.48, f"Image name:{img_name}, Generated at: {now}", fontsize=10)
                    ax_title.text(
                        0.01, 0.22,
                        f"Line points: {page.get('line_pts', [])} | "
                        f"intensity(peaks)={page.get('age', None)} | "
                        f"age_perdict(years)={page.get('cnn_age', None)}",
                        fontsize=10, va="top"
                    )

                    ax1 = fig.add_subplot(gs[1, 0])
                    ax1.set_title("Original")
                    ax1.axis("off")
                    img_orig = page.get("orig_rgb", None)
                    if img_orig is None:
                        img_orig = page.get("orig_with_line", None)
                    if img_orig is not None:
                        ax1.imshow(img_orig, interpolation="nearest")
                    else:
                        ax1.text(0.5, 0.5, "No image", ha="center", va="center")

                    ax2 = fig.add_subplot(gs[1, 1])
                    ax2.set_title("Original (with line)")
                    ax2.axis("off")
                    if page.get("orig_with_line", None) is not None:
                        ax2.imshow(page["orig_with_line"], interpolation="nearest")
                    else:
                        ax2.text(0.5, 0.5, "No ROI yet", ha="center", va="center")

                    ax3 = fig.add_subplot(gs[2, 0])
                    ax3.set_title("ROI Preview (rotated crop)")
                    ax3.axis("off")
                    if page.get("roi_preview", None) is not None:
                        ax3.imshow(page["roi_preview"], interpolation="nearest")
                    else:
                        ax3.text(0.5, 0.5, "No ROI", ha="center", va="center")

                    ax4 = fig.add_subplot(gs[2, 1])
                    ax4.set_title("Intensity (adaptive peaks)")
                    ax4.axis("off")
                    if page.get("profile_img", None) is not None:
                        ax4.imshow(page["profile_img"], interpolation="nearest")
                    else:
                        ax4.text(0.5, 0.5, "No profile", ha="center", va="center")

                    pdf.savefig(fig, bbox_inches=None)
                    plt.close(fig)

            return out_path, "<span class='muted'>✅ All pages PDF generated.</span>"

        btn_export_all.click(
            fn=lambda pstore, imgs: export_pdf_all(pstore, len(imgs) if imgs else 0),
            inputs=[perpage, state_imgs],
            outputs=[pdf_all, info]
        )

        def export_csv_all(pstore, paths):
            if not paths:
                popup_warn("No images.")
                return None, ""

            rows = []
            for i, p in enumerate(paths):
                page = load_page_state(pstore, i)
                age_int = page.get("age", None)
                age_cnn = page.get("cnn_age", None)
                if age_int is None and age_cnn is None:
                    continue
                rows.append([
                    os.path.basename(p),
                    "" if age_int is None else int(age_int),
                    "" if age_cnn is None else int(age_cnn),
                ])

            if not rows:
                popup_warn("No pages with computed ages.")
                return None, ""

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            csv_path = tmp.name
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "age_intensity", "age_cnn"])
                writer.writerows(rows)

            return csv_path, f"<span class='muted'>✅ Exported {len(rows)} rows to CSV (with age_cnn).</span>"

        btn_export_csv.click(
            fn=export_csv_all,
            inputs=[perpage, state_paths],
            outputs=[csv_all, info]
        )

    return demo


# =============================================================================
# Entry point for launching the Gradio demo server on localhost.
# =============================================================================
if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", theme=None)
