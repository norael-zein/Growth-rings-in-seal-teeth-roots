# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import glob
import os
from typing import List, Tuple

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

# --- Robust Matplotlib backend selection (before importing pyplot) ---
import matplotlib as mpl
import importlib
BACKEND = 'Agg'
try:
    if importlib.util.find_spec('PyQt5') is not None or importlib.util.find_spec('PySide2') is not None:
        mpl.use('Qt5Agg'); BACKEND = 'Qt5Agg'
    else:
        import tkinter  # noqa: F401
        mpl.use('TkAgg'); BACKEND = 'TkAgg'
except Exception:
    mpl.use('Agg'); BACKEND = 'Agg'

import matplotlib.pyplot as plt
import io
# Buttons & layout (for interactive backends)
try:
    from matplotlib import gridspec
    from matplotlib.widgets import Button
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False


# ---------------------- Processing pipeline ----------------------
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
        V_min = int(np.min(V[blue_mask > 0])); V_sub = V[blue_mask > 0]
    else:
        V_min = int(np.min(V)); V_sub = V.reshape(-1)

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
        delta_val = int(np.clip(delta_val, lo, hi))
        return delta_val

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


# ---------------------- Sampling & analysis ----------------------
def sample_line_profile(img_gray: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], width: int = 7):
    x1, y1 = float(p1[0]), float(p1[1]); x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1; dy = y2 - y1
    length = int(np.hypot(dx, dy))
    if length < 2: return None, None
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
    vis = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    cv.line(vis, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 1, cv.LINE_AA)
    return profile, vis


def analyze_profile(profile: np.ndarray, peak_distance: int = 10, peak_prominence: int = 50):
    p = profile.astype(np.float32)
    pz = p - p.mean()
    peaks, _ = find_peaks(p, distance=peak_distance, prominence=peak_prominence)
    period_pix = np.mean(np.diff(peaks)) if len(peaks) > 1 else np.nan
    F = np.fft.rfft(pz); mag = np.abs(F)
    freqs = np.fft.rfftfreq(pz.size, d=1.0)
    if mag.size > 1:
        idx_main = 1 + np.argmax(mag[1:])
        main_freq = freqs[idx_main]
        main_period = (1.0 / main_freq) if main_freq > 1e-9 else np.inf
    else:
        main_freq, main_period = 0.0, np.inf
    return peaks, period_pix, freqs, mag, main_freq, main_period


# ---------------------- Matplotlib helper ----------------------
def _show_current_plot(window_name: str = 'Analysis'):
    if BACKEND in ('Qt5Agg', 'TkAgg'):
        plt.show()
        try: plt.pause(0.1)
        except Exception: pass
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
        if img is not None:
            cv.imshow(window_name, img)
            cv.waitKey(1)


# ---------------------- ROI selection (fixed 1000x1000) ----------------------
def ensure_min_size_1000(img: np.ndarray, box: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    pad_bottom = max(0, box - h); pad_right = max(0, box - w)
    if pad_bottom or pad_right:
        img = cv.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    return img

def select_fixed_roi(img_bgr: np.ndarray, box: int = 1000):
    """
    Drag a fixed 1000x1000 box; confirm on mouse release.
    Returns (x0, y0, x1, y1), 'NEXT' to skip image, or None to quit.
    """
    img = img_bgr.copy()
    H, W = img.shape[:2]
    half = box // 2

    win = 'ROI - drag to move 1000x1000; release to confirm (r reset | n next | q/ESC quit)'
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    cx, cy = W // 2, H // 2
    x0 = max(0, min(W - box, cx - half))
    y0 = max(0, min(H - box, cy - half))

    dragging = False
    confirmed = False

    def clamp_top_left(x, y):
        return max(0, min(W - box, x)), max(0, min(H - box, y))

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, x0, y0, confirmed
        if event == cv.EVENT_LBUTTONDOWN:
            dragging = True
            param['offset'] = (x - x0, y - y0)
        elif event == cv.EVENT_MOUSEMOVE and dragging:
            ox, oy = param.get('offset', (half, half))
            nx = x - ox; ny = y - oy
            x0, y0 = clamp_top_left(nx, ny)
        elif event == cv.EVENT_LBUTTONUP and dragging:
            dragging = False
            confirmed = True

    cb_param = {}
    cv.setMouseCallback(win, on_mouse, cb_param)

    while True:
        disp = img.copy()
        cv.rectangle(disp, (x0, y0), (x0 + box, y0 + box), (0, 255, 255), 2)
        cv.putText(disp, "Drag to move fixed 1000x1000 box; release to confirm",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(disp, "Drag to move fixed 1000x1000 box; release to confirm",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv.LINE_AA)
        cv.imshow(win, disp)
        key = cv.waitKey(20) & 0xFF
        if key in (27, ord('q')):
            cv.destroyWindow(win); return None
        elif key == ord('r'):
            x0 = max(0, min(W - box, W // 2 - half))
            y0 = max(0, min(H - box, H // 2 - half))
        elif key == ord('n'):
            cv.destroyWindow(win); return 'NEXT'
        if confirmed:
            cv.destroyWindow(win)
            return (x0, y0, x0 + box, y0 + box)


# ---------------------- Main loop ----------------------
def run(images: List[str], thickness: int = 7, peak_distance: int = 10, peak_prominence: int = 50):
    try: plt.ion()
    except Exception: pass

    idx = 0; n = len(images)
    if n == 0:
        print("No images found. Please check --images."); return

    instruction_line = (
        'LMB press=start; drag to set line; release=end | r reset line | +/- thickness | d/D peak_distance | p/P peak_prominence | n next image | b reselect ROI | u upload images | q/ESC quit'
    )
    status_msg = ''

    while True:
        path = images[idx]
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            print(f"Cannot read: {path}")
            idx = (idx + 1) % n; continue

        BOX = 1000
        img_padded = ensure_min_size_1000(img, box=BOX)

        # Step 1: ROI selection
        roi_sel = select_fixed_roi(img_padded, box=BOX)
        if roi_sel is None:
            cv.destroyAllWindows(); return
        if roi_sel == 'NEXT':
            idx = (idx + 1) % n; continue
        x0, y0, x1, y1 = roi_sel

        # Step 2: processing + crop
        final_full, _, _ = describe_pipeline(img_padded)
        orig_roi  = img_padded[y0:y1, x0:x1].copy()
        final_roi = final_full[y0:y1, x0:x1].copy()

        # grayscale for sampling (internal only)
        gray_base = cv.cvtColor(final_roi, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
        eq = clahe.apply(gray_base)
        norm = cv.normalize(eq, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        # Step 3: draw line on processed ROI
        win_line = 'Result (ROI) - Draw line (drag to set)'
        cv.namedWindow(win_line, cv.WINDOW_NORMAL)
        dragging = False; p_start = None; p_end = None; pts: List[Tuple[int, int]] = []

        def on_mouse(event, x, y, flags, param):
            nonlocal dragging, p_start, p_end, pts
            if event == cv.EVENT_LBUTTONDOWN:
                dragging = True; p_start = (x, y); p_end = None; pts = []
            elif event == cv.EVENT_MOUSEMOVE and dragging:
                p_end = (x, y)
            elif event == cv.EVENT_LBUTTONUP and dragging:
                dragging = False; p_end = (x, y)
                if p_start is not None and p_end is not None and (p_start != p_end):
                    pts = [p_start, p_end]

        cv.setMouseCallback(win_line, on_mouse)

        # flags for control flow
        reselect_roi = False
        upload_new = False
        new_paths: List[str] = []

        while True:
            disp = final_roi.copy()
            cv.putText(disp, f'{instruction_line}', (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(disp, f'{instruction_line}', (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            if dragging and (p_start is not None) and (p_end is not None):
                cv.line(disp, p_start, p_end, (0, 0, 255), 1, cv.LINE_AA)
            elif len(pts) == 2:
                cv.line(disp, pts[0], pts[1], (0, 0, 255), 1, cv.LINE_AA)

            if status_msg:
                cv.putText(disp, status_msg, (10, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
                cv.putText(disp, status_msg, (10, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv.LINE_AA)

            cv.imshow(win_line, disp)
            key = cv.waitKey(20) & 0xFF

            if key in (27, ord('q')):   # Exit
                cv.destroyAllWindows(); return
            elif key == ord('r'):       # Reset line
                pts = []
            elif key in (ord('+'), ord('=')):
                thickness = min(thickness + 2, 99)
            elif key in (ord('-'), ord('_')):
                thickness = max(thickness - 2, 1)
            elif key == ord('d'):
                peak_distance = max(1, peak_distance - 1)
            elif key == ord('D'):
                peak_distance = peak_distance + 1
            elif key == ord('p'):
                peak_prominence = max(0, peak_prominence - 5)
            elif key == ord('P'):
                peak_prominence = peak_prominence + 5
            elif key == ord('n'):       # Next image (from current list)
                pts = []; idx = (idx + 1) % n; cv.destroyWindow(win_line); break
            elif key == ord('b'):       # Reselect ROI on same image
                pts = []; cv.destroyWindow(win_line); reselect_roi = True; break
            elif key == ord('u'):       # Upload/select other images
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk(); root.withdraw()
                    filetypes = [('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('All files', '*.*')]
                    sel = filedialog.askopenfilenames(title='Select images', filetypes=filetypes)
                    new_paths = list(sel)
                    root.destroy()
                except Exception:
                    new_paths = []
                if new_paths:
                    upload_new = True
                    cv.destroyWindow(win_line)
                    break

            # ---- When two points set: compute + 2x2 UI ----
            if len(pts) >= 2:
                p1, p2 = pts[0], pts[1]
                profile, _ = sample_line_profile(norm, p1, p2, width=thickness)
                if profile is None or len(profile) == 0:
                    status_msg = 'Line is too short or out-of-bounds; draw a longer line.'
                    print(status_msg); pts = []; continue

                mean_val = float(np.mean(profile))
                std_val  = float(np.std(profile))
                status_msg = f'profile_len={len(profile)} mean={mean_val:.1f} std={std_val:.1f}'
                print(status_msg)
                peaks, period_pix, freqs, mag, main_freq, main_period = analyze_profile(profile, peak_distance, peak_prominence)
                print(f"peaks={len(peaks)} avg_period≈{period_pix:.1f}px | main_freq≈{main_freq:.4f} cyc/px -> period≈{main_period:.1f}px")

                # Overlays for both images
                proc_with_line = final_roi.copy()
                orig_with_line = orig_roi.copy()
                cv.line(proc_with_line, p1, p2, (0, 0, 255), 1, cv.LINE_AA)
                cv.line(orig_with_line, p1, p2, (0, 0, 255), 1, cv.LINE_AA)

                # ---- Result UI: buttons if interactive; otherwise static figure ----
                if BACKEND in ('Qt5Agg', 'TkAgg') and HAS_WIDGETS:
                    fig = plt.figure(figsize=(14, 10))
                    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.14])

                    ax_img_proc = fig.add_subplot(gs[0, 0])
                    ax_img_orig = fig.add_subplot(gs[1, 0])
                    ax_profile  = fig.add_subplot(gs[0, 1])
                    ax_spectrum = fig.add_subplot(gs[1, 1])

                    # Bottom row: three buttons (Exit / Reselect / Upload)
                    ax_btn_exit = fig.add_subplot(gs[2, 0])
                    ax_btn_area = fig.add_subplot(gs[2, 1])
                    # Split right-bottom area into two halves for Reselect/Upload
                    bbox = ax_btn_area.get_position()
                    left = plt.axes([bbox.x0, bbox.y0, (bbox.x1 - bbox.x0)/2, bbox.height])
                    right = plt.axes([bbox.x0 + (bbox.x1 - bbox.x0)/2, bbox.y0, (bbox.x1 - bbox.x0)/2, bbox.height])

                    # Left column images
                    ax_img_proc.imshow(cv.cvtColor(proc_with_line, cv.COLOR_BGR2RGB))
                    ax_img_proc.set_title('Processed ROI (1000x1000) with line'); ax_img_proc.axis('off')

                    ax_img_orig.imshow(cv.cvtColor(orig_with_line, cv.COLOR_BGR2RGB))
                    ax_img_orig.set_title('Original ROI (1000x1000) with line'); ax_img_orig.axis('off')

                    # Right column plots
                    ax_profile.plot(profile, label='Intensity profile')
                    if len(peaks) > 0:
                        ax_profile.plot(peaks, profile[peaks], 'rx', label='Peaks')
                    ttl = f"Profile: peaks={len(peaks)}"
                    if not np.isnan(period_pix):
                        ttl += f"  avg period≈{period_pix:.1f} px"
                    ax_profile.set_title(ttl)
                    ax_profile.set_xlabel('Position along line (px)')
                    ax_profile.set_ylabel('Intensity')
                    ax_profile.legend()

                    ax_spectrum.plot(freqs, mag)
                    ax_spectrum.set_title(f"Spectrum: main freq≈{main_freq:.4f} cyc/px  → period≈{main_period:.1f} px")
                    ax_spectrum.set_xlabel('Frequency (cycles/pixel)')
                    ax_spectrum.set_ylabel('Magnitude')

                    # Buttons
                    btn_exit = Button(ax_btn_exit, 'Exit')
                    btn_reselect = Button(left, 'Reselect ROI')
                    btn_upload = Button(right, 'Upload Images')

                    ui_choice = {'val': None, 'new_paths': []}
                    def _on_exit(event):
                        ui_choice['val'] = 'quit'; plt.close(fig)
                    def _on_reselect(event):
                        ui_choice['val'] = 'reselect'; plt.close(fig)
                    def _on_upload(event):
                        # File dialog
                        try:
                            import tkinter as tk
                            from tkinter import filedialog
                            root = tk.Tk(); root.withdraw()
                            filetypes = [('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('All files', '*.*')]
                            sel = filedialog.askopenfilenames(title='Select images', filetypes=filetypes)
                            ui_choice['new_paths'] = list(sel)
                            root.destroy()
                        except Exception:
                            ui_choice['new_paths'] = []
                        ui_choice['val'] = 'upload'
                        plt.close(fig)

                    btn_exit.on_clicked(_on_exit)
                    btn_reselect.on_clicked(_on_reselect)
                    btn_upload.on_clicked(_on_upload)

                    plt.tight_layout()
                    plt.show(block=False)
                    # Wait for a button to close the figure
                    while plt.fignum_exists(fig.number) and ui_choice['val'] is None:
                        try: plt.pause(0.1)
                        except Exception: break

                    choice = ui_choice['val']
                    if choice == 'quit':
                        cv.destroyAllWindows(); return
                    elif choice == 'reselect':
                        cv.destroyWindow(win_line); reselect_roi = True; break
                    elif choice == 'upload' and ui_choice['new_paths']:
                        new_paths = ui_choice['new_paths']; upload_new = True
                        cv.destroyWindow(win_line); break
                    else:
                        # No action or empty selection -> allow drawing another line
                        pts = []; continue
                else:
                    # Fallback: static figure; use hotkeys in the line window (b/u/q)
                    plt.figure(figsize=(14, 10))
                    plt.subplot(2, 2, 1)
                    plt.imshow(cv.cvtColor(proc_with_line, cv.COLOR_BGR2RGB))
                    plt.title('Processed ROI (1000x1000) with line'); plt.axis('off')

                    plt.subplot(2, 2, 3)
                    plt.imshow(cv.cvtColor(orig_with_line, cv.COLOR_BGR2RGB))
                    plt.title('Original ROI (1000x1000) with line'); plt.axis('off')

                    plt.subplot(2, 2, 2)
                    plt.plot(profile, label='Intensity profile')
                    if len(peaks) > 0:
                        plt.plot(peaks, profile[peaks], 'rx', label='Peaks')
                    ttl = f"Profile: peaks={len(peaks)}"
                    if not np.isnan(period_pix):
                        ttl += f"  avg period≈{period_pix:.1f} px"
                    plt.title(ttl); plt.xlabel('Position (px)'); plt.ylabel('Intensity'); plt.legend()

                    plt.subplot(2, 2, 4)
                    plt.plot(freqs, mag)
                    plt.title(f"Spectrum: main freq≈{main_freq:.4f} cyc/px  → period≈{main_period:.1f} px")
                    plt.xlabel('Frequency (cycles/pixel)'); plt.ylabel('Magnitude')
                    plt.tight_layout()
                    _show_current_plot('Analysis (static; use keys in line window)')
                    # Allow drawing another line
                    pts = []
                    continue

            # Flow after line-window loop
            if reselect_roi:
                # Reselect ROI on the same image
                break
            if upload_new and new_paths:
                # Switch to newly chosen images
                images[:] = new_paths
                idx = 0; n = len(images)
                break
            # For "next image", idx already advanced and line window closed

        # back to outer while to load next image / reselect ROI / new images

# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser(description='Interactive: fixed 1000x1000 ROI -> line profile + frequency analysis (with Exit/Reselect/Upload buttons)')
    ap.add_argument('--images', type=str, default=None, help='Path or wildcard, e.g. "./imgs/*.jpg"; if omitted, a file dialog opens')
    ap.add_argument('--thickness', type=int, default=7, help='Line thickness (pixels averaged along normal)')
    ap.add_argument('--peak-distance', type=int, default=10, help='Minimum distance between peaks (pixels)')
    ap.add_argument('--peak-prominence', type=int, default=50, help='Peak prominence threshold')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    paths: List[str] = []
    if args.images:
        if any(ch in args.images for ch in ['*', '?', '[']):
            paths = sorted(glob.glob(args.images))
        else:
            paths = [args.images]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            filetypes = [('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('All files', '*.*')]
            sel = filedialog.askopenfilenames(title='Select images to analyze', filetypes=filetypes)
            paths = list(sel); root.destroy()
        except Exception:
            print('No --images and cannot open file dialog. Please use --images with a path or wildcard.')
            paths = []

    if not paths:
        print('No images found. Example: --images "./images/*.jpg"')
    else:
        run(paths, thickness=args.thickness, peak_distance=args.peak_distance, peak_prominence=args.peak_prominence)
