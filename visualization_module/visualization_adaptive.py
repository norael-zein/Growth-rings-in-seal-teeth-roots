import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def load_age_annotations(csv_path):
    df = pd.read_csv(csv_path)
    age_dict = dict(zip(df["image_id"], df["age"]))
    return age_dict


def process_image(
    img_path,
    true_age=None,
    clip_limit=5.0,
    tile_grid_size=(8, 8),
    median_ksize=3,          # (ej använd längre)
    gauss_ksize=(5, 5),      # (ej använd längre)
    gauss_sigma=0.5,         # (ej använd längre)
    band=3,
    blackhat_kernel=21,
    blackhat_strength=2.0
):

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    eq = clahe.apply(gray)

    # --- Display image (baseline) ---
    norm_display = cv2.normalize(
        eq, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --- Processed image for analysis (black-hat, utan blur) ---
    k = int(blackhat_kernel)
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k, k)
    )

    blackhat = cv2.morphologyEx(
        eq, cv2.MORPH_BLACKHAT, kernel
    )

    enhanced = cv2.addWeighted(
        eq, 1.0,
        blackhat, blackhat_strength,
        0
    )

    norm_proc = cv2.normalize(
        enhanced, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Select line on ORIGINAL display image
    plt.figure(figsize=(8, 6))
    plt.imshow(norm_display, cmap="gray")
    plt.title("Specify start and end point of line")
    plt.axis("off")

    pts = plt.ginput(2)
    plt.close()

    if len(pts) < 2:
        print("Not enough points, proceeding to next image")
        return None

    (x0, y0), (x1, y1) = pts
    length = int(np.hypot(x1 - x0, y1 - y0))
    if length < 3:
        print("Line too short, proceeding to next image")
        return None

    # Sample line coordinates
    xs = np.linspace(x0, x1, length)
    ys = np.linspace(y0, y1, length)

    xs_int = xs.astype(int)
    ys_int = ys.astype(int)

    h, w = norm_proc.shape
    mask = (
        (xs_int >= 0) & (xs_int < w) &
        (ys_int >= 0) & (ys_int < h)
    )
    xs_int = xs_int[mask]
    ys_int = ys_int[mask]

    # --- Band profile (average across thickness) ---
    b = int(band)
    if b % 2 == 0:
        b += 1
    half = b // 2

    profiles = []
    for o in range(-half, half + 1):
        yy = np.clip(ys_int + o, 0, h - 1)
        profiles.append(norm_proc[yy, xs_int])

    profile = np.mean(
        np.stack(profiles, axis=0),
        axis=0
    ).astype(np.uint8)

    # Adaptive peak detection
    p = profile.astype(np.float32)

    dyn = np.percentile(p, 95) - np.percentile(p, 5)
    prom_lo = max(5.0, 0.30 * dyn)

    cand_peaks, _ = find_peaks(
        p, prominence=prom_lo, distance=2
    )

    if len(cand_peaks) >= 3:
        diffs = np.diff(cand_peaks)
        lo, hi = np.percentile(diffs, [20, 80])
        diffs_mid = diffs[(diffs >= lo) & (diffs <= hi)]
        typical = (
            np.median(diffs_mid)
            if len(diffs_mid) > 0
            else np.median(diffs)
        )
    else:
        typical = 20

    L = len(p)
    scale = np.clip(300.0 / max(L, 1), 0.6, 1.4)
    min_dist = int(np.clip(0.9 * typical * scale, 20, 80))

    prom = max(15.0, 0.35 * dyn)

    inv = 255.0 - p
    noise = 1.4826 * np.median(
        np.abs(np.diff(inv) - np.median(np.diff(inv)))
    )
    prom = max(prom, 4.0 * noise)
    hmin_inv = np.percentile(inv, 50)

    peaks, _ = find_peaks(
        inv,
        distance=min_dist,
        prominence=prom,
        height=hmin_inv,
        width=1
    )

    pred_age = len(peaks)

    peak_xs = xs_int[peaks]
    peak_ys = ys_int[peaks]

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(norm_display, cmap="gray")
    ax1.plot([x0, x1], [y0, y1], "r--", linewidth=1)
    if len(peaks) > 0:
        ax1.plot(peak_xs, peak_ys, "rx", markersize=6, mew=2)

    title = os.path.basename(img_path)
    if true_age is not None:
        title += f"\nPredicted: {pred_age} | True: {true_age}"
    else:
        title += f"\nPredicted: {pred_age} (true missing)"
    ax1.set_title(title)
    ax1.axis("off")

    ax2.plot(profile, label="Intensity profile")
    if len(peaks) > 0:
        ax2.plot(peaks, profile[peaks], "rx", label="Detected peaks")
    ax2.invert_yaxis()
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return pred_age


def main():
    image_dir = "../images_original"
    annot_path = "../annotations_original.csv"

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg")
    )

    age_dict = load_age_annotations(annot_path)

    for file in image_files:
        img_path = os.path.join(image_dir, file)
        image_id = os.path.splitext(file)[0]
        true_age = age_dict.get(image_id)

        process_image(img_path, true_age=true_age)


if __name__ == "__main__":
    main()
