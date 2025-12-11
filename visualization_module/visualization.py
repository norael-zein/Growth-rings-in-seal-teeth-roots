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


def process_image(img_path, true_age=None, clip_limit=2.0, tile_grid_size=(8, 8)):

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #CLAHE 
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    eq = clahe.apply(gray)

    #Normalize
    norm = cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #Select line
    plt.figure(figsize=(8, 6))
    plt.imshow(norm, cmap="gray")
    plt.title("Specify start and end point of line")
    plt.axis("off")

    pts = plt.ginput(2)
    plt.close()

    if len(pts) < 2:
        print("Not enough points, proceeding to next image")
        return None

    (x0, y0), (x1, y1) = pts
    length = int(np.hypot(x1 - x0, y1 - y0))

    #Sample line
    xs = np.linspace(x0, x1, length)
    ys = np.linspace(y0, y1, length)

    xs_int = xs.astype(int)
    ys_int = ys.astype(int)

    h, w = norm.shape
    mask = (xs_int >= 0) & (xs_int < w) & (ys_int >= 0) & (ys_int < h)
    xs_int = xs_int[mask]
    ys_int = ys_int[mask]

    profile = norm[ys_int, xs_int]

    #Peak detection
    peaks, _ = find_peaks(profile, distance=10, prominence=50)
    pred_age = len(peaks)

    #Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(norm, cmap="gray")
    ax1.plot([x0, x1], [y0, y1], "r--", linewidth=1)

    title = os.path.basename(img_path)
    if true_age is not None:
        title += f"\nPredicted: {pred_age} years | True: {true_age} years"
    else:
        title += f"\nPredicted: {pred_age} years (true age missing)"
    ax1.set_title(title)
    ax1.axis("off")

    ax2.plot(profile, label="Intensity profile")
    if len(peaks) > 0:
        ax2.plot(peaks, profile[peaks], "rx", label="Peaks")
    ax2.set_xlabel("Position along line")
    ax2.set_ylabel("Intensity")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return pred_age


def main():
    image_dir = "../images_original"
    annot_path = "../annotations_original.csv"

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    image_files.sort()
    print(f"Number of images: {len(image_files)}")

    age_dict = load_age_annotations(annot_path)

    for file in image_files:
        img_path = os.path.join(image_dir, file)

        image_id = os.path.splitext(file)[0]
        true_age = age_dict.get(image_id, None)

        print(f"\nProcessing: {file}")
        process_image(img_path, true_age=true_age)


if __name__ == "__main__":
    main()
