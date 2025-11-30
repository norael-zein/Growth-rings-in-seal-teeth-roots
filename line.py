import os
import glob
import csv
import cv2
import numpy as np

# ===================== CONFIG =====================
IMAGE_DIR = "line"          # ROI image folder
MASK_DIR = "masks"          # Output mask folder
LABEL_CSV = "labels.csv"    # Format: image,age
EXTS = ["*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"]  # Supported image types
WINDOW_NAME = "Tree Ring Polyline Annotator"

os.makedirs(MASK_DIR, exist_ok=True)

# ===================== LOAD age FROM labels.csv =====================
# Format: image,age
name2age = {}
if os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print("CSV columns:", reader.fieldnames)

        for row in reader:
            # 1) image column
            fname_raw = row.get("image")
            if not fname_raw:
                continue

            # Normalize name
            fname_key = os.path.basename(fname_raw).strip().lower()
            base_key, _ = os.path.splitext(fname_key)

            # 2) age column
            age_raw = row.get("age")
            if age_raw is None or age_raw == "":
                continue

            try:
                age = float(age_raw)
            except Exception:
                print("Failed to parse age:", fname_raw, age_raw)
                continue

            # Save with extension and without extension
            name2age[fname_key] = age
            name2age[base_key] = age

    print(f"Loaded {len(name2age)} age entries from {LABEL_CSV}.")
else:
    print(f"Warning: {LABEL_CSV} not found. Age will not be shown.")


def get_age_for_image(img_path):
    """Find age for this image name from labels.csv"""
    fname = os.path.basename(img_path).strip().lower()
    base, _ = os.path.splitext(fname)

    if fname in name2age:
        return name2age[fname]
    if base in name2age:
        return name2age[base]
    return None


# ===================== COLLECT IMAGES =====================
images = []
for ext in EXTS:
    images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
images = sorted(images)

if not images:
    print(f"No images found in {IMAGE_DIR}. Check path and extensions.")
    exit(0)

current_index = 0

# Per-image state
current_image = None
display_image = None
polylines = []             # Finished polylines
current_polyline = []      # Polyline currently being drawn


# ===================== UTILS =====================
def mask_path_for_image(img_path):
    """Return path of mask file for this image."""
    name = os.path.basename(img_path)
    base, _ = os.path.splitext(name)
    return os.path.join(MASK_DIR, base + "_mask.png")


def load_image(index):
    """Load image at index. Does NOT load previous annotations."""
    global current_image, display_image, polylines, current_polyline

    img_path = images[index]
    print(f"\n=== Loading {index+1}/{len(images)}: {img_path} ===")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        return

    current_image = img
    polylines = []
    current_polyline = []

    redraw()


def redraw():
    """Redraw all polylines on display image."""
    global display_image
    if current_image is None:
        return

    disp = current_image.copy()

    # Draw finished polylines (green)
    for pl in polylines:
        pts = np.array(pl, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(disp, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
        for (x, y) in pl:
            cv2.circle(disp, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Draw current polyline (blue)
    if len(current_polyline) > 0:
        pts = np.array(current_polyline, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(disp, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        for (x, y) in current_polyline:
            cv2.circle(disp, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Age display (top-left only, no bottom bar)
    img_path = images[current_index]
    age = get_age_for_image(img_path)
    age_text = f"Age (rings): {age:.1f}" if age is not None else "Age: N/A"
    cv2.putText(disp, age_text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    display_image = disp
    cv2.imshow(WINDOW_NAME, display_image)


def save_mask_if_any():
    """
    If any polyline exists, save a mask image as xxx_mask.png.
    If no polyline drawn, do nothing.
    """
    img_path = images[current_index]
    name = os.path.basename(img_path)

    has_points = (len(polylines) > 0) or (len(current_polyline) > 0)
    if not has_points:
        print(f"No points drawn on {name}. Mask not saved.")
        return

    H, W = current_image.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # Combine all polylines
    all_lines = polylines.copy()
    if len(current_polyline) >= 2:
        all_lines.append(current_polyline)

    for pl in all_lines:
        pts = np.array(pl, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=3)

    mask_path = mask_path_for_image(img_path)
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved: {mask_path} ({len(all_lines)} lines)")


# ===================== MOUSE CALLBACK =====================
def mouse_callback(event, x, y, flags, param):
    global current_polyline, polylines

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polyline.append([int(x), int(y)])
        redraw()


# ===================== MAIN LOOP =====================
def main():
    global current_index, current_polyline, polylines

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    load_image(current_index)

    print("\nControls:")
    print("  Left-click: add point along ring")
    print("  ENTER: finish current polyline")
    print("  C: cancel current line")
    print("  Z: undo last finished line")
    print("  S: save mask (if lines exist)")
    print("  N: save mask and go to next image")
    print("  P: save mask and go to previous image")
    print("  K: skip image without saving")
    print("  Q or ESC: quit without saving current image\n")

    while True:
        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKey(50) & 0xFF

        if key == 13:  # Enter
            if len(current_polyline) >= 2:
                polylines.append(current_polyline.copy())
                print(f"Polyline completed ({len(current_polyline)} points). Now {len(polylines)} lines.")
            else:
                if len(current_polyline) > 0:
                    print("Polyline too short (<2 points). Ignored.")
            current_polyline = []
            redraw()

        elif key in (ord('c'), ord('C')):
            current_polyline = []
            print("Current line canceled.")
            redraw()

        elif key in (ord('z'), ord('Z')):
            if polylines:
                polylines.pop()
                print(f"Last polyline undone. Remaining: {len(polylines)}.")
                redraw()
            else:
                print("No polyline to undo.")

        elif key in (ord('s'), ord('S')):
            save_mask_if_any()

        elif key in (ord('n'), ord('N')):
            save_mask_if_any()
            if current_index < len(images) - 1:
                current_index += 1
                load_image(current_index)
            else:
                print("This is the last image.")

        elif key in (ord('p'), ord('P')):
            save_mask_if_any()
            if current_index > 0:
                current_index -= 1
                load_image(current_index)
            else:
                print("This is the first image.")

        elif key in (ord('k'), ord('K')):
            print("Skipping this image (no mask saved).")
            if current_index < len(images) - 1:
                current_index += 1
                load_image(current_index)
            else:
                print("This is the last image.")

        elif key == 27:  # ESC
            print("Exit without saving current image.")
            break

        elif key in (ord('q'), ord('Q')):
            print("Exit without saving current image.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
