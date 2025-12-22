import cv2
import numpy as np
from scipy.signal import find_peaks

def visualization(p1, p2, img):
    #Line start and end
    x0, y0 = p1
    x1, y1 = p2

    #Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    proc = cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    display_image = proc

    #Sample points along line
    length = int(np.hypot(x1 - x0, y1 - y0))
    xs = np.linspace(x0, x1, length).astype(int)
    ys = np.linspace(y0, y1, length).astype(int)

    h, w = proc.shape
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[mask]
    ys = ys[mask]

    #Band profile
    band = 3
    half = band // 2

    profiles = []
    for o in range(-half, half + 1):
        yy = np.clip(ys + o, 0, h - 1)
        profiles.append(proc[yy, xs])

    profile = np.mean(np.stack(profiles, axis=0), axis=0).astype(np.uint8)

    #Full adaptive peak detection
    p = profile.astype(np.float32)

    dyn = np.percentile(p, 95) - np.percentile(p, 5)
    prom_lo = max(5.0, 0.30 * dyn)

    cand_peaks, _ = find_peaks(p, prominence=prom_lo, distance=2)

    if len(cand_peaks) >= 3:
        diffs = np.diff(cand_peaks)
        lo, hi = np.percentile(diffs, [20, 80])
        diffs_mid = diffs[(diffs >= lo) & (diffs <= hi)]

        if len(diffs_mid) > 0:
            typical = np.median(diffs_mid)
        else:
            typical = np.median(diffs)
    else:
        typical = 20

    L = len(p)
    scale = np.clip(300.0 / max(L, 1), 0.6, 1.4)
    min_dist = int(np.clip(0.9 * typical * scale, 20, 80))

    prom = max(15.0, 0.35 * dyn)

    inv = 255.0 - p
    noise = 1.4826 * np.median(np.abs(np.diff(inv) - np.median(np.diff(inv))))
    prom = max(prom, 4.0 * noise)
    hmin_inv = np.percentile(inv, 50)

    peaks, _ = find_peaks(
        inv,
        distance=min_dist,
        prominence=prom,
        height=hmin_inv,
        width=1
    )

    num_rings = len(peaks)

    #Return results
    return {
        #Number of rings found 
        "num_rings": num_rings,  
        #Intensity profile (blue curve)
        "profile": profile,
        #X-coordinates of the line
        "xs": xs,     
        #Y-coordinates of the line
        "ys": ys,
        #Peak indices (red markers)
        "peaks": peaks,   
        #Image
        "display_image": display_image  
    }
