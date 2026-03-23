"""
src/signal_extractor.py
=======================
Computer-Vision pipeline that digitizes a 12-lead ECG photograph and converts it
into a normalized (1000, 12) NumPy array ready for DenseNet1D inference.

Pipeline stages
---------------
1. Color masking & grid removal   – strip red/pink grid lines; keep signal pixels.
2. 12-lead segmentation           – split the 4 × 3 waveform region into 12 ROIs.
3. 1-D signal extraction          – column-wise median of white pixels, inverted Y.
4. Resampling & normalization     – interp1d → 1 000 pts, Z-score per lead.

CLI usage
---------
    python src/signal_extractor.py path/to/ecg_image.jpg
    python src/signal_extractor.py path/to/ecg_image.jpg --out my_output.npy --debug

Output
------
    extracted_ecg_signal.npy   shape (1000, 12), dtype float32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Lead name order (standard 4 × 3 layout, row-major)
# ---------------------------------------------------------------------------
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ---------------------------------------------------------------------------
# Stage 1 – load, crop header, isolate signal pixels
# ---------------------------------------------------------------------------

def _build_signal_mask(bgr_img: np.ndarray) -> np.ndarray:
    """
    Adaptive signal isolation: returns a binary mask where 255 = ECG waveform
    pixel and 0 = background / grid.

    Auto-detection
    --------------
    We inspect the HSV saturation channel.  If the image is predominantly
    low-saturation (≥70 % of pixels have S < 30), it is a monochrome
    black-on-white printout → strategy A.  Otherwise it has a colored grid
    (pink/red) → strategy B.

    Strategy A  (monochrome black-on-white ECG)
    -------------------------------------------
    1. Convert to grayscale.
    2. Apply Gaussian blur to reduce halftone / JPEG noise.
    3. Otsu threshold → binary image where dark pixels = signal.
    4. Morphological opening (remove speckles) + closing (fill trace gaps).

    Strategy B  (colored-grid ECG)
    --------------------------------
    1. Build HSV mask for red/pink hues (the grid).
    2. Also mask near-white background.
    3. Invert: remaining dark, non-red pixels = signal.
    4. Same morphological cleanup.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # ── auto-detect image type ────────────────────────────────────────────
    low_sat_fraction = float(np.mean(saturation < 30))
    monochrome = low_sat_fraction >= 0.70   # mostly grey/white → strategy A

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if monochrome:
        # ── Strategy A: dark trace on light background ────────────────────
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Mild blur to reduce halftone dots / scan noise before thresholding
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold handles uneven illumination across the scan
        bin_img = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=8,
        )

        # Remove grid lines: they tend to be long horizontal/vertical streaks.
        # Erode horizontally then vertically to find grid-line candidates and
        # suppress them from the signal mask.
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines  = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, h_kernel)
        v_lines  = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, v_kernel)
        grid_lines = cv2.bitwise_or(h_lines, v_lines)
        signal_mask = cv2.bitwise_and(bin_img, cv2.bitwise_not(grid_lines))

    else:
        # ── Strategy B: colored (red/pink) grid ──────────────────────────
        grid_lo1 = np.array([0,   30,  80], dtype=np.uint8)
        grid_hi1 = np.array([15, 255, 255], dtype=np.uint8)
        grid_lo2 = np.array([160, 30,  80], dtype=np.uint8)
        grid_hi2 = np.array([179, 255, 255], dtype=np.uint8)
        grid_mask = (
            cv2.inRange(hsv, grid_lo1, grid_hi1) |
            cv2.inRange(hsv, grid_lo2, grid_hi2)
        )

        white_lo   = np.array([0,   0, 200], dtype=np.uint8)
        white_hi   = np.array([179, 40, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_lo, white_hi)

        noise_mask  = cv2.bitwise_or(grid_mask, white_mask)
        signal_mask = cv2.bitwise_not(noise_mask)

    # ── shared morphological cleanup ─────────────────────────────────────
    signal_mask = cv2.morphologyEx(signal_mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    signal_mask = cv2.morphologyEx(signal_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return signal_mask


def load_and_preprocess(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the ECG image, crop away the text header (top 20 %), and build the
    binary signal mask.

    Returns
    -------
    grid_bgr   : cropped BGR image  (for optional debug visualisation)
    signal_mask: binary mask, same spatial size as grid_bgr
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h, w = img.shape[:2]
    crop_top = int(h * 0.20)          # skip top 20 % (patient header text)
    grid_bgr = img[crop_top:, :]

    signal_mask = _build_signal_mask(grid_bgr)
    return grid_bgr, signal_mask


# ---------------------------------------------------------------------------
# Stage 2 – 12-lead segmentation  (4 columns × 3 rows)
# ---------------------------------------------------------------------------

def segment_leads(signal_mask: np.ndarray, n_cols: int = 4, n_rows: int = 3) -> list[np.ndarray]:
    """
    Split the cleaned signal mask into 12 ROIs using a 4 × 3 grid.

    The bottom rhythm strip (common in ECG printouts) is excluded; we only
    divide the *upper* n_rows × n_cols region.

    Returns
    -------
    List of 12 binary ROI arrays, in LEAD_NAMES order:
        [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
    """
    total_h, total_w = signal_mask.shape[:2]

    # Heuristic: if there is a bottom rhythm strip it takes ≈25 % of height.
    # Shrink the search region so we don't bleed into it.
    waveform_h = int(total_h * 0.75)
    waveform_region = signal_mask[:waveform_h, :]

    box_h = waveform_h  // n_rows
    box_w = total_w     // n_cols

    rois: list[np.ndarray] = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * box_h
            y1 = y0 + box_h
            x0 = col * box_w
            x1 = x0 + box_w
            rois.append(waveform_region[y0:y1, x0:x1])

    return rois   # length == 12


# ---------------------------------------------------------------------------
# Stage 3 – 1-D signal extraction per ROI
# ---------------------------------------------------------------------------

def extract_1d_signal(roi: np.ndarray) -> np.ndarray:
    """
    Convert a binary ROI (H × W) into a 1-D signal of length W.

    Algorithm
    ---------
    * For each column x, gather y-coordinates of white (signal) pixels.
    * Take the median as the representative amplitude at x.
    * Forward-fill missing columns (no white pixels → use last valid value).
    * Invert Y so that waveform orientation matches real-world voltage polarity.

    Returns
    -------
    signal : np.ndarray, shape (W,), dtype float64
    """
    roi_h, roi_w = roi.shape[:2]
    signal = np.full(roi_w, np.nan, dtype=np.float64)

    last_valid: float = roi_h / 2.0   # fallback: midline

    for x in range(roi_w):
        col = roi[:, x]
        ys = np.where(col > 127)[0]
        if ys.size > 0:
            val = float(np.median(ys))
            signal[x] = val
            last_valid = val
        else:
            signal[x] = last_valid   # carry-forward

    # Invert Y axis: image Y=0 is top; ECG amplitude should be positive upward
    signal = roi_h - signal

    return signal


# ---------------------------------------------------------------------------
# Stage 4 – Resampling & Z-score normalization
# ---------------------------------------------------------------------------

def resample_signal(signal: np.ndarray, target_len: int = 1000) -> np.ndarray:
    """
    Resample a 1-D signal of arbitrary length to *target_len* using linear
    interpolation (scipy.interpolate.interp1d).
    """
    src_len = len(signal)
    if src_len == target_len:
        return signal.astype(np.float32)

    x_src = np.linspace(0.0, 1.0, src_len)
    x_dst = np.linspace(0.0, 1.0, target_len)
    interp_fn = interp1d(x_src, signal, kind="linear", fill_value="extrapolate")
    return interp_fn(x_dst).astype(np.float32)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D signal in-place (returns a new array)."""
    mean = signal.mean()
    std  = signal.std()
    if std < 1e-8:
        return np.zeros_like(signal)
    return ((signal - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_ecg_array(
    image_path: str | Path,
    out_path:   str | Path = "extracted_ecg_signal.npy",
    target_len: int = 1000,
    debug: bool = False,
) -> np.ndarray:
    """
    Full pipeline: ECG photograph → normalized (1000, 12) NumPy array.

    Parameters
    ----------
    image_path : path to input ECG image (PNG / JPEG / BMP …)
    out_path   : where to save the .npy output
    target_len : number of time-steps per lead (default 1000)
    debug      : if True, write intermediate images alongside the output

    Returns
    -------
    ecg_array : np.ndarray, shape (target_len, 12), dtype float32
    """
    image_path = Path(image_path)
    out_path   = Path(out_path)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print(f"[1/4] Loading & masking  →  {image_path}")
    grid_bgr, signal_mask = load_and_preprocess(image_path)

    if debug:
        dbg_dir = out_path.parent / "signal_extractor_debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_dir / "01_grid_crop.png"), grid_bgr)
        cv2.imwrite(str(dbg_dir / "02_signal_mask.png"), signal_mask)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("[2/4] Segmenting 12 leads …")
    rois = segment_leads(signal_mask)
    assert len(rois) == 12, f"Expected 12 ROIs, got {len(rois)}"

    if debug:
        for i, roi in enumerate(rois):
            cv2.imwrite(str(dbg_dir / f"03_roi_{LEAD_NAMES[i]}.png"), roi)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("[3/4] Extracting 1-D signals …")
    raw_signals: list[np.ndarray] = []
    for i, roi in enumerate(rois):
        sig = extract_1d_signal(roi)
        raw_signals.append(sig)
        print(f"       Lead {LEAD_NAMES[i]:>4s}  width={len(sig):>5d} px")

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    print(f"[4/4] Resampling to {target_len} pts & Z-score normalising …")
    leads: list[np.ndarray] = []
    for i, sig in enumerate(raw_signals):
        resampled  = resample_signal(sig, target_len)
        normalised = zscore_normalize(resampled)
        leads.append(normalised)

    ecg_array = np.stack(leads, axis=1)   # shape: (target_len, 12)
    assert ecg_array.shape == (target_len, 12), ecg_array.shape

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(str(out_path), ecg_array)
    print(f"\n✅  Saved  {out_path}   shape={ecg_array.shape}  dtype={ecg_array.dtype}")

    return ecg_array


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Digitize a 12-lead ECG photograph → (1000, 12) NumPy array."
    )
    parser.add_argument("image", help="Path to the ECG image file (PNG/JPEG/BMP).")
    parser.add_argument(
        "--out", default="extracted_ecg_signal.npy",
        help="Output .npy file path (default: extracted_ecg_signal.npy).",
    )
    parser.add_argument(
        "--len", dest="target_len", type=int, default=1000,
        help="Number of time-steps per lead after resampling (default: 1000).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Write intermediate debug images into signal_extractor_debug/ folder.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    extract_ecg_array(
        image_path=args.image,
        out_path=args.out,
        target_len=args.target_len,
        debug=args.debug,
    )
