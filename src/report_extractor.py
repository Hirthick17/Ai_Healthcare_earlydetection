"""
src/report_extractor.py
Medical OCR Extraction Pipeline — Reads a photographed ECG report header,
extracts clinical parameters, and outputs doctor_ground_truth.json.

Usage:
    python src/report_extractor.py --image path/to/ecg_report.jpg
    python src/report_extractor.py --image path/to/ecg_report.jpg --debug
"""

import cv2
import re
import json
import argparse
import sys
from pathlib import Path

# ── Tesseract (pytesseract) ────────────────────────────────────────────────────
try:
    import pytesseract
    # Auto-detect Tesseract binary on Windows common install paths
    import shutil, os
    _tess = shutil.which('tesseract')
    if not _tess:
        _win_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        ]
        for _p in _win_paths:
            if os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                print(f"[INFO] Tesseract found at: {_p}")
                break
        else:
            print("[ERROR] Tesseract binary not found!")
            print("        Download and install from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("        Direct link: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.5.0.20241111.exe")
            sys.exit(1)
except ImportError:
    print("[ERROR] pytesseract not installed. Run: pip install pytesseract")
    sys.exit(1)

import numpy as np

BASE   = Path(__file__).parent.parent
OUTPUT = BASE / 'doctor_ground_truth.json'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Image Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path: str, debug: bool = False):
    """
    Loads the ECG report image, crops the top 20% header (where text resides),
    applies grayscale + Otsu thresholding to maximise text contrast.

    Returns the binary thresholded ROI ready for Tesseract.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        sys.exit(1)

    print(f"[STEP 1] Loaded image — size: {img.shape[1]}x{img.shape[0]} px")

    # ROI: top 20% of the image (header contains printed parameters)
    h, w = img.shape[:2]
    roi_h = int(h * 0.20)
    roi   = img[:roi_h, :]
    print(f"         ROI cropped  — {w}x{roi_h} px (top 20% header only)")

    # Grayscale conversion
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Upscale for better OCR accuracy on small fonts (2x)
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2),
                      interpolation=cv2.INTER_CUBIC)

    # Otsu's binarisation — automatically picks the optimal threshold
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite('debug_roi_binary.png', binary)
        print("         Debug image saved -> debug_roi_binary.png")

    return binary


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — OCR & Regex Extraction
# ─────────────────────────────────────────────────────────────────────────────
def run_ocr(binary_roi) -> str:
    """Passes the preprocessed ROI to Tesseract and returns the raw string."""
    # psm 6: uniform block of text. Remove char whitelist so spaces are kept.
    config = '--oem 3 --psm 6'
    raw = pytesseract.image_to_string(binary_roi, config=config)
    print(f"[STEP 2] Tesseract raw output:\n{'-'*50}\n{raw.strip()}\n{'-'*50}")
    return raw


def extract_parameters(raw_text: str) -> dict:
    """
    Applies robust regex patterns to extract the three key clinical parameters.
    Accounts for spacing variations, label aliases, and trailing units.
    """
    def find_value(patterns, text):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    ventricular_rate = find_value([
        r'(?:ventricular\s+rate|heart\s+rate|rate)[^\d]*(\d{2,3})\s*(?:bpm)?',
        r'\brate\b[^\d]*(\d{2,3})',
    ], raw_text)

    pr_interval = find_value([
        r'(?:pr\s*interval|pr)[^\d]*(\d{2,4})\s*(?:ms)?',
        r'\bpr\b[^\d]*(\d{2,4})',
    ], raw_text)

    qrs_duration = find_value([
        r'(?:qrs\s*duration|qrs)[^\d]*(\d{2,3})\s*(?:ms)?',
        r'\bqrs\b[^\d]*(\d{2,3})',
    ], raw_text)

    print(f"[STEP 2] Extracted parameters:")
    print(f"         Ventricular Rate : {ventricular_rate} bpm")
    print(f"         PR Interval      : {pr_interval} ms")
    print(f"         QRS Duration     : {qrs_duration} ms")

    return {
        "ventricular_rate": ventricular_rate,
        "pr_interval":      pr_interval,
        "qrs_duration":     qrs_duration,
    }


def extract_diagnosis_text(raw_text: str) -> str:
    """
    Extracts the trailing descriptive diagnostic sentence from the header text.
    Skips numeric-heavy lines (QT interval, P-R-T Axes, etc.) and only keeps
    human-readable diagnostic sentences (e.g. 'Sinus rhythm. Normal ECG.')
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    diagnosis_lines = []
    for line in lines:
        # Skip lines that are over 50% digits/colons — these are parameter rows
        digit_ratio = sum(1 for c in line if c.isdigit() or c in ':/') / max(len(line), 1)
        if digit_ratio > 0.4:
            continue
        # Only keep lines with real words (4+ consecutive letters)
        if re.search(r'[a-zA-Z]{4,}', line) and len(line) > 8:
            # Normalise spaces (OCR sometimes merges words without spaces)
            cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)  # camelCase fix
            cleaned = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', cleaned)
            cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', cleaned)
            diagnosis_lines.append(cleaned.strip())

    if diagnosis_lines:
        return '. '.join(diagnosis_lines[-2:])   # last 1-2 meaningful lines

    # Fallback: clean the whole raw string
    fallback = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_text)
    return fallback.strip()[-200:]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PTB-XL Classification Mapping
# ─────────────────────────────────────────────────────────────────────────────
MAPPING_RULES = {
    'NORM': ['normal', 'sinus rhythm', 'sinus bradycardia',
             'unremarkable', 'within normal'],
    'MI'  : ['infarct', 'infarction', 'q-wave', 'q wave',
             'myocardial', 'anterior wall', 'inferior wall'],
    'STTC': ['ischemia', 'ischaemia', 'st depression', 'st elevation',
             't-wave inversion', 't wave inversion', 'st change', 'sttc'],
    'CD'  : ['block', 'lbbb', 'rbbb', 'bundle branch',
             'av block', 'fascicular', 'wpw', 'conduction'],
    'HYP' : ['hypertrophy', 'lvh', 'rvh', 'enlargement',
             'overload', 'hyp'],
}

def classify_diagnosis(raw_text: str) -> str:
    """Maps the raw diagnostic text to a PTB-XL superclass label."""
    text_lower = raw_text.lower()
    for ptbxl_class, keywords in MAPPING_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                print(f"[STEP 3] Keyword '{kw}' matched -> {ptbxl_class}")
                return ptbxl_class
    print("[STEP 3] No keyword matched -> defaulting to NORM")
    return 'NORM'


# ─────────────────────────────────────────────────────────────────────────────
# CLI Runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Medical OCR Extraction Pipeline — ECG Report Parser'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to the photographed ECG report image (.jpg, .png, .tiff)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Save intermediate debug images (debug_roi_binary.png)'
    )
    args = parser.parse_args()

    print("\n========================================")
    print("  MEDICAL OCR EXTRACTION PIPELINE v1.0")
    print("========================================\n")

    # Step 1: Preprocess
    binary_roi = preprocess_image(args.image, debug=args.debug)

    # Step 2: OCR + Regex
    raw_ocr    = run_ocr(binary_roi)
    parameters = extract_parameters(raw_ocr)
    diag_text  = extract_diagnosis_text(raw_ocr)

    # Step 3: Classify
    ptbxl_class = classify_diagnosis(diag_text)

    # Step 4: Build and save artifact
    artifact = {
        "parameters": {
            "ventricular_rate": parameters["ventricular_rate"],
            "pr_interval":      parameters["pr_interval"],
            "qrs_duration":     parameters["qrs_duration"],
        },
        "raw_text":        diag_text,
        "predicted_class": ptbxl_class,
    }

    OUTPUT.write_text(json.dumps(artifact, indent=2))

    print(f"\n[STEP 4] Output artifact saved -> {OUTPUT}")
    print(f"\n========================================")
    print(f"  EXTRACTION RESULT:")
    print(f"  Ventricular Rate : {parameters['ventricular_rate']} bpm")
    print(f"  PR Interval      : {parameters['pr_interval']} ms")
    print(f"  QRS Duration     : {parameters['qrs_duration']} ms")
    print(f"  Diagnosis Text   : {diag_text[:80]}...")
    print(f"  PTB-XL Class     : {ptbxl_class}")
    print(f"========================================\n")

    return artifact


if __name__ == '__main__':
    main()
