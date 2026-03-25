"""
src/report_extractor.py
=======================
Medical PDF Extraction Pipeline — ECG On-Demand / Technomed 12-Lead ECG Report

Extraction strategy (in priority order):
  1. Google Gemini API  — best accuracy, handles any PDF layout (requires API key)
  2. PyMuPDF regex      — fast local fallback, works on structured PDFs

Output schema (same regardless of strategy):
  {
    "parameters":           { ventricular_rate, rr_interval, p_duration, pr_interval,
                              qrs_duration, qt_interval, qtc_interval, qrs_axis, p_axis, t_axis },
    "qualitative_findings": { ECG Quality, Ventricular Rate, ... },
    "raw_text":             "<full interpretation string>",
    "predicted_class":      "NORM" | "MI" | "STTC" | "CD" | "HYP",
    "patient_metadata":     { job_number, patient_number, patient_name, birth_date, gender, recorded }
  }

Usage:
    python src/report_extractor.py --pdf path/to/ecg_report.pdf
    python src/report_extractor.py --pdf path/to/ecg_report.pdf --debug
    python src/report_extractor.py --pdf path/to/ecg_report.pdf --no-gemini   # force local
"""

import re
import json
import os
import argparse
import sys
from pathlib import Path

# ── PyMuPDF ─────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    print("[WARN] PyMuPDF not installed. Run: pip install pymupdf")
    FITZ_AVAILABLE = False

# ── Google Gemini API ────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── pytesseract (OPTIONAL) ────────────────────────────────────────────────────
try:
    import pytesseract
    import shutil
    TESSERACT_AVAILABLE = False
    _tess = shutil.which('tesseract')
    if _tess:
        pytesseract.pytesseract.tesseract_cmd = _tess
        TESSERACT_AVAILABLE = True
    else:
        # Common Windows install locations
        _win_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        ]
        for _p in _win_paths:
            if os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                TESSERACT_AVAILABLE = True
                break
except ImportError:
    TESSERACT_AVAILABLE = False

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

BASE   = Path(__file__).resolve().parent.parent
OUTPUT = BASE / 'doctor_ground_truth.json'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _int(pattern: str, text: str, flags: int = re.IGNORECASE):
    m = re.search(pattern, text, flags)
    if m:
        try:
            return int(m.group(1))
        except (IndexError, ValueError):
            pass
    return None


def _float(pattern: str, text: str, flags: int = re.IGNORECASE):
    m = re.search(pattern, text, flags)
    if m:
        try:
            return float(m.group(1))
        except (IndexError, ValueError):
            pass
    return None


def _str(pattern: str, text: str, flags: int = re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _get_gemini_key() -> str | None:
    """
    Resolve the Gemini API key from (in order):
      1. GEMINI_API_KEY env var (set this on Streamlit Cloud via Secrets)
      2. Streamlit st.secrets (when running inside Streamlit)
    """
    # 1) Environment variable (works everywhere)
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key

    # 2) Streamlit secrets (works when run via `streamlit run ...`)
    try:
        import streamlit as st
        try:
            # SecretDict supports indexing more reliably than .get() across versions
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = st.secrets.get("GEMINI_API_KEY", "")  # type: ignore[attr-defined]
        if key:
            return key
    except Exception:
        pass

    # 3) Local fallback: read .streamlit/secrets.toml from project root
    try:
        secrets_path = BASE / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            raw = secrets_path.read_text(encoding="utf-8")
            data = None
            try:
                import tomllib  # py>=3.11
                data = tomllib.loads(raw)
            except Exception:
                try:
                    import tomli  # optional backport
                    data = tomli.loads(raw)  # type: ignore[attr-defined]
                except Exception:
                    data = None
            if isinstance(data, dict):
                key = data.get("GEMINI_API_KEY", "")
                if key:
                    return key
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers for Gemini responses
# ─────────────────────────────────────────────────────────────────────────────

def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object from a model response.
    Handles accidental markdown fences and extra surrounding text.
    """
    raw = (text or "").strip()
    if not raw:
        return raw

    # Remove accidental markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1].strip()
    return raw


def _json_from_gemini_text(response_text: str, debug: bool = False) -> dict | None:
    candidate = _extract_first_json_object(response_text)
    if debug:
        print("[GEMINI] JSON candidate:\n", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _get_gemini_model_name() -> str:
    # Use Gemini model that is known to exist for your account.
    # Override with GEMINI_MODEL env var if needed.
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — Gemini API extraction
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_PROMPT = """
You are a medical AI assistant that extracts structured data from ECG PDF reports.
Analyze the attached PDF and return ONLY a JSON object with the following structure (no markdown, no explanation):

{
  "parameters": {
    "ventricular_rate": <integer bpm or null>,
    "rr_interval": <integer ms or null>,
    "p_duration": <integer ms or null>,
    "pr_interval": <integer ms or null>,
    "qrs_duration": <integer ms or null>,
    "qt_interval": <integer ms or null>,
    "qtc_interval": <integer ms or null>,
    "qrs_axis": <integer degrees or null>,
    "p_axis": <integer degrees or null>,
    "t_axis": <integer degrees or null>
  },
  "qualitative_findings": {
    "ECG Quality": "<value or null>",
    "Ventricular Rate": "<value or null>",
    "PR Interval": "<value or null>",
    "QRS Duration": "<value or null>",
    "QTc Interval": "<value or null>",
    "Cardiac Axis": "<value or null>",
    "Sinus Rhythm Present": "<value or null>",
    "Other Rhythm": "<value or null>",
    "Atrial pause of more than 2 seconds": "<value or null>",
    "AV Conduction": "<value or null>",
    "Ventricular Ectopics": "<value or null>",
    "Atrial Ectopics": "<value or null>",
    "P-Wave Morphology": "<value or null>",
    "QRS Morphology": "<value or null>",
    "Q-Wave": "<value or null>",
    "T-Wave Morphology": "<value or null>",
    "ST Segment": "<value or null>"
  },
  "raw_text": "<full interpretation / diagnosis text from the report>",
  "patient_metadata": {
    "job_number": "<value or null>",
    "patient_number": "<value or null>",
    "patient_name": "<value or null>",
    "birth_date": "<value or null>",
    "gender": "<value or null>",
    "recorded": "<value or null>"
  }
}

Rules:
- Use null (not "null") for any field you cannot find.
- For numeric fields: extract only the number (e.g. 38, not "38 bpm").
- The `parameters` object MUST contain ALL keys listed under `parameters` above.
- Numeric values MUST be integers. If units/extra text exist, extract only the number.
- raw_text should be the complete interpretation/conclusion/findings paragraph(s).
- Return ONLY the JSON — no markdown code fences, no extra text.
"""

_GEMINI_TEXT_PROMPT = """
You are a medical AI assistant that extracts structured data from ECG report text.

You will be given extracted text from an ECG PDF. Extract the same fields as the PDF-based extraction.
Return ONLY a JSON object with the following structure (no markdown, no explanation):

{
  "parameters": {
    "ventricular_rate": <integer bpm or null>,
    "rr_interval": <integer ms or null>,
    "p_duration": <integer ms or null>,
    "pr_interval": <integer ms or null>,
    "qrs_duration": <integer ms or null>,
    "qt_interval": <integer ms or null>,
    "qtc_interval": <integer ms or null>,
    "qrs_axis": <integer degrees or null>,
    "p_axis": <integer degrees or null>,
    "t_axis": <integer degrees or null>
  },
  "qualitative_findings": {
    "ECG Quality": "<value or null>",
    "Ventricular Rate": "<value or null>",
    "PR Interval": "<value or null>",
    "QRS Duration": "<value or null>",
    "QTc Interval": "<value or null>",
    "Cardiac Axis": "<value or null>",
    "Sinus Rhythm Present": "<value or null>",
    "Other Rhythm": "<value or null>",
    "Atrial pause of more than 2 seconds": "<value or null>",
    "AV Conduction": "<value or null>",
    "Ventricular Ectopics": "<value or null>",
    "Atrial Ectopics": "<value or null>",
    "P-Wave Morphology": "<value or null>",
    "QRS Morphology": "<value or null>",
    "Q-Wave": "<value or null>",
    "T-Wave Morphology": "<value or null>",
    "ST Segment": "<value or null>"
  },
  "raw_text": "<full interpretation / diagnosis text from the report text>",
  "patient_metadata": {
    "job_number": "<value or null>",
    "patient_number": "<value or null>",
    "patient_name": "<value or null>",
    "birth_date": "<value or null>",
    "gender": "<value or null>",
    "recorded": "<value or null>"
  }
}

Rules:
- Use null (not "null") for any field you cannot find.
- For numeric fields: extract only the number (e.g. 38, not "38 bpm").
- The `parameters` object MUST contain ALL keys listed under `parameters` above.
- Numeric values MUST be integers. If units/extra text exist, extract only the number.
- raw_text should be the complete interpretation/conclusion/findings paragraph(s) present in the text.
- Return ONLY the JSON — no markdown code fences, no extra text.
"""


def extract_via_gemini_text(pdf_text: str, debug: bool = False) -> dict | None:
    """
    Gemini extraction using text-only input.
    Useful when PDF upload/analysis fails, but we still have local extracted content.
    """
    if not GEMINI_AVAILABLE:
        print("[GEMINI-TEXT] google-generativeai not installed — skipping.")
        return None

    api_key = _get_gemini_key()
    if not api_key:
        print("[GEMINI-TEXT] No API key found (set GEMINI_API_KEY env var) — skipping.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_gemini_model_name())

        inputs = [
            _GEMINI_TEXT_PROMPT,
            f"EXTRACTED_PDF_TEXT:\n{pdf_text}",
        ]

        try:
            response = model.generate_content(
                inputs,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            response = model.generate_content(inputs)

        response_text = getattr(response, "text", "") or ""
        extracted = _json_from_gemini_text(response_text, debug=debug)
        if extracted is None:
            print("[GEMINI-TEXT] Could not parse JSON response.")
            if debug and response_text:
                print("[GEMINI-TEXT] Raw response (truncated):\n", response_text[:2000])
            return None

        print("[GEMINI-TEXT] Extraction successful.")
        return extracted
    except Exception as e:
        print(f"[GEMINI-TEXT] API error: {e} — skipping.")
        return None


def extract_via_gemini(pdf_path: str, debug: bool = False) -> dict | None:
    """
    Upload the PDF to the Gemini File API and extract structured ECG data.

    Returns the parsed dict on success, or None if Gemini is unavailable
    or the key is missing.
    """
    if not GEMINI_AVAILABLE:
        print("[GEMINI] google-generativeai not installed — skipping.")
        return None

    api_key = _get_gemini_key()
    if not api_key:
        print("[GEMINI] No API key found (set GEMINI_API_KEY env var) — skipping.")
        return None

    try:
        genai.configure(api_key=api_key)

        print(f"[GEMINI] Uploading PDF: {pdf_path}")
        uploaded = genai.upload_file(path=pdf_path, mime_type="application/pdf")
        print(f"[GEMINI] File uploaded → {uploaded.name}")

        model = genai.GenerativeModel(_get_gemini_model_name())
        # Some google-generativeai versions are picky about response_mime_type.
        try:
            response = model.generate_content(
                [uploaded, _GEMINI_PROMPT],
                generation_config=genai.GenerationConfig(
                    temperature=0.0,        # deterministic
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            response = model.generate_content([uploaded, _GEMINI_PROMPT])

        response_text = getattr(response, "text", "") or ""
        extracted = _json_from_gemini_text(response_text, debug=debug)

        # Clean up the uploaded file (avoid quota waste)
        try:
            genai.delete_file(uploaded.name)
        except Exception:
            pass

        if debug and response_text:
            print("[GEMINI] Raw response:\n", response_text[:2000])

        if extracted is None:
            print("[GEMINI] Could not parse JSON response — falling back.")
            return None

        print("[GEMINI] Extraction successful.")
        return extracted

    except Exception as e:
        print(f"[GEMINI] API error: {e} — falling back to local pipeline.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — Local PyMuPDF + regex extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str, debug: bool = False, rotate_k: int = 0) -> dict:
    """Uses PyMuPDF for text extraction. Falls back to Tesseract for image pages."""
    if not FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")

    pages = {}
    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"[STEP 1] PDF opened — {total} page(s) found: {pdf_path} (PyMuPDF)")

    for i in range(total):
        page_num = i + 1
        page = doc.load_page(i)
        text = page.get_text("text") or ""

        if not text.strip() and TESSERACT_AVAILABLE:
            print(f"         Page {page_num}: no embedded text → Tesseract OCR")
            pix = page.get_pixmap(dpi=300)
            try:
                if pix.n == 1:
                    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                    if rotate_k:
                        gray = np.rot90(gray, k=rotate_k)
                else:
                    channels = 3 if pix.n >= 3 else pix.n
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    img = img[:, :, :channels]
                    # Convert to grayscale without relying on cv2 (more portable).
                    if rotate_k:
                        img = np.rot90(img, k=rotate_k)
                    if channels == 3:
                        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
                    else:
                        gray = img[:, :, 0]

                from PIL import Image
                pil_img = Image.fromarray(gray)
                text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6')
            except Exception as ocr_e:
                print(f"         Page {page_num}: OCR failed: {ocr_e}")
                text = ""

        pages[page_num] = text
        if debug:
            print(f"\n{'─'*55}  PAGE {page_num}\n{text}\n{'─'*55}")

    doc.close()
    return pages


def extract_numeric_measurements(pages: dict) -> dict:
    p2 = pages.get(2, "")

    ventricular_rate = _int(r'ventricular\s+rate\s*[:\-]?\s*(\d{2,3})\s*bpm', p2)
    rr_interval      = _int(r'rr\s+interval\s*[:\-]?\s*(\d{3,5})\s*ms', p2)
    p_duration       = _int(r'p\s+duration\s*[:\-]?\s*(\d{2,3})\s*ms', p2)
    pr_interval      = _int(r'pr\s+duration\s*[:\-]?\s*(\d{2,4})\s*ms', p2)
    qrs_duration     = _int(r'qrs\s+duration\s*[:\-]?\s*(\d{2,3})\s*ms', p2)

    qt_qtc_match = re.search(
        r'qt\s*/\s*qtc[a-z]*\s+interval\s*[:\-]?\s*(\d{3,4})\s*/\s*(\d{3,4})\s*ms',
        p2, re.IGNORECASE
    )
    qt_interval  = int(qt_qtc_match.group(1)) if qt_qtc_match else None
    qtc_interval = int(qt_qtc_match.group(2)) if qt_qtc_match else None
    qrs_axis     = _int(r'qrs\s+axis\s*[:\-]?\s*(-?\d{1,3})\s*[°d]', p2)

    # Page 3 fallback
    p3 = pages.get(3, "")
    if p_duration is None:
        p_duration = _int(r'p\s+duration\s*[:\-]?\s*(\d{2,3})\s*ms', p3)
    if pr_interval is None:
        pr_interval = _int(r'pr\s+duration\s*[:\-]?\s*(\d{2,4})\s*ms', p3)
    if qrs_duration is None:
        qrs_duration = _int(r'qrs\s+duration\s*[:\-]?\s*(\d{2,3})\s*ms', p3)
    if qt_interval is None or qtc_interval is None:
        qt_qtc_p3 = re.search(
            r'qt\s*/\s*qtc[a-z]*\s+interval\s*[:\-]?\s*(\d{3,4})\s*/\s*(\d{3,4})\s*ms',
            p3, re.IGNORECASE
        )
        if qt_qtc_p3:
            qt_interval  = qt_interval  or int(qt_qtc_p3.group(1))
            qtc_interval = qtc_interval or int(qt_qtc_p3.group(2))

    # Page 1 history fallback
    p1 = pages.get(1, "")
    history_rows = re.findall(
        r'\d{2}/[x\d]{2}/[x\d]{4}\s+[x\d]{2}:[x\d]{2}:[x\d]{2}\s+'
        r'(\d{2,4})\s+(\d{2,3})\s+(\d{3,4})',
        p1
    )
    if history_rows:
        most_recent = history_rows[0]
        if pr_interval  is None: pr_interval  = int(most_recent[0])
        if qrs_duration is None: qrs_duration = int(most_recent[1])
        if qtc_interval is None: qtc_interval = int(most_recent[2])

    p_axis = _int(r'\bp\s*axis\s*[:\-]?\s*(-?\d{1,3})', p1)
    t_axis = _int(r'\bt\s*axis\s*[:\-]?\s*(-?\d{1,3})', p1)

    print("[STEP 2] Numeric measurements extracted:")
    print(f"         Ventricular Rate : {ventricular_rate} bpm")
    print(f"         RR Interval      : {rr_interval} ms")
    print(f"         P Duration       : {p_duration} ms")
    print(f"         PR Interval      : {pr_interval} ms")
    print(f"         QRS Duration     : {qrs_duration} ms")
    print(f"         QT Interval      : {qt_interval} ms")
    print(f"         QTc Interval     : {qtc_interval} ms")
    print(f"         QRS Axis         : {qrs_axis}°")
    print(f"         P Axis           : {p_axis}°  |  T Axis: {t_axis}°")

    return {
        "ventricular_rate": ventricular_rate,
        "rr_interval":      rr_interval,
        "p_duration":       p_duration,
        "pr_interval":      pr_interval,
        "qrs_duration":     qrs_duration,
        "qt_interval":      qt_interval,
        "qtc_interval":     qtc_interval,
        "qrs_axis":         qrs_axis,
        "p_axis":           p_axis,
        "t_axis":           t_axis,
    }


PHYSIO_FIELDS = [
    "ECG Quality", "Ventricular Rate", "PR Interval", "QRS Duration",
    "QTc Interval", "Cardiac Axis", "Sinus Rhythm Present", "Other Rhythm",
    "Atrial pause of more than 2 seconds", "AV Conduction",
    "Ventricular Ectopics", "Atrial Ectopics", "P-Wave Morphology",
    "QRS Morphology", "Q-Wave", "T-Wave Morphology", "ST Segment",
]

PARAM_KEYS = [
    "ventricular_rate",
    "rr_interval",
    "p_duration",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "qtc_interval",
    "qrs_axis",
    "p_axis",
    "t_axis",
]


def _coerce_int(v: object) -> int | None:
    """Convert Gemini/local numeric values to int; return None on failure."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        # Avoid turning NaN into an int
        if isinstance(v, float) and (v != v):  # NaN check
            return None
        return int(round(float(v)))
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    m = re.search(r"-?\d+", s.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def extract_qualitative_findings(pages: dict) -> dict:
    p1 = pages.get(1, "")
    block_match = re.search(
        r"Physiologist's Report.*?(?=Reporting Physiologist)",
        p1, re.IGNORECASE | re.DOTALL
    )
    block = block_match.group(0) if block_match else p1
    findings = {}
    for field in PHYSIO_FIELDS:
        esc = re.escape(field)
        m = re.search(rf'{esc}\s+(.+)', block, re.IGNORECASE)
        if not m:
            findings[field] = None
            continue
        raw_value = m.group(1).strip()
        raw_value = re.split(r'\d{2}/[x\d]{2}/[x\d]{4}', raw_value)[0].strip()
        raw_value = re.sub(
            r'(Measurement History.*|Recorded\s+PR\s+QRS.*)', '', raw_value,
            flags=re.IGNORECASE
        ).strip()
        raw_value = re.sub(r'\s+', ' ', raw_value).strip()
        findings[field] = raw_value or None

    print("[STEP 3] Qualitative findings extracted.")
    return findings


def extract_interpretation(pages: dict) -> str:
    p4 = pages.get(4, "").strip()
    if p4:
        skip = re.compile(r'^(Extended Report|Reference #|Page \d+ of \d+)$', re.IGNORECASE)
        lines = [l.strip() for l in p4.splitlines() if l.strip() and not skip.match(l.strip())]
        interp = ' '.join(lines)
        print(f"[STEP 4] Interpretation from Page 4 ({len(interp)} chars)")
        if interp.strip():
            return interp

    p1 = pages.get(1, "")
    m = re.search(
        r'Cardiology\s+Advice\s+(.*?)(?:Physiologist|Measurement History)',
        p1, re.IGNORECASE | re.DOTALL
    )
    if m:
        interp = re.sub(r'\s+', ' ', m.group(1)).strip()
        print(f"[STEP 4] Interpretation from Page 1 ({len(interp)} chars)")
        if interp.strip():
            return interp

    # Fallback: concatenate all page text and return
    all_text = ' '.join(pages.values())
    print("[STEP 4] WARNING: No dedicated interpretation block — using full text.")
    return all_text[:10000]


MAPPING_RULES: dict = {
    'NORM': ['normal sinus', 'sinus rhythm', 'sinus bradycardia', 'unremarkable',
             'within normal limits', 'normal ecg', 'no significant', 'normal findings'],
    'MI':   ['infarct', 'infarction', 'q-wave', 'q wave', 'myocardial',
             'anterior wall', 'inferior wall', 'anteroseptal', 'stemi', 'nstemi'],
    'STTC': ['ischemia', 'ischaemia', 'st depression', 'st elevation', 't-wave inversion',
             't wave inversion', 'st change', 'sttc', 'ischemic st',
             'repolarisation', 'repolarization'],
    'CD':   ['block', 'lbbb', 'rbbb', 'bundle branch', 'av block', 'fascicular',
             'wpw', 'conduction', 'wenckebach', 'mobitz', 'heart block',
             'degree av', 'degree heart block', 'pacemaker'],
    'HYP':  ['hypertrophy', 'lvh', 'rvh', 'enlargement', 'overload', 'hyp',
             'voltage criteria'],
}


def classify_diagnosis(text: str) -> str:
    """Maps interpretation text to a PTB-XL superclass. Priority: CD > STTC > MI > HYP > NORM."""
    text_lower = text.lower()
    for ptbxl_class in ['CD', 'STTC', 'MI', 'HYP', 'NORM']:
        for kw in MAPPING_RULES[ptbxl_class]:
            if kw in text_lower:
                print(f"[STEP 5] Keyword '{kw}' matched → {ptbxl_class}")
                return ptbxl_class
    print("[STEP 5] No keyword matched → defaulting to NORM")
    return 'NORM'


def extract_patient_metadata(pages: dict) -> dict:
    p2 = pages.get(2, "")

    def _field(label: str, text: str):
        m = re.search(rf'{re.escape(label)}\s*[:\-]\s*(.+)', text, re.IGNORECASE)
        if not m:
            return None
        val = re.split(
            r'(?:\s{2,}|\t|(?=Ventricular|RR interval|P duration|PR duration|QRS duration|QT /))',
            m.group(1)
        )[0]
        return val.strip() or None

    metadata = {
        "job_number":     _field("Job Number",     p2),
        "patient_number": _field("Patient Number", p2),
        "patient_name":   _field("Name",           p2),
        "birth_date":     _field("Birth Date",     p2),
        "gender":         _field("Gender",         p2),
        "recorded":       _field("Recorded",       p2),
    }
    if metadata["birth_date"]:
        metadata["birth_date"] = re.sub(r'\s*\(.*?\)', '', metadata["birth_date"]).strip()
    return metadata


def _local_pipeline(pdf_path: str, debug: bool = False, rotate_k: int = 0) -> dict:
    """Full local extraction using PyMuPDF + regex."""
    pages        = extract_pages(pdf_path, debug=debug, rotate_k=rotate_k)
    parameters   = extract_numeric_measurements(pages)
    print()
    qualitative  = extract_qualitative_findings(pages)
    print()
    interpretation = extract_interpretation(pages)
    print()
    ptbxl_class  = classify_diagnosis(interpretation)
    metadata     = extract_patient_metadata(pages)

    return {
        "parameters":           parameters,
        "qualitative_findings": qualitative,
        "raw_text":             interpretation,
        "predicted_class":      ptbxl_class,
        "patient_metadata":     metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline — Gemini first, local fallback
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    debug: bool = False,
    use_gemini: bool = True,
    rotate_k: int = 0,
    save_output: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Full extraction pipeline.

    Args:
        pdf_path:   Path to the ECG PDF.
        debug:      Print raw page text (local pipeline only).
        use_gemini: Try Gemini API first (default True). Set False to force local.

    Returns:
        Artifact dict with keys: parameters, qualitative_findings,
        raw_text, predicted_class, patient_metadata.
    """
    print(f"\n{'='*60}")
    print("  ECG Report Extraction Pipeline")
    print(f"{'='*60}\n")

    artifact = None

    # ── Strategy 1: Gemini API ────────────────────────────────────────────────
    if use_gemini:
        gemini_data = None

        # If the PDF/image was rotated in the UI, prefer rotated text extraction.
        # Gemini's PDF upload may still preserve the original orientation and miss fields.
        if rotate_k:
            try:
                pdf_text = "\n".join(
                    t.strip()
                    for t in extract_pages(pdf_path, debug=False, rotate_k=rotate_k).values()
                    if t.strip()
                )
                pdf_text = re.sub(r"\s+", " ", pdf_text).strip()[:12000]
            except Exception:
                pdf_text = ""

            if pdf_text:
                gemini_data = extract_via_gemini_text(pdf_text, debug=debug)
                if gemini_data is not None:
                    print("[PIPELINE] Used: Gemini API (rotated text)")
        else:
            gemini_data = extract_via_gemini(pdf_path, debug=debug)

            if gemini_data is None:
                # If PDF upload/parse fails (quota/format), try Gemini using extracted text.
                try:
                    pdf_text = "\n".join(
                        t.strip() for t in extract_pages(pdf_path, debug=False).values() if t.strip()
                    )
                    pdf_text = re.sub(r"\s+", " ", pdf_text).strip()[:12000]
                except Exception:
                    pdf_text = ""

                if pdf_text:
                    try:
                        gemini_data = extract_via_gemini_text(pdf_text, debug=debug)
                        print("[PIPELINE] Used: Gemini API (text fallback)")
                    except Exception:
                        gemini_data = None

        if gemini_data is not None:
            # Ensure predicted_class is set (Gemini doesn't classify PTB-XL directly)
            raw_text = gemini_data.get("raw_text", "")
            if "predicted_class" not in gemini_data or not gemini_data["predicted_class"]:
                gemini_data["predicted_class"] = classify_diagnosis(raw_text)

            # Ensure all top-level keys exist
            if not isinstance(gemini_data.get("parameters"), dict):
                gemini_data["parameters"] = {}
            for k in PARAM_KEYS:
                gemini_data["parameters"][k] = _coerce_int(gemini_data["parameters"].get(k))
            gemini_data.setdefault("qualitative_findings", {field: None for field in PHYSIO_FIELDS})
            gemini_data.setdefault("patient_metadata", {})

            artifact = gemini_data
            print("[PIPELINE] Used: Gemini API")

    # ── Strategy 2: Local PyMuPDF fallback ────────────────────────────────────
    if artifact is None:
        print("[PIPELINE] Using local PyMuPDF + regex pipeline.")
        artifact = _local_pipeline(pdf_path, debug=debug, rotate_k=rotate_k)
        print("[PIPELINE] Used: local PyMuPDF")

    # ── Save (optional) ─────────────────────────────────────────────────────
    if save_output:
        target = Path(output_path) if output_path else OUTPUT
        output_str = json.dumps(artifact, indent=2)
        try:
            target.write_text(output_str)
            print(f"\n{'='*60}")
            print(f"  OUTPUT saved → {target}")
            print(f"{'='*60}\n")
        except Exception as save_e:
            # Don't fail extraction if the working directory is read-only (common on Streamlit Cloud).
            print(f"[WARN] Could not write OUTPUT file: {save_e}")

        if debug:
            print(output_str)

    return artifact


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ECG PDF Extraction Pipeline (Gemini API + PyMuPDF fallback)'
    )
    parser.add_argument('--pdf', type=str, required=True, help='Path to the ECG report PDF')
    parser.add_argument('--debug', action='store_true', help='Print raw per-page text')
    parser.add_argument('--no-gemini', action='store_true', help='Force local PyMuPDF pipeline')
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"[ERROR] File not found: {args.pdf}")
        sys.exit(1)

    run_pipeline(
        args.pdf,
        debug=args.debug,
        use_gemini=not args.no_gemini,
        save_output=True,
        output_path=str(OUTPUT),
    )


if __name__ == '__main__':
    main()