"""
Gemini connectivity + PDF processing test.

Loads `GEMINI_API_KEY` from `.streamlit/secrets.toml` and calls the configured
Gemini model to verify the API key and basic PDF processing.

Usage examples:
  python gemini_pdf_connectivity_test.py --pdf "D:\\Downloads\\report_ecg.pdf" --model "gemini-2.5-flash"
  python gemini_pdf_connectivity_test.py --text-only --model "gemini-2.5-flash"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _load_gemini_key_from_secrets(project_root: Path) -> str | None:
    # 1) Environment variable
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # 2) Streamlit secrets file (local dev)
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    raw = secrets_path.read_text(encoding="utf-8")

    # Try real TOML parsing first (tomllib/tomli), then fall back to a
    # lightweight regex for this project's secrets format.
    data = None
    try:
        import tomllib  # py>=3.11

        data = tomllib.loads(raw)
    except Exception:
        try:
            import tomli  # type: ignore

            data = tomli.loads(raw)  # type: ignore[attr-defined]
        except Exception:
            data = None

    if isinstance(data, dict):
        key = str(data.get("GEMINI_API_KEY", "")).strip()
        return key or None

    # Fallback: handle simple `KEY = "value"` format
    m = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', raw)
    if not m:
        m = re.search(r"GEMINI_API_KEY\s*=\s*'([^']+)'", raw)
    if m:
        return m.group(1).strip() or None

    return None


def _extract_first_json_object(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1].strip()
    return raw


def _gemini_text_ping(genai, model_name: str) -> None:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content("Respond with exactly the word OK.")
    print("[PING] Model text response:", getattr(resp, "text", "").strip())


def _gemini_pdf_extract(genai, model_name: str, pdf_path: Path, debug: bool = False) -> dict | None:
    """
    Uses the Generative AI File API to upload a PDF and request JSON output.
    """
    prompt = """
You are a medical AI assistant that extracts structured data from ECG PDF reports.
Analyze the attached PDF and return ONLY a JSON object.

Return the keys (ensure all numeric fields exist under `parameters`, using null if not found):
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
  "raw_text": "<interpretation text or findings block, or empty string>"
}
Rules:
- Return ONLY JSON (no markdown fences).
- Numeric values MUST be integers. If units/extra text exist, extract only the number.
"""

    uploaded = genai.upload_file(path=str(pdf_path), mime_type="application/pdf")
    if debug:
        print("[PDF] Uploaded:", uploaded.name)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [uploaded, prompt],
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    response_text = getattr(response, "text", "") or ""
    candidate = _extract_first_json_object(response_text)

    try:
        parsed = json.loads(candidate)
    except Exception:
        if debug:
            print("[PDF] Raw response (truncated):", response_text[:2000])
        return None

    # Best-effort cleanup
    try:
        genai.delete_file(uploaded.name)
    except Exception:
        pass

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini connectivity test + PDF processing")
    parser.add_argument("--pdf", type=str, default="", help="Path to an ECG report PDF to process")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--debug", action="store_true", help="Print extra details / raw response")
    parser.add_argument("--text-only", action="store_true", help="Only ping the model with text (no PDF upload)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    key = _load_gemini_key_from_secrets(project_root)
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not found. Ensure `.streamlit/secrets.toml` exists and contains GEMINI_API_KEY."
        )

    # Keep consistent with the app's current dependency.
    try:
        import google.generativeai as genai  # deprecated lib used in this project
    except ModuleNotFoundError as e:
        raise SystemExit(
            "google-generativeai is not installed in this Python environment. "
            "Run the script using the same environment as the Streamlit app, e.g.:\n\n"
            "  conda run -n myenv python gemini_pdf_connectivity_test.py --text-only --model \"gemini-2.5-flash\"\n\n"
            "or install it:\n\n"
            "  pip install google-generativeai\n"
        ) from e

    genai.configure(api_key=key)

    print("[INFO] Gemini key loaded from secrets. Model:", args.model)

    try:
        _gemini_text_ping(genai, args.model)
    except Exception as e:
        raise SystemExit(f"Gemini ping failed (model/api connectivity issue): {e}")

    if args.text_only:
        return

    if not args.pdf:
        raise SystemExit("Please provide --pdf to process a PDF, or use --text-only.")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    print("[INFO] Processing PDF via File API:", pdf_path)
    try:
        parsed = _gemini_pdf_extract(genai, args.model, pdf_path, debug=args.debug)
    except Exception as e:
        print("[FAIL] PDF extraction call failed:", e)
        return
    if parsed is None:
        print("[FAIL] Could not parse JSON from Gemini PDF response.")
        return

    print("[OK] PDF extraction succeeded. Parsed keys:", list(parsed.keys()))
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()

