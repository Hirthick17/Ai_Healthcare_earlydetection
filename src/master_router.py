"""
src/master_router.py
--------------------
Master Orchestration Router for the AI-Based Vital Analysis System.
Routes file uploads to the Computer Vision Pipeline (signal_extractor),
the DenseNet1D AI model (inference_engine), and the Clinical OCR (report_extractor).
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import tempfile
import uuid

from src.signal_extractor import extract_ecg_array
from src.report_extractor import run_pipeline, classify_diagnosis
from src.densenet1d import DenseNet1D

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

def pdf_page_to_image(pdf_path, page_idx=0):
    """Converts a single PDF page into a PNG image using PyMuPDF (fitz)."""
    import fitz
    import cv2
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    
    out_path = str(Path(pdf_path).with_suffix('.png'))
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path

def process_medical_document(uploaded_file, page_num=1, rotate_k=0):
    """
    Main pipeline entry point.
    Returns: (final_triage_report (dict), npy_path (str))
    """
    # Use OS temp directory to avoid Streamlit Cloud working-dir permission issues.
    temp_root = Path(tempfile.gettempdir()) / "healthmonitoring_system"
    temp_dir = temp_root / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    file_path = temp_dir / safe_name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    img_path = str(file_path)
    
    # ── 1. PDF Conversion (If Applicable) ──────────────────────────────
    if file_path.suffix.lower() == ".pdf":
        img_path = pdf_page_to_image(str(file_path), page_idx=page_num - 1)
        
    # ── Apply Rotation if requested ────────────────────────────────────
    if rotate_k != 0:
        import cv2
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            img_bgr = np.rot90(img_bgr, k=rotate_k)
            cv2.imwrite(img_path, img_bgr)
            
    # ── 2. Signal Extraction (Computer Vision) ─────────────────────────
    npy_path = str(temp_dir / f"{file_path.stem}_extracted.npy")
    # This automatically produces (1000, 12) array and saves it
    ecg_array = extract_ecg_array(img_path, out_path=npy_path, target_len=1000, debug=False)
    
    # ── 3. AI Inference (DenseNet1D) ───────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet1D(in_channels=12, num_classes=5)
    
    # Check if a 5-class weight file exists
    weights_path = Path("models/densenet_ecg_5class.pt")
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        # Fallback for UI demo purposes if standard weights aren't supplied
        pass
        
    model.to(device)
    model.eval()
    
    tensor_data = torch.from_numpy(ecg_array).float()
    # Transpose from (1000, 12) to (1, 12, 1000)
    tensor_data = tensor_data.transpose(0, 1).unsqueeze(0).to(device)
    
    with torch.enable_grad():
        x = tensor_data.clone().requires_grad_(True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        
        # Saliency w.r.t the predicted class
        logits[0, pred_idx].backward()
        saliency_scores = (x.grad * x).abs().detach().cpu().numpy()[0].mean(axis=1).tolist()
        
    ai_class = PTBXL_CLASSES[pred_idx]
    ai_conf = float(probs[pred_idx])
    
    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lead_importance = {name: round(score, 6) for name, score in zip(LEAD_NAMES, saliency_scores)}
    
    # ── 4. Doctor Report (pdfplumber pipeline — PDF inputs only) ───────
    extraction_error = None
    try:
        if file_path.suffix.lower() == ".pdf":
            report = run_pipeline(str(file_path), save_output=False, rotate_k=rotate_k)
            parameters = report.get("parameters", {})
            diag_text  = report.get("raw_text", "")
            doc_class  = report.get("predicted_class", classify_diagnosis(diag_text))
        else:
            # Image-only upload: skip PDF text extraction
            raise ValueError("Input is an image, not a PDF — skipping pdfplumber pipeline.")
    except Exception as e:
        print(f"Warning: Report extraction failed: {e}")
        extraction_error = str(e)
        doc_class  = "NORM"
        parameters = {"ventricular_rate": None, "pr_interval": None, "qrs_duration": None}
        diag_text  = "No text extracted."
    
    # ── 5. Consensus Compilation ───────────────────────────────────────
    consensus_match = (ai_class == doc_class)
    
    final_triage_report = {
        "status": "Match" if consensus_match else "Mismatch",
        "doctor_report": {
            "predicted_class": doc_class,
            "parameters": parameters,
            "raw_text": diag_text,
            "extraction_error": extraction_error,
        },
        "ai_prediction": {
            "predicted_class": ai_class,
            "confidence": ai_conf,
            "probabilities": {cls: float(p) for cls, p in zip(PTBXL_CLASSES, probs)},
            "lead_importance": lead_importance
        }
    }
    
    # Write JSON artifact
    json_path = temp_dir / "final_triage_report.json"
    with open(json_path, "w") as f:
        json.dump(final_triage_report, f, indent=4)
        
    return final_triage_report, npy_path
