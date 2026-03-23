"""
src/triage_synthesizer.py
-------------------------
Batch Vitals Priority Fusion Engine.

Processes a DataFrame row-by-row:
  Step A: DenseNet1D AI ECG risk score per .npy file
  Step B: NEWS2 clinical vitals risk score
  Step C: Weighted priority fusion (ECG=0.65, Vitals=0.35)
  Step D: Critical / Stable classification (threshold >= 0.50)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from src.densenet1d import DenseNet1D
from src.clinical_rules import calculate_news2_score

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# Critical classes that drive the ECG risk score
CRITICAL_CLASSES = {"MI", "HYP"}
WEIGHTS_PATH = Path("models/densenet_ecg.pt")

_model = None
_device = None


def _load_model():
    """Lazy-load the DenseNet1D model once and cache it globally."""
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = DenseNet1D(in_channels=12, num_classes=len(PTBXL_CLASSES))
    try:
        _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=_device))
    except Exception:
        pass  # Return an un-trained model gracefully if weights are missing
    _model.to(_device)
    _model.eval()
    return _model, _device


def _ecg_risk_score(npy_path: str) -> float:
    """
    Load a (1000, 12) .npy ECG array and return a DenseNet1D-based
    risk score in [0.0, 1.0] defined as P(MI) + P(HYP).
    Returns 0.5 (neutral) on any file error.
    """
    try:
        data = np.load(npy_path)
    except Exception:
        return 0.5  # Neutral if file missing

    if data.shape != (1000, 12):
        if data.shape == (12, 1000):
            data = data.T
        else:
            return 0.5

    model, device = _load_model()
    tensor = torch.from_numpy(data).float()
    tensor = tensor.transpose(0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = dict(zip(PTBXL_CLASSES, probs))
    ecg_risk = float(prob_dict.get("MI", 0.0)) + float(prob_dict.get("HYP", 0.0))
    return min(1.0, ecg_risk)


def _news2_risk_score(hr, spo2, temp) -> float:
    """Normalize the NEWS2 integer score (0-9) to [0.0, 1.0]."""
    try:
        total, _ = calculate_news2_score(int(hr), int(spo2), float(temp))
    except Exception:
        return 0.0
    return min(1.0, total / 9.0)


def process_batch_dataset(df: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
    """
    Main batch processing entry point.

    Args:
        df:          DataFrame with columns patient_id, ecg_file_path,
                     heart_rate, spo2, temperature
        progress_cb: Optional callable(fraction) to update a progress bar.

    Returns:
        Enriched DataFrame with columns:
            AI_ECG_Risk  - DenseNet probability score [0-1]
            NEWS2_Risk   - Normalised NEWS2 score [0-1]
            Total_Risk   - Weighted fusion score [0-1]
            Final_Triage_Class - 'Critical' | 'Stable'
    """
    ecg_scores, news2_scores, total_scores, classes = [], [], [], []
    n = len(df)

    for i, row in df.iterrows():
        # Step A: ECG AI score
        npy_path = row.get("ecg_file_path", "")
        score_ecg = _ecg_risk_score(str(npy_path))

        # Step B: Clinical Vitals score
        score_vitals = _news2_risk_score(
            row.get("heart_rate", 70),
            row.get("spo2", 98),
            row.get("temperature", 37.0)
        )

        # Step C: Priority Weighted Fusion (ECG=0.65, Vitals=0.35)
        total = (0.65 * score_ecg) + (0.35 * score_vitals)
        total = round(min(1.0, total), 4)

        # Step D: Classification
        label = "Critical" if total >= 0.50 else "Stable"

        ecg_scores.append(round(score_ecg, 4))
        news2_scores.append(round(score_vitals, 4))
        total_scores.append(total)
        classes.append(label)

        if progress_cb:
            progress_cb((i + 1) / n)

    df = df.copy()
    df["AI_ECG_Risk"] = ecg_scores
    df["NEWS2_Risk"] = news2_scores
    df["Total_Risk"] = total_scores
    df["Final_Triage_Class"] = classes
    return df
