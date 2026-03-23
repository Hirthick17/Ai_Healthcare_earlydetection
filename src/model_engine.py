"""
src/model_engine.py
DenseNet1D — Binary Cardiac Risk Triage Engine
Input : (batch, 12, 1000) float32 tensor
Output: (batch, 1)  logit  →  sigmoid → 0=Low Risk, 1=High Risk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── PTB-XL labeling constraints ───────────────────────────────────────────────
# Classes that must trigger "High Risk" (indices inside our 20-class vector)
# NORM=0, HYP=3 → LOW RISK  |  MI=1,STTC=2,CD=4 → HIGH RISK
HIGH_RISK_INDICES = [1, 2, 4]          # MI, STTC, CD
LOW_RISK_INDICES  = [0, 3]             # NORM, HYP


# ── Network Blocks ─────────────────────────────────────────────────────────────
class _DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.bn1  = nn.BatchNorm1d(in_ch)
        self.c1   = nn.Conv1d(in_ch, growth * 4, 1, bias=False)
        self.bn2  = nn.BatchNorm1d(growth * 4)
        self.c2   = nn.Conv1d(growth * 4, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.c1(F.leaky_relu(self.bn1(x), 0.1))
        out = self.c2(F.leaky_relu(self.bn2(out), 0.1))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, n_layers: int, in_ch: int, growth: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(_DenseLayer(in_ch + i * growth, growth))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn   = nn.BatchNorm1d(in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.leaky_relu(self.bn(x), 0.1)))


# ── Main Architecture ──────────────────────────────────────────────────────────
class DenseNet1D(nn.Module):
    """
    Lightweight 1D-DenseNet tuned for 12-lead ECG @ 1000 time-steps.
    Two dense blocks with transition layers → global average pool → binary classifier.
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 1):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        # Dense backbone
        self.db1    = _DenseBlock(n_layers=4, in_ch=64,  growth=32)   # out: 192
        self.trans1 = _Transition(192, 96)
        self.db2    = _DenseBlock(n_layers=4, in_ch=96,  growth=32)   # out: 224
        self.trans2 = _Transition(224, 112)

        # Head — binary logit
        self.head = nn.Sequential(
            nn.Linear(112, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):                    # x: (B, 12, T)
        out = self.stem(x)
        out = self.db1(out);   out = self.trans1(out)
        out = self.db2(out);   out = self.trans2(out)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)   # (B, 112)
        return self.head(out)                              # (B, 1)


# ── Convenience helpers ────────────────────────────────────────────────────────
def load_model(weights_path: str, device: str = 'cpu') -> DenseNet1D:
    model = DenseNet1D()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def preprocess(array) -> torch.Tensor:
    """
    array : numpy (T, 12)  or  (12, T)  or  (1, T, 12)
    returns: torch float32 tensor (1, 12, T) ready for inference
    """
    import numpy as np
    arr = np.array(array, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.shape[0] != 12:          # was (T, 12) → transpose
        arr = arr.T
    # Normalize per lead
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1, keepdims=True) + 1e-8
    arr  = (arr - mean) / std
    # Pad / truncate to 1000 samples
    T = 1000
    if arr.shape[1] < T:
        arr = np.pad(arr, ((0, 0), (0, T - arr.shape[1])))
    else:
        arr = arr[:, :T]
    return torch.tensor(arr).unsqueeze(0)   # (1, 12, 1000)


def predict(model: DenseNet1D, tensor: torch.Tensor):
    """Returns (label_str, probability, risk_flag)."""
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    high_risk = prob >= 0.5
    label = "HIGH RISK" if high_risk else "LOW RISK"
    return label, round(prob, 4), high_risk
