"""
src/explainability.py
SHAP GradientExplainer for DenseNet1D — generates per-lead importance scores.
Output: dict with 'lead_scores' (list of 12 values) and 'lead_names'.
"""

import numpy as np
import torch

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def compute_lead_importance(model, input_tensor: torch.Tensor,
                             n_background: int = 50) -> dict:
    """
    Computes per-lead importance using Input × Gradient saliency.
    This avoids SHAP shape ambiguity for binary single-output 1D-CNNs.

    Parameters
    ----------
    model        : DenseNet1D in eval mode
    input_tensor : (1, 12, 1000) float32 tensor

    Returns
    -------
    dict with keys:
        lead_names   – list of 12 standard lead labels
        lead_scores  – list of 12 mean |saliency| averaged over time
        top_lead     – name of the most important lead
    """
    model.eval()
    x = input_tensor.clone().requires_grad_(True)

    # Forward pass
    logit = model(x)
    logit.backward()   # backprop the scalar output directly

    # Gradient * Input saliency map — shape: (1, 12, 1000)
    saliency = (x.grad * x).abs().detach().numpy()
    saliency  = saliency[0]  # → (12, 1000)

    lead_scores = saliency.mean(axis=1).tolist()  # list of exactly 12 floats

    top_idx  = int(np.argmax(lead_scores))
    top_lead = LEAD_NAMES[top_idx]

    return {
        'lead_names' : LEAD_NAMES,
        'lead_scores': [round(s, 6) for s in lead_scores],
        'top_lead'   : top_lead,
        'top_score'  : round(lead_scores[top_idx], 6),
    }



def build_overlay_figure(input_tensor: torch.Tensor, shap_result: dict):
    """
    Returns a Plotly figure showing the ECG waveform of the most important lead
    overlaid with its SHAP importance as a shaded area.
    """
    import plotly.graph_objects as go

    lead_names  = shap_result['lead_names']
    lead_scores = shap_result['lead_scores']
    top_lead    = shap_result['top_lead']
    top_idx     = lead_names.index(top_lead)

    ecg_signal = input_tensor[0, top_idx].numpy()       # (1000,)
    time_axis  = np.arange(len(ecg_signal)) / 100.0     # assuming 100 Hz after padding

    # Normalize shap score to ECG amplitude for visual overlay
    importance_amplitude = lead_scores[top_idx] * 10
    importance_band = np.full_like(ecg_signal, importance_amplitude)

    fig = go.Figure()

    # ECG waveform
    fig.add_trace(go.Scatter(
        x=time_axis, y=ecg_signal,
        mode='lines', name=f'ECG — Lead {top_lead}',
        line=dict(color='#0EA5C8', width=1.5)
    ))

    # SHAP importance overlay (filled region)
    fig.add_trace(go.Scatter(
        x=time_axis, y=importance_band,
        mode='lines', name=f'SHAP Importance (Lead {top_lead})',
        line=dict(color='#EF4444', width=1, dash='dot'),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)'
    ))

    # Lead importance bar chart (secondary, using annotations)
    fig.update_layout(
        title=f'ECG Waveform + SHAP Lead Importance  |  Top Lead: {top_lead}',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV, normalized)',
        paper_bgcolor='#0B1D3A',
        plot_bgcolor='#132952',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=380,
    )

    return fig


def build_lead_bar(shap_result: dict):
    """Standalone bar chart ranking all 12 leads by SHAP importance."""
    import plotly.graph_objects as go

    names  = shap_result['lead_names']
    scores = shap_result['lead_scores']
    colors = ['#EF4444' if s == max(scores) else '#3B82F6' for s in scores]

    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker_color=colors,
        text=[f'{s:.4f}' for s in scores],
        textposition='outside'
    ))
    fig.update_layout(
        title='Lead Importance Heat Map (SHAP Mean |φᵢ|)',
        xaxis_title='ECG Lead',
        yaxis_title='Mean |SHAP Value|',
        paper_bgcolor='#0B1D3A',
        plot_bgcolor='#132952',
        font=dict(color='white'),
        height=320,
    )
    return fig
