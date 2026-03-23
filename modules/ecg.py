import numpy as np, math
import plotly.graph_objects as go

def generate_ecg(bpm=72, noise=0.02, n_beats=4, fs=250):
    tpb = 60.0 / bpm
    ns  = int(n_beats * tpb * fs)
    ecg = np.zeros(ns)
    bs  = int(tpb * fs)

    for beat in range(n_beats):
        off = beat * bs
        waves = [
            (0.20, 0.18, 12),   # P wave
            (0.38,-0.12, 3),    # Q wave
            (0.41, 1.00, 4),    # R wave (tallest)
            (0.44,-0.20, 3.5),  # S wave
            (0.65, 0.30, 18),   # T wave
        ]
        for (frac, amp, sig) in waves:
            c = off + int(frac * bs)
            lo = max(0, c - int(3*sig))
            hi = min(ns, c + int(3*sig))
            for j in range(lo, hi):
                ecg[j] += amp * math.exp(-((j-c)**2)/(2*sig**2))

    ecg += np.random.normal(0, noise, ns)
    step = max(1, ns // 500)
    return ecg[::step][:500]

def ecg_chart(signal, bpm=72, at_risk=False):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, height=0.5, distance=30)
    line_col = '#FF3B5C' if at_risk else '#00D4FF'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=signal, mode='lines',
        line=dict(color=line_col, width=1.5), name='ECG'
    ))
    fig.add_trace(go.Scatter(
        x=peaks, y=signal[peaks], mode='markers',
        marker=dict(color='#FF6B35', size=8, symbol='diamond'),
        name='R-peaks'
    ))
    fig.update_layout(
        title=f'ECG Waveform — {bpm} bpm',
        yaxis_title='Amplitude (mV)',
        paper_bgcolor='#0B1D3A', plot_bgcolor='#0F2444',
        font=dict(color='white'), height=280,
        showlegend=True, margin=dict(t=40,b=20),
    )
    return fig

def gauge_chart(value, label, unit, min_val, max_val,
                normal_lo, normal_hi, color='#0EA5C8'):
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=value,
        title=dict(text=f'{label} ({unit})', font=dict(color='white',size=14)),
        delta=dict(reference=(normal_lo+normal_hi)/2,
                   valueformat='.1f'),
        gauge=dict(
            axis=dict(range=[min_val, max_val],
                      tickcolor='white', tickfont=dict(color='white')),
            bar=dict(color=color),
            bgcolor='#132952',
            steps=[
                dict(range=[min_val, normal_lo], color='#7F1D1D'),
                dict(range=[normal_lo, normal_hi], color='#064E3B'),
                dict(range=[normal_hi, max_val], color='#7F1D1D'),
            ],
            threshold=dict(
                line=dict(color='white',width=2),
                value=value
            )
        ),
        number=dict(font=dict(color='white',size=28)),
    ))
    fig.update_layout(
        paper_bgcolor='#0B1D3A', height=220,
        margin=dict(t=40,b=0,l=20,r=20)
    )
    return fig
