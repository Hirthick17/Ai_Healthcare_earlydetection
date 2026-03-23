# modules/explainer.py — Step 4: Master XAI Report
import shap, numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib

BASE = Path(__file__).parent.parent

def get_shap_values(X_scaled_row, feature_names):
    """
    Evaluates exclusively the CatBoost arm of the hard-voting TriBoostCardio ensemble
    to yield exact, interpretable Tree values for the 4 explicit features.
    """
    ensemble = joblib.load(BASE / 'models' / 'master_ensemble.pkl')
    
    # Extract CatBoost sub-model which inherently powers SHAP structure natively
    cat_model = ensemble.named_estimators_['cat']
    explainer = shap.TreeExplainer(cat_model)
    shap_vals = explainer.shap_values(X_scaled_row)
    
    expected_val = explainer.expected_value
    base_value = float(expected_val[0]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val)
    vals = shap_vals[0] if len(np.array(shap_vals).shape) == 2 else shap_vals

    order = np.argsort(np.abs(vals))[::-1]
    return {
        'features':   [feature_names[i] for i in order],
        'values':     [float(vals[i])    for i in order],
        'data':       [float(X_scaled_row[0][i]) for i in order],
    }

def waterfall_chart(shap_result):
    features = shap_result['features']
    values   = shap_result['values']
    
    # Red for Risk increase, Green for Risk decrease (Protective)
    colors   = ['#EF4444' if v > 0 else '#10B981' for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.2f}' for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        title='AI Reasoning: Risk Drivers (Red) vs Protective Factors (Green)',
        xaxis_title='SHAP Impact Magnitude',
        paper_bgcolor='#0B1D3A',
        plot_bgcolor='#132952',
        font=dict(color='white'),
        height=320,
    )
    fig.add_vline(x=0, line_color='white', line_width=1)
    return fig
