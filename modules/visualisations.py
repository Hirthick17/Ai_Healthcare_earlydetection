import plotly.graph_objects as go
import pandas as pd, numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score
)

def confusion_matrix_chart(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = [['TN\n(Correct Healthy)', 'FP\n(False Alarm)'],
              ['FN\n(Missed Disease)', 'TP\n(Correct At Risk)']]
    vals   = [[tn, fp], [fn, tp]]
    colors = [['#059669','#D97706'],['#DC2626','#059669']]

    fig = go.Figure()
    for i in range(2):
        for j in range(2):
            fig.add_trace(go.Heatmap(
                z=[[vals[i][j]]], x=['Pred: Healthy','Pred: At Risk'][j:j+1],
                y=['True: Healthy','True: At Risk'][i:i+1],
                colorscale=[[0,colors[i][j]],[1,colors[i][j]]],
                showscale=False, text=[[f'{labels[i][j]}\n{vals[i][j]}']],
                texttemplate='%{text}', textfont={'size':14,'color':'white'},
            ))
    fig.update_layout(
        title='Confusion Matrix — AI vs Doctor Report',
        paper_bgcolor='#0B1D3A', plot_bgcolor='#132952',
        font=dict(color='white'), height=350,
    )
    metrics = {
        'Accuracy':    (tp+tn)/(tp+tn+fp+fn),
        'Sensitivity': tp/(tp+fn) if (tp+fn)>0 else 0,
        'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
        'F1 Score':    f1_score(y_true, y_pred),
    }
    return fig, metrics

def roc_curve_chart(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
        line=dict(color='#0EA5C8',width=2.5),
        name=f'AUC = {auc:.4f}'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
        line=dict(color='gray',dash='dash'), name='Random',
        showlegend=False))
    fig.update_layout(
        title=f'ROC Curve  (AUC = {auc:.4f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='#0B1D3A', plot_bgcolor='#132952',
        font=dict(color='white'), height=380,
    )
    return fig, auc

def bland_altman_chart(ai_values, reference_values, label='SpO2 (%)'):
    ai   = np.array(ai_values)
    ref  = np.array(reference_values)
    means = (ai + ref) / 2
    diffs = ai - ref
    bias  = np.mean(diffs)
    sd    = np.std(diffs)
    loa_hi = bias + 1.96 * sd
    loa_lo = bias - 1.96 * sd

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=means, y=diffs, mode='markers',
        marker=dict(color='#0EA5C8', size=7, opacity=0.7),
        name='Data points'
    ))
    fig.add_hline(y=bias,   line_color='white',  line_width=2,
        annotation_text=f'Bias={bias:+.3f}',annotation_font_color='white')
    fig.add_hline(y=loa_hi, line_color='#EF4444', line_dash='dash',
        annotation_text=f'+1.96 SD={loa_hi:+.3f}',annotation_font_color='#EF4444')
    fig.add_hline(y=loa_lo, line_color='#EF4444', line_dash='dash',
        annotation_text=f'-1.96 SD={loa_lo:+.3f}',annotation_font_color='#EF4444')
    fig.update_layout(
        title=f'Bland-Altman — {label} Agreement (AI vs Device)',
        xaxis_title=f'Mean of AI and Device ({label})',
        yaxis_title='AI - Device',
        paper_bgcolor='#0B1D3A', plot_bgcolor='#132952',
        font=dict(color='white'), height=380,
    )
    return fig, {'bias':round(bias,4),'loa_hi':round(loa_hi,4),'loa_lo':round(loa_lo,4)}
