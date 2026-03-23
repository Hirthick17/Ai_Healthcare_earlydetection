"""
app.py
------
Interactive Streamlit Dashboard for the AI-Based Vital Analysis System.
Mobile-First "Soft Neumorphic Slate & Indigo" Aesthetic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Bare imports — let the server surface the real error if any dependency is missing
from src.master_router import process_medical_document
from src.clinical_rules import calculate_news2_score

# ── 1. General Configuration ───────────────────────────────────────────────
# The theme is natively enforced via .streamlit/config.toml
st.set_page_config(page_title="Vitian Cardiac AI", layout="centered", initial_sidebar_state="collapsed", page_icon="🫀")

def inject_custom_css():
    st.markdown("""
<style>
/* Style the main data containers as premium cards */
div[data-testid="stVerticalBlock"] > div {
    background-color: #ffffff;
    border-radius: 1.5rem; /* 24px heavily rounded corners */
    padding: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.08); /* Soft indigo shadow */
    border: 1px solid #f1f5f9;
    margin-bottom: 1rem;
}
/* Remove card styling from the very top container to prevent stacking bugs */
div[data-testid="stVerticalBlock"] { background: transparent; box-shadow: none; border: none; padding: 0;}

/* Style the file uploader to look modern */
section[data-testid="stFileUploadDropzone"] {
    background-color: #f8fafc;
    border: 2px dashed #c7d2fe; /* Indigo-200 */
    border-radius: 1rem;
}

/* Gradient Headers for that "AI Tech" feel */
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #4f46e5, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Hide default streamline noise */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stSidebar"] {display: none;} /* Completely hide sidebar if user toggles it */

/* Customize streamlit button */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
    color: white; border: none; border-radius: 100px; font-weight: 700;
    padding: 12px 24px; transition: transform 0.1s, box-shadow 0.1s;
    box-shadow: 0 4px 14px 0 rgba(79,70,229,0.39);
}
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px 0 rgba(79,70,229,0.45); color: white; border: none; }
div.stButton > button:active { transform: translateY(0); color: white; }

/* Tabs Styling */
button[data-baseweb="tab"] { font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

inject_custom_css()

# Session State
if "triage_report" not in st.session_state:
    st.session_state.triage_report = None
if "npy_path" not in st.session_state:
    st.session_state.npy_path = None

st.title("Vitian Cardiac AI")

# Triage Mode Selector Tabs
tab1, tab2 = st.tabs(["🩺 Single Patient (ECG)", "📊 Batch Dataset (Vitals)"])

# ==============================================================================
# TAB 1: Single Patient (ECG)
# ==============================================================================
with tab1:
    with st.container():
        st.subheader("1. Upload Medical Document")
        
        uploaded_file = st.file_uploader("Upload ECG Scan (PDF, JPG, PNG)", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
        
        page_num = 1
        rotation_k = 0
        
        if uploaded_file:
            if uploaded_file.name.lower().endswith(".pdf"):
                page_num = st.number_input("PDF Page Number", min_value=1, value=1, step=1)
                
            st.markdown("**🔄 Rotate Document (Optional)**")
            rotation_label = st.radio(
                "Rotation angle",
                options=["0° (no rotation)", "90° clockwise", "180°", "270° clockwise"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )
            _rot_map = {"0° (no rotation)": 0, "90° clockwise": 3, "180°": 2, "270° clockwise": 1}
            rotation_k = _rot_map[rotation_label]
                
            run_btn = st.button("Run Diagnostic Analysis", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Analyzing patient vitals..."):
                    try:
                        report, npy_path = process_medical_document(uploaded_file, page_num, rotate_k=rotation_k)
                        st.session_state.triage_report = report
                        st.session_state.npy_path = npy_path
                    except Exception as e:
                        st.error(f"Error during processing: {e}")

    # ── Comparison Dashboard ────────────────────────────────
    if st.session_state.triage_report:
        report = st.session_state.triage_report
        npy_path = st.session_state.npy_path
        
        doc_class = report["doctor_report"]["predicted_class"]
        ai_class  = report["ai_prediction"]["predicted_class"]
        is_mismatch = (doc_class != ai_class)
        
        # ── Mismatch Alerts ──
        if not is_mismatch:
            st.success(f"✅ Consensus Reached: Both AI and Clinic confirm **{ai_class}**.")
        else:
            st.error(f"⚠️ Clinical Discrepancy Detected.\n\nAI predicts **{ai_class}**, but Doctor reports **{doc_class}**.")
            
        with st.container():
            st.subheader("🩺 Clinical Report")
            doc = report["doctor_report"]
            
            st.metric("Clinical Class Mapping", doc["predicted_class"])
            params = doc["parameters"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Heart Rate", f"{params.get('ventricular_rate') or '--'} bpm")
            c2.metric("PR Interval", f"{params.get('pr_interval') or '--'} ms")
            c3.metric("QRS Duration", f"{params.get('qrs_duration') or '--'} ms")
            
            st.info(f"**Raw finding:** {doc['raw_text']}")

        with st.container():
            st.subheader("🤖 AI Analysis (DenseNet1D)")
            ai = report["ai_prediction"]
            
            st.metric("Neural Network Prediction", ai["predicted_class"])
            
            conf = ai["confidence"]
            st.markdown(f"**Confidence:** {conf*100:.1f}%")
            
            pb = st.progress(0)
            target = int(conf * 100)
            for i in range(target + 1):
                pb.progress(i)
                time.sleep(0.003)
                
            st.caption("Class Probability Distribution:")
            for cls, prob in ai["probabilities"].items():
                st.markdown(f"<small><b>{cls}</b>: {prob*100:.1f}%</small>", unsafe_allow_html=True)
                st.progress(prob)

        top_leads = []
        if is_mismatch and "lead_importance" in ai:
            with st.container():
                st.subheader("🧠 Model Explainability (XAI)")
                importances = ai["lead_importance"]
                df_imp = pd.DataFrame(list(importances.items()), columns=["Lead", "Importance"])
                top_leads = df_imp.sort_values(by="Importance", ascending=False).head(2)["Lead"].tolist()
                
                if top_leads:
                    st.info(f"The AI model's prediction of **{ai_class}** was primarily driven by anomalies detected in Lead **{top_leads[0]}**. Please manually review this specific channel on the extracted waveform below to verify the model's finding.")
                    
                st.bar_chart(df_imp.set_index("Lead"))

        if npy_path and Path(npy_path).exists():
            data = np.load(npy_path)
            if data.shape == (1000, 12):
                with st.container():
                    st.subheader("📈 Live Waveform Data")
                    
                    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                    plot_leads = top_leads if len(top_leads) >= 2 else ["II"]
                    
                    fig = make_subplots(rows=len(plot_leads), cols=1, shared_xaxes=True, subplot_titles=[f"Lead {n}" for n in plot_leads])
                    
                    for idx, lead_name in enumerate(plot_leads):
                        l_idx = LEAD_NAMES.index(lead_name)
                        fig.add_trace(go.Scatter(
                            y=data[:, l_idx], mode='lines', 
                            line=dict(color='#4f46e5', width=3, shape='spline'),
                            fill='tozeroy', fillcolor='rgba(79,70,229,0.1)',
                            name=f"Lead {lead_name}"
                        ), row=idx+1, col=1)
                        
                        fig.update_yaxes(showgrid=True, gridcolor='#e2e8f0', zeroline=False, row=idx+1, col=1)
                        fig.update_xaxes(showgrid=False, zeroline=False, title_text=f"Lead {lead_name} (Resampled)" if idx == len(plot_leads)-1 else "", row=idx+1, col=1)
                        
                    fig.update_layout(
                        height=200 * len(plot_leads) + 50,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False, margin=dict(l=0, r=0, t=20, b=0),
                        font=dict(color='#0f172a')
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==============================================================================
# TAB 2: Batch Dataset (Vitals)
# ==============================================================================
with tab2:
    with st.container():
        st.markdown("### Import Vitals Dataset")
        st.info(
            "**Expected CSV schema:** `patient_id`, `ecg_file_path` (path to .npy), "
            "`heart_rate`, `spo2`, `temperature`\n\n"
            "The `ecg_file_path` column should contain relative or absolute paths to 12-lead "
            "`.npy` arrays (shape 1000×12) stored on the same server."
        )

        csv_file = st.file_uploader(
            "Upload CSV feed", type=["csv", "xlsx"], label_visibility="collapsed"
        )

    if csv_file:
        try:
            df_raw = (
                pd.read_csv(csv_file)
                if csv_file.name.endswith(".csv")
                else pd.read_excel(csv_file)
            )
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
            df_raw = None

        if df_raw is not None:
            st.subheader("Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)

            run_batch = st.button(
                "Run AI Triage Analysis", type="primary", use_container_width=True
            )

            if run_batch:
                from src.triage_synthesizer import process_batch_dataset

                st.markdown("**Running AI Triage on Batch...**")
                progress_bar = st.progress(0)

                def update_progress(fraction):
                    progress_bar.progress(min(1.0, fraction))

                with st.spinner("Fusing DenseNet ECG scores with NEWS2 vitals..."):
                    results_df = process_batch_dataset(df_raw, progress_cb=update_progress)

                progress_bar.progress(1.0)
                st.success(f"✅ Triage complete for {len(results_df)} patients.")

                # ── Batch Summary Metrics ──────────────────────────────────────
                total_patients = len(results_df)
                critical_count = int((results_df["Final_Triage_Class"] == "Critical").sum())
                avg_risk = round(results_df["Total_Risk"].mean(), 3)

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Patients", total_patients)
                m2.metric("🔴 Critical Alerts", critical_count)
                m3.metric("Avg. Total Risk", avg_risk)

                # ── Styled Triage Table ───────────────────────────────────────
                def highlight_triage(row):
                    if row.get("Final_Triage_Class") == "Critical":
                        return ["background-color: #fee2e2; color: #991b1b"] * len(row)
                    return ["background-color: #dcfce7; color: #166534"] * len(row)

                st.dataframe(
                    results_df.style.apply(highlight_triage, axis=1),
                    use_container_width=True,
                )

