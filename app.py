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
if "single_upload_id" not in st.session_state:
    st.session_state.single_upload_id = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "batch_upload_id" not in st.session_state:
    st.session_state.batch_upload_id = None

st.title("Vitian Cardiac AI")

# Triage Mode Selector Tabs
tab1, tab2 = st.tabs(["🩺 Single Patient (ECG)", "📊 Batch Dataset (Vitals)"])

# ==============================================================================
# TAB 1: Single Patient (ECG)
# ==============================================================================
with tab1:
    with st.container():
        st.subheader("1. Upload Medical Document")
        st.caption("Designed for quick doctor use: upload, choose page/rotation, then run.")
        
        uploaded_file = st.file_uploader("Upload ECG Scan (PDF, JPG, PNG)", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
        
        page_num = 1
        rotation_k = 0
        
        if uploaded_file:
            upload_id = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
            if st.session_state.single_upload_id != upload_id:
                # New file selected: clear previous results so the UI is always consistent.
                st.session_state.single_upload_id = upload_id
                st.session_state.triage_report = None
                st.session_state.npy_path = None
                st.info("New document selected. Click `Run Diagnostic Analysis` to refresh results.")

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
            doc    = report["doctor_report"]
            params = doc["parameters"]

            st.metric("Clinical Class", doc["predicted_class"])

            # ── Row 1: Main intervals ──────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("❤️ Heart Rate",
                      f"{params.get('ventricular_rate') or '—'} bpm")
            c2.metric("⏱ PR Interval",
                      f"{params.get('pr_interval') or '—'} ms")
            c3.metric("📉 QRS Duration",
                      f"{params.get('qrs_duration') or '—'} ms")

            # ── Row 2: P Duration + QT/QTc ────────────────────────────────
            c4, c5, c6 = st.columns(3)
            c4.metric("P Duration",
                      f"{params.get('p_duration') or '—'} ms")
            c5.metric("QT Interval",
                      f"{params.get('qt_interval') or '—'} ms")
            c6.metric("QTc Interval",
                      f"{params.get('qtc_interval') or '—'} ms")

            # ── Row 3: P/QRS/T Axes ───────────────────────────────────────
            p_ax  = params.get('p_axis')
            q_ax  = params.get('qrs_axis')
            t_ax  = params.get('t_axis')
            if any(v is not None for v in [p_ax, q_ax, t_ax]):
                c7, c8, c9 = st.columns(3)
                c7.metric("P Axis",   f"{p_ax if p_ax is not None else '—'}°")
                c8.metric("QRS Axis", f"{q_ax if q_ax is not None else '—'}°")
                c9.metric("T Axis",   f"{t_ax if t_ax is not None else '—'}°")

            # ── Interpretation text ───────────────────────────────────────
            st.info(f"**Interpretation:** {doc['raw_text']}")

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

        # ── XAI: lead importance + interactive waveform viewer ─────────────────
        top_leads = []
        lead_importance = ai.get("lead_importance") if isinstance(ai, dict) else None
        if isinstance(lead_importance, dict) and lead_importance:
            with st.container():
                st.subheader("🧠 Model Explainability (XAI)")
                df_imp = pd.DataFrame(list(lead_importance.items()), columns=["Lead", "Importance"])
                df_imp = df_imp.sort_values(by="Importance", ascending=False)
                top_leads = df_imp.head(3)["Lead"].tolist()

                if top_leads:
                    if is_mismatch:
                        st.error(
                            f"Clinical discrepancy detected. Review the extracted waveform for the most influential lead: "
                            f"**{df_imp.iloc[0]['Lead']}**."
                        )
                    st.caption(
                        f"Top contributing leads for the current prediction: {', '.join(top_leads)}."
                        if len(top_leads) > 1
                        else f"Top contributing lead for the current prediction: {top_leads[0]}."
                    )

                    # Lightweight “animation”: pulse the bar chart to draw attention to the top lead.
                    top_imp = float(df_imp.iloc[0]["Importance"])
                    pulse = st.progress(0, text=f"Highlighting most important lead: {df_imp.iloc[0]['Lead']}")
                    for i in range(0, 101, 5):
                        pulse.progress(i)
                        time.sleep(0.01)

                st.bar_chart(df_imp.set_index("Lead")["Importance"])

        if npy_path and Path(npy_path).exists():
            data = np.load(npy_path)
            if data.shape == (1000, 12):
                with st.container():
                    st.subheader("📈 Extracted Waveform (12-Lead)")

                    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                    lead_options = top_leads if top_leads else ["II"]
                    lead_to_view = st.selectbox("Select lead to visualize", options=lead_options, index=0)

                    l_idx = LEAD_NAMES.index(lead_to_view)
                    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=[f"Lead {lead_to_view}"])

                    fig.add_trace(
                        go.Scatter(
                            y=data[:, l_idx],
                            mode="lines",
                            line=dict(color="#4f46e5", width=3, shape="spline"),
                            fill="tozeroy",
                            fillcolor="rgba(79,70,229,0.1)",
                            name=f"Lead {lead_to_view}",
                        ),
                        row=1,
                        col=1,
                    )

                    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, row=1, col=1)
                    fig.update_xaxes(showgrid=False, zeroline=False, title_text="Resampled time steps", row=1, col=1)
                    fig.update_layout(
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        margin=dict(l=0, r=0, t=20, b=0),
                        font=dict(color="#0f172a"),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==============================================================================
# TAB 2: Batch Dataset (Vitals)
# ==============================================================================
with tab2:
    with st.container():
        st.markdown("### Import Vitals Dataset")
        st.caption("Upload a CSV/XLSX, run the batch, then select a patient to see XAI reasoning.")
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
        upload_id = f"{csv_file.name}:{getattr(csv_file, 'size', 0)}"
        if st.session_state.batch_upload_id != upload_id:
            st.session_state.batch_upload_id = upload_id
            st.session_state.batch_results = None

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

            clear_btn = st.button("Clear Results", use_container_width=True)
            if clear_btn:
                st.session_state.batch_results = None

            if run_batch or st.session_state.batch_results is not None:
                if run_batch:
                    from src.triage_synthesizer import process_batch_dataset
                    st.markdown("**Running AI Triage on Batch...**")
                    progress_bar = st.progress(0)

                    def update_progress(fraction):
                        progress_bar.progress(min(1.0, fraction))

                    with st.spinner("Fusing DenseNet ECG scores with NEWS2 vitals..."):
                        results_df = process_batch_dataset(df_raw, progress_cb=update_progress)

                    progress_bar.progress(1.0)
                    st.session_state["batch_results"] = results_df
                else:
                    results_df = st.session_state["batch_results"]

                st.success(f"✅ Triage complete for {len(results_df)} patients.")

                # ── 1. Summary Metrics Strip ─────────────────────────────────
                total_patients = len(results_df)
                critical_count = int((results_df["Final_Triage_Class"] == "Critical").sum())
                avg_risk = round(results_df["Total_Risk"].mean(), 3)

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Patients", total_patients)
                m2.metric("🔴 Critical Alerts", critical_count)
                m3.metric("Avg. Total Risk", f"{avg_risk:.1%}")

                # ── 2. Cohort Trend Scatter Plot ─────────────────────────────
                st.subheader("Cohort Risk Trends")
                import plotly.express as px

                color_map = {"Critical": "#ef4444", "Stable": "#10b981"}
                scatter_fig = px.scatter(
                    results_df,
                    x="spo2",
                    y="heart_rate",
                    color="Final_Triage_Class",
                    size="Total_Risk",
                    size_max=30,
                    hover_data=["patient_id", "temperature", "Total_Risk"],
                    color_discrete_map=color_map,
                    labels={
                        "spo2": "SpO₂ (%)",
                        "heart_rate": "Heart Rate (BPM)",
                        "Final_Triage_Class": "Triage",
                    },
                    title="Patient Cohort — Risk Distribution",
                )
                scatter_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0f172a"),
                    legend_title_text="Triage Class",
                    xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                )
                st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})

                # ── 3. Modern Data Grid with ProgressColumns ─────────────────
                st.subheader("Triage Results")

                def highlight_triage(row):
                    if row.get("Final_Triage_Class") == "Critical":
                        return ["background-color: #fee2e2; color: #991b1b"] * len(row)
                    return ["background-color: #dcfce7; color: #166534"] * len(row)

                st.dataframe(
                    results_df.style.apply(highlight_triage, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "AI_ECG_Risk": st.column_config.ProgressColumn(
                            "AI ECG Risk", min_value=0.0, max_value=1.0, format="%.0f%%"
                        ),
                        "NEWS2_Risk": st.column_config.ProgressColumn(
                            "NEWS2 Vitals Risk", min_value=0.0, max_value=1.0, format="%.0f%%"
                        ),
                        "Total_Risk": st.column_config.ProgressColumn(
                            "Total Risk Score", min_value=0.0, max_value=1.0, format="%.0f%%"
                        ),
                        "Final_Triage_Class": st.column_config.TextColumn("Triage Class"),
                    },
                )

                # ── 4. Patient Deep Dive (XAI) ────────────────────────────────
                st.divider()
                st.subheader("Patient XAI & Clinical Reasoning")

                selected_pid = st.selectbox(
                    "Select Patient ID for Detailed Analysis",
                    options=results_df["patient_id"].tolist(),
                )

                patient = results_df[results_df["patient_id"] == selected_pid].iloc[0]

                ecg_contribution  = round(float(patient["AI_ECG_Risk"])  * 0.65, 4)
                vitals_contribution = round(float(patient["NEWS2_Risk"]) * 0.35, 4)

                # XAI Contribution Horizontal Bar
                fused_risk = float(patient.get("Total_Risk", 0.0))
                meter = st.progress(0, text="Fused risk meter (0% -> 100%)")
                for i in range(0, int(min(100, fused_risk * 100)) + 1, 5):
                    meter.progress(i)
                    time.sleep(0.01)

                xai_fig = px.bar(
                    x=[ecg_contribution, vitals_contribution],
                    y=["ECG AI (W=0.65)", "Vitals NEWS2 (W=0.35)"],
                    orientation="h",
                    color=["ECG AI (W=0.65)", "Vitals NEWS2 (W=0.35)"],
                    color_discrete_map={
                        "ECG AI (W=0.65)": "#4f46e5",
                        "Vitals NEWS2 (W=0.35)": "#0ea5e9",
                    },
                    labels={"x": "Weighted Risk Contribution", "y": ""},
                    title=f"Risk Contribution Breakdown — {selected_pid}",
                    text_auto=".3f",
                )
                xai_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0f172a"),
                    showlegend=False,
                    xaxis=dict(range=[0, 1], showgrid=True, gridcolor="#e2e8f0"),
                    shapes=[
                        dict(
                            type="line", x0=0.50, x1=0.50, y0=-0.5, y1=1.5,
                            line=dict(color="#ef4444", width=2, dash="dot"),
                        )
                    ],
                    annotations=[
                        dict(x=0.50, y=1.6, text="Critical Threshold (0.50)",
                             showarrow=False, font=dict(color="#ef4444", size=11))
                    ],
                )
                st.plotly_chart(xai_fig, use_container_width=True, config={"displayModeBar": False})

                with st.expander("How this XAI reasoning is computed (simple view)"):
                    st.write(
                        "Total risk = (0.65 × ECG AI risk) + (0.35 × NEWS2 vitals risk). "
                        "A triage is marked Critical when the fused score is >= 0.50."
                    )

                # ── 5. Dynamic Clinical Reasoning Engine ─────────────────────
                triage_class = patient["Final_Triage_Class"]
                ecg_risk_raw = float(patient["AI_ECG_Risk"])
                news2_risk_raw = float(patient["NEWS2_Risk"])
                spo2_val = patient.get("spo2", "--")
                hr_val = patient.get("heart_rate", "--")

                if triage_class == "Critical" and ecg_risk_raw > 0.6:
                    reasoning = (
                        f"⚠️ **ECG-Driven Critical Alert:** The AI model heavily weighted the "
                        f"anomalous ECG waveform (ECG Score: {ecg_risk_raw:.1%}), triggering a "
                        f"critical alert despite the secondary vitals. "
                        f"Immediate cardiology review of the ECG trace is recommended."
                    )
                elif triage_class == "Critical" and news2_risk_raw > ecg_risk_raw:
                    reasoning = (
                        f"⚠️ **Vitals-Driven Critical Alert:** While the ECG did not show severe "
                        f"morphological anomalies (ECG Score: {ecg_risk_raw:.1%}), the patient's "
                        f"deteriorating vitals — specifically SpO₂ of **{spo2_val}%** and HR of "
                        f"**{hr_val} BPM** — drove the risk score above the critical threshold. "
                        f"Clinical monitoring and a nursing assessment are strongly indicated."
                    )
                elif triage_class == "Critical":
                    reasoning = (
                        f"⚠️ **Fused Critical Alert:** Both the ECG AI engine (Score: {ecg_risk_raw:.1%}) "
                        f"and the NEWS2 vitals score (Score: {news2_risk_raw:.1%}) contributed to breach "
                        f"the critical threshold. This patient requires urgent multi-disciplinary review."
                    )
                else:
                    reasoning = (
                        f"✅ **Low Risk Profile:** Both the neural network analysis of the ECG "
                        f"(Score: {ecg_risk_raw:.1%}) and the clinical NEWS2 vitals assessment "
                        f"(Score: {news2_risk_raw:.1%}) indicate a stable, low-risk profile. "
                        f"Routine monitoring is sufficient at this time."
                    )

                st.info(reasoning)
