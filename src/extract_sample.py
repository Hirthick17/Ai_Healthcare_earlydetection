# python src/extract_sample.py
import pandas as pd
import numpy as np

meta    = pd.read_csv('dataset/archive/test_meta.csv')
# Read enough rows to cover the first few patients
signals = pd.read_csv('dataset/archive/test_signal.csv', nrows=20000)

def extract_patient(row_idx, output_name):
    patient_id = meta['ecg_id'].iloc[row_idx]
    norm = meta['NORM'].iloc[row_idx]
    mi   = meta['MI'].iloc[row_idx]
    cd   = meta['CD'].iloc[row_idx]
    sttc = meta['STTC'].iloc[row_idx]

    print(f"\n--- Patient ECG ID: {patient_id} ---")
    print(f"  Ground Truth  ->  NORM:{norm}  MI:{mi}  CD:{cd}  STTC:{sttc}")
    verdict = "HIGH RISK" if (mi or cd or sttc) else "LOW RISK / NORMAL"
    print(f"  Expected Pred ->  {verdict}")

    patient_rows = signals[signals['ecg_id'] == patient_id]
    ch_cols      = [c for c in patient_rows.columns if c.startswith('channel-')]
    arr          = patient_rows[ch_cols].values.astype(np.float32)
    np.save(output_name, arr)
    print(f"  Saved         ->  {output_name}  shape={arr.shape}")

# ── LOW RISK sample (NORM=1) ──────────────────────────────────────────────────
norm_idx = meta[meta['NORM'] == 1].index[0]
extract_patient(norm_idx, 'sample_ecg_normal.npy')

# ── HIGH RISK sample (MI=1 or CD=1) ──────────────────────────────────────────
risk_idx = meta[(meta['MI'] == 1) | (meta['CD'] == 1) | (meta['STTC'] == 1)].index[0]
extract_patient(risk_idx, 'sample_ecg_highrisk.npy')

print("\nDone! Upload these files in the Streamlit app to test both outcomes:")
print("  Low  Risk ->  sample_ecg_normal.npy")
print("  High Risk ->  sample_ecg_highrisk.npy")
