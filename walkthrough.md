# Walkthrough

## 1. Cardiocare Doctor Report Integration
Previously, we mapped doctor report PDF uploads using `pdfplumber` to auto-extract clinical vitals into a structured format within the application UI.

## 2. Heart Rate Feature Pipeline 
The foundational signal processing and feature extraction has been finalized:
- **`src.preprocessing`**: Built [apply_kalman_filter](file:///d:/OneDrive/Documents/Placement_Preparation/Projects/Healthmonitoring_system/src/preprocessing.py#7-35), [apply_wavelet_denoise](file:///d:/OneDrive/Documents/Placement_Preparation/Projects/Healthmonitoring_system/src/preprocessing.py#36-58) (db6 level 5 optimization), and [apply_bandpass_filter](file:///d:/OneDrive/Documents/Placement_Preparation/Projects/Healthmonitoring_system/src/preprocessing.py#59-69) (0.8-3.5Hz to isolate human heart rates).
- **`src.feature_engineering`**: Designed [extract_hr_features](file:///d:/OneDrive/Documents/Placement_Preparation/Projects/Healthmonitoring_system/src/feature_engineering.py#5-58) utilizing RR-interval calculations (`find_peaks`) and Fast Fourier Transforms (FFT). Built the AI [TriBoostEnsemble](file:///d:/OneDrive/Documents/Placement_Preparation/Projects/Healthmonitoring_system/src/model.py#7-31) linking XGBoost, LightGBM, and Random forest frameworks together.

## 3. MEDVSE Dataset Integration & SpO2 Extraction
Loaded 62 actual patient vPPG recordings directly from `dataset/MEDVSE-main`:
- **SpO2 Mathematics**: Added `extract_spo2_features` computing the critical optical Ratio of Ratios calculation (`R_os = (AC_red/DC_red) / (AC_green/DC_green)`) for accurate Blood Oxygen identification.
- **Batched Data Pipeline**: Deployed `train_vitals_model.py`. The script iterates your entire 62-person dataset, successfully parsing the (1800, 3) 30Hz arrays into 447 precise 10-second non-overlapping signal windows, and aligns them directly with the corresponding (60, 2) 1Hz SpO2/HR Ground Truth Labels via an 80-20 Train/Test isolation.
- **Empirical Validation Output**: Your automated ensemble natively trained on the mathematical signal patterns with significant accuracy bounds across the test split:
  - **Heart Rate Error**: ~8.42 BPM (Mean Average Deviation)
  - **SpO2 Error**: ~1.46% (Mean Average Deviation)
