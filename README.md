# AI-Powered Health Monitoring & Cardiac Triage Engine

## Overview
This project is an advanced, AI-driven Health Monitoring System designed to act as a supportive tool for healthcare professionals. It accelerates the triage process, providing doctors with rapid AI insights to cross-reference with junior doctor reports, ensuring accurate and efficient confirmation of patient health status.

## Key Features

### 1. ECG Pattern Analysis
- Upload and analyze ECG files.
- Automatically detects abnormal heart rate patterns within the ECG signals.
- Classifies patient cardiac risk based on these intricate ECG patterns.

### 2. Clinical Risk Scoring (NEWS)
- Evaluates essential vital signs: **SpO2, Temperature, and Heart Rate**.
- Automatically calculates and classifies patient conditions based on the standardized **NEWS (National Early Warning Score)** system to determine clinical urgency.

### 3. Explainable AI (XAI) Integration
- **Transparent AI Decisions:** Provides manual interpretability to understand the AI's reasoning, particularly ensuring transparency if the AI encounters ambiguous data or "fails" to make a high-confidence prediction.
- **Signal Fluctuation Tracking:** Highlights localized channel fluctuations and anomalies in the heart rate.
- **Batch Processing & Feature Contributions:** When multiple patient records are uploaded via a CSV file, the XAI module detects overarching patterns and explicitly details the individual decision contribution of each physiological parameter to the final model prediction.

## Model Parameters & Performance
- **Architecture**: `DenseNet1D` optimized for 12-lead ECG physiological sequence data.
- **Training Parameters**:
  - **Optimizer**: AdamW (Learning Rate: 0.001, Weight Decay: 1e-4)
  - **Loss Function**: Binary Cross-Entropy (BCEWithLogitsLoss)
  - **Learning Rate Scheduler**: StepLR (Step size: 5, Gamma: 0.5)
  - **Hyperparameters**: Batch Size: 32, Max Epochs: 40, Early Stopping Patience: 8
- **Evaluation Accuracy**: The model consistently achieves an **accuracy of 85%**, ensuring reliable and robust classification of cardiac risk.

## Objectives
- **Supportive Tool for Doctors:** Delivers faster AI-driven insights to assist in clinical decision-making.
- **Clinical Validation:** Provides a reliable, data-backed foundation for senior doctors to check against and validate clinical reports prepared by junior medical staff.

## Installation & Setup
To install and run this application locally on your machine, follow these steps:

1. **Clone the repository** (or download the source code):
   ```bash
   git clone <your-repository-url>
   cd Healthmonitoring_system# AI-Powered Health Monitoring & Cardiac Triage Engine


## Installation & Setup
To install and run this application locally on your machine, follow these steps:

1. **Clone the repository** (or download the source code):
   ```bash
   git clone <your-repository-url>
   cd Healthmonitoring_system
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv myenv
   # Activate on Windows:
   myenv\Scripts\activate
   # Activate on macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install Dependencies**:
   Install all the required Python libraries using `pip`.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For Streamlit Cloud deployments, system dependencies are configured in `packages.txt`.)*

4. **Run the Application**:
   Launch the user interface using Streamlit:
   ```bash
   streamlit run app.py
   ```

   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv myenv
   # Activate on Windows:
   myenv\Scripts\activate
   # Activate on macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install Dependencies**:
   Install all the required Python libraries using `pip`.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For Streamlit Cloud deployments, system dependencies are configured in `packages.txt`.)*

4. **Run the Application**:
   Launch the user interface using Streamlit:
   ```bash
   streamlit run app.py
   ```
