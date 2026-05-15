# AI-Powered Health Monitoring & Cardiac Triage Engine

## Description
This project is an advanced, AI-driven Health Monitoring System designed to act as a supportive tool for healthcare professionals. It accelerates the triage process, providing doctors with rapid AI insights to cross-reference with junior doctor reports, ensuring accurate and efficient confirmation of patient health status. The system integrates cutting-edge signal processing, machine learning, and explainable AI to deliver comprehensive cardiac analysis and clinical risk assessment.

## Technologies
- **Core Framework**: Streamlit 1.32.0
- **Deep Learning**: PyTorch 2.1.0 (CPU-optimized), DenseNet1D architecture
- **Signal Processing**: NumPy, SciPy, PyWavelets, OpenCV
- **OCR & PDF Processing**: Pytesseract, PyMuPDF
- **Data Processing**: Pandas, OpenPyXL
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
- **Explainable AI**: Captum
- **AI Integration**: Google Generative AI (Gemini)
- **Clinical Standards**: NEWS (National Early Warning Score) system

## Features
### 1. ECG Pattern Analysis
- Upload and analyze ECG files (12-lead ECG support)
- Automatically detects abnormal heart rate patterns within ECG signals
- Advanced signal preprocessing with Kalman filtering, wavelet denoising, and bandpass filtering
- Classifies patient cardiac risk based on intricate ECG patterns using DenseNet1D

### 2. Clinical Risk Scoring (NEWS)
- Evaluates essential vital signs: **SpO2, Temperature, and Heart Rate**
- Automatically calculates and classifies patient conditions based on the standardized **NEWS (National Early Warning Score)** system
- Determines clinical urgency and prioritizes patient triage

### 3. Explainable AI (XAI) Integration
- **Transparent AI Decisions**: Provides manual interpretability to understand AI reasoning
- **Signal Fluctuation Tracking**: Highlights localized channel fluctuations and heart rate anomalies
- **Batch Processing & Feature Contributions**: When multiple patient records are uploaded via CSV, the XAI module detects overarching patterns and details individual decision contributions

### 4. Advanced Signal Processing
- **Heart Rate Feature Pipeline**: RR-interval calculations, Fast Fourier Transforms (FFT)
- **SpO2 Extraction**: Optical Ratio of Ratios calculation for accurate Blood Oxygen identification
- **TriBoostEnsemble**: Combines XGBoost, LightGBM, and Random Forest frameworks

### 5. Mobile-First Interface
- Modern "Soft Neumorphic Slate & Indigo" aesthetic design
- Responsive layout optimized for various screen sizes
- Intuitive file upload and data visualization

## Process
1. **Data Input**: Upload medical documents (PDFs) or ECG files through the Streamlit interface
2. **Signal Processing**: 
   - Apply Kalman filtering for noise reduction
   - Use wavelet denoising (db6 level 5 optimization)
   - Apply bandpass filtering (0.8-3.5Hz) to isolate human heart rates
3. **Feature Extraction**:
   - Extract heart rate features using RR-interval calculations
   - Compute SpO2 using optical Ratio of Ratios
   - Generate FFT-based frequency domain features
4. **AI Analysis**:
   - Process through DenseNet1D model for ECG classification
   - Apply TriBoostEnsemble for comprehensive risk assessment
   - Calculate NEWS scores for clinical urgency
5. **Results Visualization**:
   - Display risk classifications and confidence scores
   - Show explainable AI insights and feature contributions
   - Generate detailed clinical reports

## What I Learned
- **Advanced Signal Processing**: Mastered ECG signal preprocessing techniques including Kalman filtering, wavelet denoising, and bandpass filtering
- **Deep Learning Architecture**: Implemented and optimized DenseNet1D for physiological sequence data
- **Explainable AI**: Developed transparent ML systems using Captum for interpretability
- **Clinical Integration**: Applied real-world medical standards (NEWS scoring) in AI systems
- **Full-Stack Development**: Built complete ML pipeline from data ingestion to deployment
- **Performance Optimization**: Achieved 85% accuracy with efficient CPU-optimized models
- **User Experience**: Created intuitive mobile-first interfaces for complex medical applications

## How to Run the Project
### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
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
   Install all the required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For Streamlit Cloud deployments, system dependencies are configured in `packages.txt`.)*

4. **Run the Application**:
   Launch the user interface using Streamlit:
   ```bash
   streamlit run app.py
   ```

## Project Demonstration Video

### 🎬 Watch the Demo Video
Watch the project demonstration video to see the system in action:

[![Project Demonstration](https://drive.google.com/file/d/11xBJJbPHMsdvrBTXXMgukA9e93LpqE0i/view?usp=sharing)](https://drive.google.com/file/d/11xBJJbPHMsdvrBTXXMgukA9e93LpqE0i/view?usp=sharing)



**Video Features:**
- File upload and processing workflows
- ECG analysis and cardiac risk classification
- Clinical scoring and report generation
- Interactive dashboard features and user interface

### 📁 Video File Location
The video file is also included directly in the repository:
```
Working_video/Project_Demonstration_Healthsystem.mp4
```

**Note:** For optimal GitHub repository performance, videos are best hosted externally (YouTube, Vimeo) and embedded as shown above. The local video file ensures you have a backup copy and can be downloaded directly if needed.

### 🔄 How to Add Video to GitHub
1. **Upload the video file** to your repository in the `Working_video/` folder
2. **For better performance**, consider uploading to YouTube/Vimeo and using the embed code above
3. **Large files**: Consider using Git LFS for video files over 100MB
4. **Alternative**: Use GitHub releases for video distribution

### 📋 Video Content Overview
The demonstration covers:
- **System Architecture**: Overview of the AI-powered health monitoring pipeline
- **User Interface**: Interactive dashboard with file upload capabilities
- **ECG Processing**: Real-time ECG signal analysis and visualization
- **Risk Assessment**: Cardiac risk classification and clinical scoring
- **Explainable AI**: Feature importance and model transparency
- **Clinical Integration**: NEWS scoring and report generation

## Model Performance
- **Architecture**: DenseNet1D optimized for 12-lead ECG physiological sequence data
- **Training Parameters**:
  - Optimizer: AdamW (Learning Rate: 0.001, Weight Decay: 1e-4)
  - Loss Function: Binary Cross-Entropy (BCEWithLogitsLoss)
  - Learning Rate Scheduler: StepLR (Step size: 5, Gamma: 0.5)
  - Hyperparameters: Batch Size: 32, Max Epochs: 40, Early Stopping Patience: 8
- **Evaluation Accuracy**: Consistently achieves **85%** accuracy for cardiac risk classification
- **Signal Processing Accuracy**: 
  - Heart Rate Error: ~8.42 BPM (Mean Average Deviation)
  - SpO2 Error: ~1.46% (Mean Average Deviation)
