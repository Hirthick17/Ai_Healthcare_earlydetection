"""
src/clinical_rules.py
---------------------
Deterministic, Clinically-Validated Rule-Based Expert System
Implements the NEWS2 Protocol for secondary vitals assessment.
"""

def calculate_news2_score(hr: int, spo2: int, temp: float):
    # SpO2 Scoring
    if spo2 >= 96:
        spo2_pts = 0
    elif 94 <= spo2 <= 95:
        spo2_pts = 1
    elif 92 <= spo2 <= 93:
        spo2_pts = 2
    else:  # <= 91
        spo2_pts = 3
        
    # Heart Rate Scoring
    if 51 <= hr <= 90:
        hr_pts = 0
    elif (41 <= hr <= 50) or (91 <= hr <= 110):
        hr_pts = 1
    elif 111 <= hr <= 130:
        hr_pts = 2
    else:  # <=40 or >=131
        hr_pts = 3
        
    # Temperature Scoring
    if 36.1 <= temp <= 38.0:
        temp_pts = 0
    elif (35.1 <= temp <= 36.0) or (38.1 <= temp <= 39.0):
        temp_pts = 1
    elif temp >= 39.1:
        temp_pts = 2
    else:  # <= 35.0
        temp_pts = 3
        
    total_score = hr_pts + spo2_pts + temp_pts
    
    breakdown = {
        "Heart Rate": hr_pts,
        "SpO2": spo2_pts,
        "Temperature": temp_pts
    }
    
    return total_score, breakdown
