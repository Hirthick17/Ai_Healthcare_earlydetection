"""
src/triage_synthesizer.py
-------------------------
Priority Scoring Engine for parsing and prioritizing batch dataset vitals.
"""

import pandas as pd

class PriorityScoringEngine:
    @staticmethod
    def calculate_risk(df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates over a DataFrame of patient vitals and computes a risk classification.
        Threshold: >= 0.50 is Critical.
        """
        scores = []
        classifications = []
        
        for idx, row in df.iterrows():
            score = 0.0
            
            hr = float(row.get('heart_rate', 70))
            spo2 = float(row.get('spo2', 99))
            temp = float(row.get('temperature', 37.0))
            ecg = str(row.get('ecg_status', 'NORM')).upper()
            
            # Simple clinically-inspired weighting
            if hr > 110 or hr < 50: score += 0.3
            if spo2 < 93: score += 0.4
            if spo2 < 90: score += 0.2
            if temp > 38.5 or temp < 35.0: score += 0.2
            if ecg in ['MI', 'STTC', 'CD']: score += 0.5
            
            final_score = min(score, 1.0)
            scores.append(final_score)
            classifications.append('Critical' if final_score >= 0.50 else 'Stable')
            
        df['Risk_Score'] = scores
        df['Risk_Classification'] = classifications
        return df
