import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def apply_kalman_filter(signal_1d):
    """Simple 1D Kalman Filter for signal smoothing."""
    n = len(signal_1d)
    if n == 0:
        return signal_1d
    xhat = np.zeros(n)      # a posteri estimate of x
    P = np.zeros(n)         # a posteri error estimate
    xhatminus = np.zeros(n) # a priori estimate of x
    Pminus = np.zeros(n)    # a priori error estimate
    K = np.zeros(n)         # gain or blending factor
    
    Q = 1e-5 # process variance
    R = 0.01 # estimate of measurement variance, change to see effect
    
    xhat[0] = signal_1d[0]
    P[0] = 1.0
    
    for k in range(1, n):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k]*(signal_1d[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

def apply_wavelet_denoise(signal_1d, wavelet='db6', level=5):
    """Denoise signal using discrete wavelet transform."""
    if len(signal_1d) == 0:
        return signal_1d
    # Ensure signal length is long enough for the requested level
    max_level = pywt.dwt_max_level(data_len=len(signal_1d), filter_len=pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    
    if level == 0:
        return signal_1d
        
    coeffs = pywt.wavedec(signal_1d, wavelet, mode='per', level=level)
    
    # Calculate threshold based on highest frequency detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal_1d)))
    
    # Apply soft thresholding
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])
    
    # Reconstruct signal
    return pywt.waverec(coeffs, wavelet, mode='per')[:len(signal_1d)]

def apply_bandpass_filter(signal_1d, fs, lowcut=0.8, highcut=3.5, order=4):
    """Bandpass filter using Butterworth design."""
    if len(signal_1d) <= order * 3: # To prevent ValueError in filtfilt
        return signal_1d
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal_1d)
    return y

def scale_data(X_train, X_test=None):
    """Scale data using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler

def apply_smote(X, y):
    """Apply SMOTE for imbalanced classes."""
    # Ensure more than 1 class sample is present
    if len(np.unique(y)) < 2:
        return X, y
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
