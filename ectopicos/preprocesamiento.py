"""Preprocesamiento del ECG."""
from scipy.signal import butter, filtfilt, iirnotch


def preprocesar_ecg(señal, fs, f_low=0.5, f_high=100.0, f_notch=60.0, Q=30):
    """
    Tres filtros en cascada, aplicados con filtfilt (sin desfasaje):
      1) pasa-altos Butterworth orden 2 en 0,5 Hz  → deriva de la línea de base
      2) pasa-bajos Butterworth orden 4 en 100 Hz  → ruido de alta frecuencia
      3) notch IIR en la frecuencia de red (Q = 30) → interferencia eléctrica
    """
    b_hp, a_hp = butter(2, f_low / (fs / 2), btype='high')
    sig = filtfilt(b_hp, a_hp, señal)

    f_high_adj = min(f_high, fs / 2 - 1)
    b_lp, a_lp = butter(4, f_high_adj / (fs / 2), btype='low')
    sig = filtfilt(b_lp, a_lp, sig)

    if f_notch < fs / 2:
        b_n, a_n = iirnotch(f_notch / (fs / 2), Q)
        sig = filtfilt(b_n, a_n, sig)
    return sig
