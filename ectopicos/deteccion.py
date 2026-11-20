"""Detección de complejos QRS: Pan-Tompkins por canal, fusión AND/OR y validación."""
import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .config import REFRACTARIO, SIMBOLOS_LATIDO
from .preprocesamiento import preprocesar_ecg


def pan_tompkins_single(señal, fs):
    """Pan-Tompkins sobre un canal. Devuelve los picos y la señal integrada."""
    # 1. Pasa-banda 5–15 Hz
    b_hp, a_hp = butter(2, 5.0 / (fs / 2), btype='high')
    b_lp, a_lp = butter(2, 15.0 / (fs / 2), btype='low')
    sig = filtfilt(b_lp, a_lp, filtfilt(b_hp, a_hp, señal))
    # 2. Derivada de 5 puntos
    deriv = np.convolve(sig, np.array([-1, -2, 0, 2, 1]) * (fs / 8), mode='same')
    # 3. Cuadrado
    squared = deriv ** 2
    # 4. Integración con ventana móvil de 150 ms
    n_win = int(0.150 * fs)
    integ = np.convolve(squared, np.ones(n_win) / n_win, mode='same')
    # 5. Umbral adaptativo (picos separados al menos 200 ms)
    init_win = min(int(2 * fs), len(integ))
    SPK, NPK = 0.5 * np.max(integ[:init_win]), 0.0
    umbral = SPK
    candidatos, _ = find_peaks(integ, distance=int(0.200 * fs),
                               height=0.15 * np.max(integ[:init_win]))
    picos = []
    for p in candidatos:
        if integ[p] >= umbral:
            picos.append(p)
            SPK = 0.125 * integ[p] + 0.875 * SPK
        else:
            NPK = 0.125 * integ[p] + 0.875 * NPK
        umbral = NPK + 0.25 * (SPK - NPK)
    return np.array(picos, dtype=int), integ


def ajuste_fino_picos(picos_raw, señal_filtrada, fs, half_win_ms=50):
    """Reubica cada pico en el máximo absoluto de la señal filtrada en ±50 ms."""
    hw = int(half_win_ms * fs / 1000)
    ajustados = []
    for p in picos_raw:
        ini, fin = max(0, p - hw), min(len(señal_filtrada), p + hw)
        ajustados.append(int(ini + np.argmax(np.abs(señal_filtrada[ini:fin]))))
    return np.array(ajustados, dtype=int)


def refractario_dinamico(picos, fs, fraccion=REFRACTARIO['fraccion'],
                         min_ms=REFRACTARIO['min_ms'], max_ms=REFRACTARIO['max_ms']):
    """
    Período refractario dinámico: después de cada latido aceptado se descarta
    cualquier pico que llegue antes de fraccion × RR estimado (entre min_ms y max_ms).
    """
    if len(picos) == 0:
        return picos
    min_s, max_s = int(min_ms * fs / 1000), int(max_ms * fs / 1000)
    filtrados = [picos[0]]
    rr_est = int(0.800 * fs)                       # arranca suponiendo 75 lpm
    for p in picos[1:]:
        refrac = int(np.clip(fraccion * rr_est, min_s, max_s))
        if p - filtrados[-1] >= refrac:
            rr_est = int(0.875 * rr_est + 0.125 * (p - filtrados[-1]))
            filtrados.append(p)
    return np.array(filtrados, dtype=int)


def calidad_local(integ, pico, fs, ventana_ms=300):
    """Cociente entre el pico y la mediana de la señal integrada en ±300 ms."""
    hw = int(ventana_ms * fs / 1000)
    entorno = np.median(integ[max(0, pico - hw):min(len(integ), pico + hw)])
    if entorno < 1e-12:
        return 10.0
    return float(integ[pico] / (entorno + 1e-12))


def fusionar_and_or(picos_c0, picos_c1, integ_c0, integ_c1, fs,
                    ventana_ms=80, umbral_calidad=2.5, refractario=REFRACTARIO):
    """
    Combina las detecciones de los dos canales:
      · AND: los dos canales detectan un pico a menos de 80 ms → se acepta (posición de MLII)
      · OR : lo detecta un solo canal → se acepta solo si su calidad local es ≥ 2,5
    Al final se ordena y se aplica el período refractario dinámico.
    """
    tol = int(ventana_ms * fs / 1000)
    usados1 = np.zeros(len(picos_c1), dtype=bool)
    fusionados = []
    for p0 in picos_c0:
        if len(picos_c1):
            dists = np.abs(picos_c1 - p0)
            j = np.argmin(dists)
            if dists[j] <= tol and not usados1[j]:
                fusionados.append(int(p0))            # AND
                usados1[j] = True
                continue
        if calidad_local(integ_c0, p0, fs) >= umbral_calidad:
            fusionados.append(int(p0))                # OR, solo MLII
    for j, p1 in enumerate(picos_c1):
        if not usados1[j] and calidad_local(integ_c1, p1, fs) >= umbral_calidad:
            fusionados.append(int(p1))                # OR, solo segundo canal
    if not fusionados:
        return np.array([], dtype=int)
    return refractario_dinamico(np.sort(np.array(fusionados, dtype=int)), fs, **refractario)


def cargar_registro(ruta_datos, reg):
    """Lee un registro MIT-BIH: señales MLII y segundo canal, fs y anotaciones de latido."""
    import wfdb
    path = os.path.join(ruta_datos, reg)
    rec, ann = wfdb.rdrecord(path), wfdb.rdann(path, 'atr')
    i0 = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0  # MLII por nombre (114 está invertido)
    sym = np.array(ann.symbol)
    mask = np.isin(sym, list(SIMBOLOS_LATIDO))
    return rec.p_signal[:, i0], rec.p_signal[:, 1 - i0], rec.fs, ann.sample[mask], sym[mask]


def detectar_qrs(ruta_datos, reg):
    """Pipeline de detección de un registro. Devuelve picos, MLII filtrada, fs y anotaciones."""
    s0, s1, fs, ref_pos, ref_sym = cargar_registro(ruta_datos, reg)
    f_notch = 60.0 if fs >= 250 else 50.0
    c0 = preprocesar_ecg(s0, fs, f_notch=f_notch)
    c1 = preprocesar_ecg(s1, fs, f_notch=f_notch)
    raw0, int0 = pan_tompkins_single(c0, fs)
    raw1, int1 = pan_tompkins_single(c1, fs)
    picos = fusionar_and_or(ajuste_fino_picos(raw0, c0, fs), ajuste_fino_picos(raw1, c1, fs),
                            int0, int1, fs)
    return picos, c0, fs, ref_pos, ref_sym


def match_detecciones_referencia(det_peaks, ref_peaks, fs, tolerancia_ms=150):
    """
    Emparejamiento uno a uno entre detecciones y anotaciones (±150 ms), del menor
    al mayor error. Devuelve la referencia de cada detección (-1 = FP) y qué
    referencias fueron detectadas (False = FN).
    """
    det_peaks = np.sort(np.asarray(det_peaks, dtype=int))
    tol = int(tolerancia_ms * fs / 1000)
    pares = []
    for i, ref in enumerate(ref_peaks):
        j0 = np.searchsorted(det_peaks, ref - tol, side='left')
        j1 = np.searchsorted(det_peaks, ref + tol, side='right')
        pares += [(abs(det_peaks[j] - ref), i, j) for j in range(j0, j1)]
    pares.sort()
    ref_de_det = -np.ones(len(det_peaks), dtype=int)
    ref_usada = np.zeros(len(ref_peaks), dtype=bool)
    for _, i, j in pares:
        if not ref_usada[i] and ref_de_det[j] == -1:
            ref_usada[i], ref_de_det[j] = True, i
    return ref_de_det, ref_usada


def metricas_deteccion(det_peaks, ref_peaks, fs, tolerancia_ms=150):
    """TP, FP, FN, Se, Pp, F1 y error de posición de la detección de QRS."""
    det_peaks = np.sort(np.asarray(det_peaks, dtype=int))
    ref_de_det, ref_usada = match_detecciones_referencia(det_peaks, ref_peaks, fs, tolerancia_ms)
    TP = int(ref_usada.sum())
    FP, FN = len(det_peaks) - TP, len(ref_peaks) - TP
    Se, Pp = TP / max(TP + FN, 1), TP / max(TP + FP, 1)
    m = ref_de_det >= 0
    err = np.abs(det_peaks[m] - ref_peaks[ref_de_det[m]]) / fs * 1000
    return {'TP': TP, 'FP': FP, 'FN': FN, 'Se': Se, 'Pp': Pp,
            'F1': 2 * Se * Pp / max(Se + Pp, 1e-12),
            'Error mediano (ms)': float(np.median(err)) if len(err) else np.nan}, ref_usada
