"""Segmentación de latidos, plantilla normal por morfología dominante y 13 características."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from .config import PRE_MS, POST_MS


def ms2n(ms, fs):
    """Convierte milisegundos a cantidad de muestras."""
    return int(round(ms * fs / 1000))


def segmentar(señal, picos_r, fs, pre_ms=PRE_MS, post_ms=POST_MS):
    """Recorta cada latido de −250 a +400 ms del pico R (NaN si no entra completo)."""
    pre, post = ms2n(pre_ms, fs), ms2n(post_ms, fs)
    latidos = np.full((len(picos_r), pre + post), np.nan)
    for k, p in enumerate(picos_r):
        if p - pre >= 0 and p + post <= len(señal):
            latidos[k] = señal[p - pre:p + post]
    return latidos


def corr(a, b):
    """Correlación de Pearson entre dos tramos de señal."""
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def morfologia_dominante(latidos, fs, corr_min=0.9, n_max=600, semilla=0):
    """
    Índices de los latidos cuyo QRS tiene la forma que más se repite en el registro.
    No usa etiquetas ni intervalos RR:
      1) muestra al azar de hasta 600 latidos;
      2) agrupamiento jerárquico del QRS (±60 ms) con distancia 1 − correlación;
      3) el grupo más numeroso define el QRS de referencia (mediana);
      4) se eligen todos los latidos con correlación ≥ 0,9 con esa referencia.
    """
    pre = ms2n(PRE_MS, fs)
    q0, q1 = pre - ms2n(60, fs), pre + ms2n(60, fs)
    validos = np.where(~np.isnan(latidos[:, 0]))[0]
    validos = validos[np.std(latidos[validos, q0:q1], axis=1) > 1e-9]

    rng = np.random.default_rng(semilla)
    muestra = rng.choice(validos, size=min(n_max, len(validos)), replace=False)
    Z = linkage(latidos[muestra, q0:q1], method='average', metric='correlation')
    grupos = fcluster(Z, t=1 - corr_min, criterion='distance')
    dominante = np.bincount(grupos).argmax()
    qrs_ref = np.median(latidos[muestra[grupos == dominante], q0:q1], axis=0)

    r = np.array([corr(latidos[k, q0:q1], qrs_ref) for k in validos])
    return validos[r >= corr_min]


def plantilla_normal(latidos, fs, corr_min=0.9):
    """Plantilla del paciente: mediana de los latidos con la morfología dominante."""
    elegidos = morfologia_dominante(latidos, fs, corr_min=corr_min)
    return np.median(latidos[elegidos], axis=0), elegidos


def ancho_qrs(señal, p, fs, frac=0.15, max_ms=250):
    """Ancho del QRS: tiempo en que la señal supera el 15 % de la amplitud del complejo."""
    hw = ms2n(max_ms / 2, fs)
    v = señal[max(0, p - hw):p + hw]
    seg = v - np.median(v)
    amp = np.max(np.abs(seg))
    if amp < 1e-9:
        return np.nan
    c = p - max(0, p - hw)
    li, ld = c, c
    while li > 0 and abs(seg[li]) >= frac * amp:
        li -= 1
    while ld < len(seg) - 1 and abs(seg[ld]) >= frac * amp:
        ld += 1
    return (ld - li) / fs * 1000


def xcorr_max(a, b, max_lag):
    """Máxima correlación entre a y b desplazando hasta ±max_lag muestras."""
    mejor, n = -1.0, len(a)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        mejor = max(mejor, corr(x, y))
    return mejor


def extraer_caracteristicas(señal, picos_r, fs):
    """
    13 características de cada latido de un registro.
    Devuelve la tabla (un latido por fila), la plantilla y los índices usados para armarla.
    """
    latidos = segmentar(señal, picos_r, fs)
    plantilla, elegidos = plantilla_normal(latidos, fs)

    pre = ms2n(PRE_MS, fs)
    q0, q1 = pre - ms2n(60, fs), pre + ms2n(60, fs)     # QRS (±60 ms)
    p0, p1 = pre - ms2n(250, fs), pre - ms2n(80, fs)    # onda P
    lag = ms2n(30, fs)                                  # desplazamiento máximo ±30 ms

    amp_pl = np.max(np.abs(plantilla[q0:q1] - np.median(plantilla)))
    qrs_pl = plantilla[q0:q1]
    rr = np.diff(picos_r) / fs * 1000

    filas = []
    for k in range(1, len(picos_r) - 1):
        lat = latidos[k]
        if np.isnan(lat[0]) or np.isnan(latidos[k - 1, 0]):
            continue
        rr_prev, rr_post = rr[k - 1], rr[k]
        rr_local = np.median(rr[max(0, k - 8):k + 8])
        qrs = lat[q0:q1]
        base = np.median(lat)
        filas.append({
            'idx': k, 'muestra': picos_r[k],
            # Temporales
            'rr_prev': rr_prev,
            'rr_post': rr_post,
            'prematuridad': rr_prev / rr_local,
            'pausa_comp': (rr_prev + rr_post) / (2 * rr_local),
            # Morfológicas
            'ancho_qrs': ancho_qrs(señal, picos_r[k], fs),
            'area_qrs': np.trapezoid(np.abs(qrs - base), dx=1000 / fs),
            'amp_rel': np.max(np.abs(qrs - base)) / amp_pl if amp_pl > 1e-9 else np.nan,
            # Parecido con la plantilla normal
            'corr_qrs': corr(qrs, qrs_pl),
            'xcorr_qrs': xcorr_max(lat[q0 - lag:q1 + lag], plantilla[q0 - lag:q1 + lag], lag),
            'corr_latido': corr(lat, plantilla),
            'corr_p': corr(lat[p0:p1], plantilla[p0:p1]),
            'corr_prev': corr(qrs, latidos[k - 1][q0:q1]),
            'rmse_norm': np.sqrt(np.mean((lat - plantilla) ** 2)) / amp_pl if amp_pl > 1e-9 else np.nan,
        })
    return pd.DataFrame(filas), plantilla, elegidos
