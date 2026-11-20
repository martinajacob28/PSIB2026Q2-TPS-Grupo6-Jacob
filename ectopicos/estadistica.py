"""Análisis estadístico de las características (tests no paramétricos y tamaño del efecto)."""
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from .config import CLASES, NOMBRES_CLASE

PARES = [('N', 'A'), ('N', 'V'), ('A', 'V')]


def cliff_delta(x, y):
    """δ de Cliff a partir del estadístico U de Mann-Whitney: δ = 2U/(n1·n2) − 1."""
    U = mannwhitneyu(x, y, alternative='two-sided').statistic
    return 2 * U / (len(x) * len(y)) - 1


def magnitud(d):
    """Umbrales de Romano et al. (2006) para |δ|."""
    d = abs(d)
    if d < 0.147:
        return 'despreciable'
    if d < 0.33:
        return 'pequeño'
    if d < 0.474:
        return 'mediano'
    return 'grande'


def tabla_estadistica(df, features):
    """
    Para cada característica: Kruskal-Wallis entre N, PAC y PVC (con ε² = H/(N−1))
    y, para cada par de clases, δ de Cliff y p de Mann-Whitney corregido por Bonferroni.
    """
    filas = []
    for f in features:
        grupos = {c: df.loc[df['clase'] == c, f].values for c in CLASES}
        if any(len(g) < 2 for g in grupos.values()):
            continue
        H, p = kruskal(*grupos.values())
        n = sum(len(g) for g in grupos.values())
        fila = {'Característica': f, 'H': H, 'p (KW)': p, 'ε²': H / (n - 1)}
        for a, b in PARES:
            p_ab = mannwhitneyu(grupos[a], grupos[b], alternative='two-sided').pvalue
            fila[f'δ {a}-{b}'] = cliff_delta(grupos[a], grupos[b])
            fila[f'p {a}-{b}'] = min(p_ab * len(PARES), 1.0)
        filas.append(fila)
    return pd.DataFrame(filas).set_index('Característica').sort_values('ε²', ascending=False)


def resumen_legible(est):
    """ε² y magnitud del efecto de cada par (* = no significativo tras Bonferroni)."""
    res = pd.DataFrame(index=est.index)
    res['ε²'] = est['ε²'].round(3)
    for a, b in PARES:
        res[f'{NOMBRES_CLASE[a]} vs {NOMBRES_CLASE[b]}'] = [
            f"{d:+.2f} ({magnitud(d)}){'' if p < 0.05 else ' *'}"
            for d, p in zip(est[f'δ {a}-{b}'], est[f'p {a}-{b}'])]
    return res
