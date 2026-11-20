"""Clasificador por reglas (umbrales de Youden) y dos Random Forest binarios."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_fscore_support, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict

from .config import CLASES, NOMBRES_CLASE, FEATURES


# ── Métricas ──────────────────────────────────────────────────────────────────
def metricas_por_clase(y_true, y_pred):
    """Se, Pp y F1 de cada clase, y el promedio SOLO de PAC y PVC."""
    pp, se, f1, n = precision_recall_fscore_support(y_true, y_pred, labels=CLASES, zero_division=0)
    t = pd.DataFrame({'Se': se, 'Pp': pp, 'F1': f1, 'n latidos': n},
                     index=[NOMBRES_CLASE[c] for c in CLASES])
    t.loc['Promedio PAC y PVC'] = list(t.loc[['PAC', 'PVC'], ['Se', 'Pp', 'F1']].mean()) + [np.nan]
    return t


# ── Clasificador por reglas ───────────────────────────────────────────────────
def umbral_youden(valores, es_positivo, menor_es_positivo=True):
    """Umbral que maximiza J = Se + Esp − 1 sobre la curva ROC. Devuelve también el AUC."""
    puntaje = -valores if menor_es_positivo else valores
    fpr, tpr, thr = roc_curve(es_positivo, puntaje)
    k = np.argmax(tpr - fpr)
    umbral = -thr[k] if menor_es_positivo else thr[k]
    return umbral, roc_auc_score(es_positivo, puntaje), tpr[k], 1 - fpr[k]


def ajustar_reglas(df_ds1):
    """
    Umbrales elegidos con DS1:
      · PVC si corr_qrs ≤ umbral (PVC contra todo el resto)
      · PAC si prematuridad ≤ umbral (PAC contra Normal, sin las PVC)
    """
    thr_corr, auc_c, se_c, esp_c = umbral_youden(df_ds1['corr_qrs'].values,
                                                 (df_ds1['clase'] == 'V').values)
    no_v = df_ds1[df_ds1['clase'] != 'V']
    thr_prem, auc_p, se_p, esp_p = umbral_youden(no_v['prematuridad'].values,
                                                 (no_v['clase'] == 'A').values)
    return {'thr_corr': thr_corr, 'thr_prem': thr_prem,
            'roc': pd.DataFrame({'Umbral': [thr_corr, thr_prem], 'AUC': [auc_c, auc_p],
                                 'Se': [se_c, se_p], 'Esp': [esp_c, esp_p]},
                                index=['PVC: corr_qrs', 'PAC: prematuridad'])}


def clasificar_reglas(df, thr_corr, thr_prem):
    """Primero PVC (forma del QRS), después PAC (prematuridad); el resto es Normal."""
    pred = np.full(len(df), 'N', dtype=object)
    pred[df['prematuridad'].values <= thr_prem] = 'A'
    pred[df['corr_qrs'].values <= thr_corr] = 'V'
    return pred


# ── Random Forest: dos clasificadores binarios ────────────────────────────────
GRILLA = {
    'max_depth':        [None, 8],
    'min_samples_leaf': [5, 20],
    'max_features':     ['sqrt', 0.5],
    'class_weight':     [None, 'balanced_subsample'],
}


def entrenar_binario(df_ds1, positiva, grilla=GRILLA, n_estimators=100, n_pliegues=5):
    """
    Random Forest Normal vs `positiva` ('A' o 'V'), solo con DS1:
      1) búsqueda de hiperparámetros con GroupKFold por paciente (métrica: AP);
      2) umbral de decisión que maximiza el F1 de la clase ectópica, elegido con
         probabilidades fuera de pliegue.
    """
    d = df_ds1[df_ds1['clase'].isin(['N', positiva])]
    X, y, g = d[FEATURES].values, (d['clase'] == positiva).astype(int).values, d['registro'].values
    cv = GroupKFold(n_splits=n_pliegues)

    busqueda = GridSearchCV(RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
                            grilla, scoring='average_precision', cv=cv, n_jobs=1)
    busqueda.fit(X, y, groups=g)

    p_oof = cross_val_predict(busqueda.best_estimator_, X, y, groups=g, cv=cv,
                              method='predict_proba')[:, 1]
    prec, sens, umbrales = precision_recall_curve(y, p_oof)
    f1 = 2 * prec * sens / (prec + sens + 1e-12)
    k = np.argmax(f1[:-1])
    return {'modelo': busqueda.best_estimator_, 'umbral': float(umbrales[k]),
            'f1_cv': float(f1[k]), 'ap_cv': float(busqueda.best_score_),
            'mejores_parametros': busqueda.best_params_}


def clasificar_rf(modelos, df):
    """Primero se pregunta si es PVC; si no, si es PAC; si no, es Normal."""
    X = df[FEATURES].values
    es_v = modelos['V']['modelo'].predict_proba(X)[:, 1] >= modelos['V']['umbral']
    es_a = modelos['A']['modelo'].predict_proba(X)[:, 1] >= modelos['A']['umbral']
    return np.where(es_v, 'V', np.where(es_a, 'A', 'N'))


def evaluar_binario(modelos, df_ds2, positiva):
    """AP del Random Forest en DS2 para el problema Normal vs `positiva`."""
    d = df_ds2[df_ds2['clase'].isin(['N', positiva])]
    y = (d['clase'] == positiva).astype(int).values
    p = modelos[positiva]['modelo'].predict_proba(d[FEATURES].values)[:, 1]
    return average_precision_score(y, p), y.mean()
