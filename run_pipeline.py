"""
Corre todo el trabajo de principio a fin y guarda los resultados en tablas CSV.

Uso:
    python run_pipeline.py --datos RUTA_A_MITBIH
    python run_pipeline.py --datos RUTA_A_MITBIH --descargar      # baja la base si no está
    python run_pipeline.py --datos RUTA_A_MITBIH --reusar         # usa las características ya calculadas

Pasos:
    1) Detección de QRS en los 44 registros (Pan-Tompkins + fusión AND/OR) y validación.
    2) Plantilla por morfología dominante y 13 características de cada latido.
    3) Estadística en DS1: Kruskal-Wallis, Mann-Whitney + Bonferroni y δ de Cliff.
    4) Clasificador por reglas con umbrales de Youden (DS1) evaluado en DS2.
    5) Dos Random Forest binarios (búsqueda por paciente en DS1) evaluados en DS2.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ectopicos.config import DS1, REGISTROS, FEATURES, CLASES, SINUSALES
from ectopicos.deteccion import detectar_qrs, metricas_deteccion, match_detecciones_referencia
from ectopicos.latidos import extraer_caracteristicas
from ectopicos.estadistica import tabla_estadistica, resumen_legible
from ectopicos.clasificacion import (ajustar_reglas, clasificar_reglas, entrenar_binario,
                                     clasificar_rf, evaluar_binario, metricas_por_clase)


def paso(n, texto):
    print(f"\n{'=' * 70}\n{n}) {texto}\n{'=' * 70}")


def deteccion_y_caracteristicas(ruta_datos, registros):
    """Pasos 1 y 2: detección, validación y características de cada registro."""
    det, tablas, verif = [], [], []
    for k, reg in enumerate(registros, 1):
        picos, c0, fs, ref_pos, ref_sym = detectar_qrs(ruta_datos, reg)
        m, _ = metricas_deteccion(picos, ref_pos, fs)
        m.update(Registro=reg, Subconjunto='DS1' if reg in DS1 else 'DS2')
        det.append(m)

        df_f, _, elegidos = extraer_caracteristicas(c0, picos, fs)
        ref_de_det, _ = match_detecciones_referencia(picos, ref_pos, fs)
        idx = ref_de_det[df_f['idx'].values]
        df_f['clase'] = np.where(idx >= 0, ref_sym[np.maximum(idx, 0)], 'FP')
        df_f['registro'] = reg
        df_f['subconjunto'] = 'DS1' if reg in DS1 else 'DS2'
        tablas.append(df_f)

        clase_pl = np.where(ref_de_det[elegidos] >= 0, ref_sym[np.maximum(ref_de_det[elegidos], 0)], 'FP')
        verif.append({'registro': reg, 'latidos en la plantilla': len(elegidos),
                      '% sinusales': 100 * np.isin(clase_pl, SINUSALES).mean()})
        print(f"   ({k}/{len(registros)}) {reg}: Se={m['Se']:.4f}  Pp={m['Pp']:.4f}  "
              f"latidos={len(df_f)}  plantilla={verif[-1]['% sinusales']:.1f} % sinusales")

    df_det = pd.DataFrame(det).set_index('Registro')
    return df_det, pd.concat(tablas, ignore_index=True), pd.DataFrame(verif).set_index('registro')


def main():
    ap = argparse.ArgumentParser(description='Detección y clasificación de latidos ectópicos (MIT-BIH)')
    ap.add_argument('--datos', required=True, help='Carpeta con los archivos de MIT-BIH (.hea, .dat, .atr)')
    ap.add_argument('--salida', default='resultados', help='Carpeta donde se guardan las tablas')
    ap.add_argument('--registros', nargs='*', default=REGISTROS, help='Registros a usar (por defecto los 44)')
    ap.add_argument('--descargar', action='store_true', help='Descarga MIT-BIH en --datos si no está')
    ap.add_argument('--reusar', action='store_true', help='Reusa caracteristicas.csv si ya existe')
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    salida = lambda nombre: os.path.join(args.salida, nombre)
    t0 = time.time()

    if args.descargar:
        import wfdb
        os.makedirs(args.datos, exist_ok=True)
        if len([f for f in os.listdir(args.datos) if f.endswith('.hea')]) < 48:
            print("Descargando MIT-BIH ...")
            wfdb.dl_database('mitdb', args.datos)

    # 1 y 2) Detección y características
    if args.reusar and os.path.exists(salida('caracteristicas.csv')):
        paso(1, 'Detección y características: se reusan las ya calculadas')
        df_todo = pd.read_csv(salida('caracteristicas.csv'), dtype={'registro': str})
    else:
        paso(1, f'Detección de QRS y extracción de características ({len(args.registros)} registros)')
        df_det, df_todo, df_verif = deteccion_y_caracteristicas(args.datos, args.registros)
        TP, FP, FN = df_det[['TP', 'FP', 'FN']].sum()
        print(f"\n   Detección global: Se = {TP / (TP + FN):.4f}   Pp = {TP / (TP + FP):.4f}")
        print(f"   Plantilla: mediana de % sinusales = {df_verif['% sinusales'].median():.1f} %")
        df_det.to_csv(salida('deteccion_por_registro.csv'))
        df_verif.to_csv(salida('verificacion_plantilla.csv'))
        df_todo.to_csv(salida('caracteristicas.csv'), index=False)

    df_lat = df_todo[df_todo['clase'].isin(CLASES)].dropna(subset=FEATURES).reset_index(drop=True)
    df_ds1 = df_lat[df_lat['subconjunto'] == 'DS1']
    df_ds2 = df_lat[df_lat['subconjunto'] == 'DS2']
    print("\n   Latidos por clase y subconjunto:")
    print(pd.crosstab(df_lat['subconjunto'], df_lat['clase']).to_string())

    # 3) Estadística
    paso(3, 'Estadística de las características (DS1)')
    est = tabla_estadistica(df_ds1, FEATURES)
    est.to_csv(salida('estadistica_por_latido.csv'))
    df_pac = df_ds1.groupby(['registro', 'clase'])[FEATURES].median().reset_index()
    tabla_estadistica(df_pac, FEATURES).to_csv(salida('estadistica_por_paciente.csv'))
    print(resumen_legible(est).to_string())

    # 4) Reglas
    paso(4, 'Clasificador por reglas (umbrales de Youden en DS1, evaluación en DS2)')
    reglas = ajustar_reglas(df_ds1)
    print(reglas['roc'].round(6).to_string())
    res_reglas = metricas_por_clase(df_ds2['clase'].values,
                                    clasificar_reglas(df_ds2, reglas['thr_corr'], reglas['thr_prem']))
    print("\n" + res_reglas.round(3).to_string())

    # 5) Random Forest
    paso(5, 'Dos Random Forest binarios (búsqueda por paciente en DS1, evaluación en DS2)')
    modelos = {}
    for positiva in ['V', 'A']:
        modelos[positiva] = entrenar_binario(df_ds1, positiva)
        ap_ds2, azar = evaluar_binario(modelos, df_ds2, positiva)
        modelos[positiva]['ap_ds2'] = ap_ds2
        print(f"   Normal vs {'PVC' if positiva == 'V' else 'PAC'}: {modelos[positiva]['mejores_parametros']}")
        print(f"      umbral = {modelos[positiva]['umbral']:.3f}   AP validación (DS1) = "
              f"{modelos[positiva]['ap_cv']:.3f}   AP en DS2 = {ap_ds2:.3f} (azar = {azar:.3f})")
    res_rf = metricas_por_clase(df_ds2['clase'].values, clasificar_rf(modelos, df_ds2))
    print("\n" + res_rf.round(3).to_string())

    # Comparación final, solo clases ectópicas
    comp = pd.concat({'Reglas': res_reglas.loc[['PAC', 'PVC', 'Promedio PAC y PVC'], ['Se', 'Pp', 'F1']],
                      'Random Forest': res_rf.loc[['PAC', 'PVC', 'Promedio PAC y PVC'], ['Se', 'Pp', 'F1']]},
                     axis=1)
    comp.to_csv(salida('comparacion_ds2.csv'))
    paso('✓', 'Comparación final en DS2 (solo PAC y PVC)')
    print(comp.round(3).to_string())

    with open(salida('resumen.json'), 'w', encoding='utf-8') as fh:
        json.dump({'umbral_corr_qrs': reglas['thr_corr'], 'umbral_prematuridad': reglas['thr_prem'],
                   'rf': {c: {k: v for k, v in m.items() if k != 'modelo'} for c, m in modelos.items()}},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nTablas guardadas en '{args.salida}/'. Tiempo total: {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
