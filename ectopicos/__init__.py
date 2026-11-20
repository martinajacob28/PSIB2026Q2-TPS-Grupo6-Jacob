"""
Detección y clasificación de latidos ectópicos (Normal / PAC / PVC) en la base MIT-BIH.

Módulos:
    config            parámetros y división DS1/DS2
    preprocesamiento  filtros del ECG
    deteccion         Pan-Tompkins, fusión AND/OR, período refractario y validación
    latidos           segmentación, plantilla por morfología dominante y 13 características
    estadistica       Kruskal-Wallis, Mann-Whitney + Bonferroni y δ de Cliff
    clasificacion     clasificador por reglas (Youden) y dos Random Forest binarios
"""
