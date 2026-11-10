# PSIB20262Q-TPS-Jacob

Pipeline automatizado para la detección y clasificación de latidos ectópicos mediante características temporales y morfológicas

16.63 Procesamiento de Señales e Imágenes Biomédicas (ITBA, 2026 Q2)

Alumna: Martina Jacob

## Descripción

El trabajo detecta los latidos del ECG de la MIT-BIH Arrhythmia Database y los clasifica en tres clases: latidos normales (N), contracciones auriculares prematuras (PAC) y contracciones ventriculares prematuras (PVC).

El trabajo se divide en las siguientes etapas:

1. **Preprocesamiento:** filtro pasa-altos (0,5 Hz), pasa-bajos (100 Hz) y notch (60 Hz), verificado con la PSD de Welch.
2. **Detección de QRS:** algoritmo de Pan-Tompkins sobre los dos canales, con fusión AND/OR y período refractario ajustado para no eliminar los latidos ectópicos. Validación contra las anotaciones de la base.
3. **Latido normal de referencia:** plantilla del latido normal de cada paciente, construida sin usar las etiquetas.
4. **Extracción de características:** 13 características temporales, morfológicas y de correlación con la plantilla.
5. **Análisis estadístico:** Kruskal-Wallis, Mann-Whitney con corrección de Bonferroni, δ de Cliff y correlación de Spearman.
6. **Clasificación:** clasificador por reglas (umbrales elegidos con el índice de Youden) y Random Forest, con división inter-paciente DS1/DS2 (de Chazal et al., 2004).

## Contenido del repositorio

| Archivo | Contenido |
|---|---|
| `PSIB2026Q2_TPS_Grupo6_Jacob.ipynb` | Notebook con todo el código y el texto explicativo de cada etapa |
| `PSIB2026Q2-TPS-Grupo6-Jacob.pdf` | Informe del trabajo |
| `requirements.txt` | Librerías necesarias |
| `README.md` | Este archivo |

## Base de datos

Se usa la **MIT-BIH Arrhythmia Database** de PhysioNet: 48 registros de unos 30 minutos, con dos canales a 360 Hz.
https://physionet.org/content/mitdb/1.0.0/

No hace falta descargarla a mano: el notebook la descarga sola con `wfdb.dl_database('mitdb', ...)` la primera vez que se ejecuta.

## Instalación

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

En Google Colab, la primera celda del notebook ya las instala.

## Cómo ejecutarlo

### Opción 1: Google Colab (recomendada)

1. Abrir `PSIB2026Q2_TPS_Grupo6_Jacob.ipynb` en Google Colab.
2. Ejecutar las celdas en orden (*Entorno de ejecución → Ejecutar todas*).
3. Cuando lo pida, autorizar el acceso a Google Drive. La base se guarda en la carpeta definida en `DATASET_PATH`, que se puede cambiar por cualquier carpeta del Drive.

### Opción 2: de forma local (Jupyter)

1. Instalar las dependencias con `pip install -r requirements.txt`.
2. En la celda de carga del dataset, borrar las líneas de `google.colab` y `drive.mount(...)`.
3. Cambiar `DATASET_PATH` por una carpeta local, por ejemplo `DATASET_PATH = './mitdb'`.
4. Ejecutar todas las celdas en orden.

La primera ejecución tarda más porque descarga la base y procesa los 44 registros.

## Resultados principales

- **Detección de QRS (44 registros):** Se = 99,2 %, Pp = 99,7 %.
- **Clasificación sobre DS2 (pacientes no usados en el ajuste):**

| Clase | F1 reglas | F1 Random Forest |
|---|---|---|
| Normal | 0,913 | 0,965 |
| PAC | 0,142 | 0,091 |
| PVC | 0,774 | 0,968 |

Los PVC se clasifican muy bien. Los PAC son la clase más difícil, porque su QRS es igual al normal y la prematuridad falla cuando el ritmo es irregular o hay muchos PAC.

## Referencias principales

- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Trans. Biomed. Eng.*, 32(3), 230–236.
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng. Med. Biol. Mag.*, 20(3), 45–50.
- de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7), 1196–1206.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
