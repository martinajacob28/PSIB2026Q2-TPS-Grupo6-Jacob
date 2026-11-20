"""Parámetros generales del trabajo."""

# Registros con marcapasos: se excluyen
REG_MARCAPASOS = ['102', '104', '107', '217']

# División inter-paciente de de Chazal et al. (2004)
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
       '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
       '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
REGISTROS = sorted(DS1 + DS2)

# Todos los símbolos de MIT-BIH que corresponden a un latido
SIMBOLOS_LATIDO = set('NLRBAaJSVrFejnE/fQ')
# Latidos de origen sinusal (solo se usan para verificar la plantilla)
SINUSALES = ['N', 'L', 'R', 'e', 'j']

# Período refractario dinámico ajustado: 40 % del RR, entre 200 y 300 ms
REFRACTARIO = {'fraccion': 0.40, 'min_ms': 200, 'max_ms': 300}

# Ventana de cada latido alrededor del pico R
PRE_MS, POST_MS = 250, 400

CLASES = ['N', 'A', 'V']
NOMBRES_CLASE = {'N': 'Normal', 'A': 'PAC', 'V': 'PVC'}

FEATURES = ['rr_prev', 'rr_post', 'prematuridad', 'pausa_comp',
            'ancho_qrs', 'area_qrs', 'amp_rel',
            'corr_qrs', 'xcorr_qrs', 'corr_latido', 'corr_p', 'corr_prev', 'rmse_norm']
