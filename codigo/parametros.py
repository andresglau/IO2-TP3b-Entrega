"""Carga de datos y calibracion - Motor RedBaires (TP3 parte B, IO2).

Este modulo es la "fuente unica" de todo lo que el motor (motor.py) y el costeo
(costos.py) necesitan para correr: parametros economicos, el pool de bootstrap de
atributos de actor, los coeficientes de los dos GLM (multiplicador_base y FLAG), las
frecuencias empiricas de TEMA/superficie/formato, y los 3 escenarios de la Tabla 1 del
informe (conservador/intermedio/pesimista).

Todo lo que SE PUEDE derivar de los datos crudos se calcula aca en tiempo de carga
(frecuencias de TEMA, superficie, formato, ratio_por_superficie) en vez de copiarse a
mano desde el .docx - asi queda una sola fuente de verdad y es trivial de auditar. Lo
que NO esta en ningun CSV (solo existe en prosa en TP3_entrega_final.docx) se deja como
constante con un comentario que cita la seccion exacta - ver "Constantes que solo
existen en prosa" mas abajo.

Todos los loaders estan decorados con @lru_cache: se leen los CSV una sola vez por
proceso sin importar cuantas replicas se corran (mismo criterio que evitar leer el
pronostico de demanda 364 veces en tp2/codigo/simulacion.py).

Decisiones de esta sesion (2026-10-06, confirmadas con el usuario, ver
tp3/parte b/CLAUDE.md seccion 7 para el detalle completo):
- El pool de bootstrap incluye account_age_days (parte a lo bootstrapeaba con 4
  columnas, sin esa; el propio .docx dice que se bootstrapea junto con las demas -
  se completa aca el gap, documentado, no silencioso).
- La amortizacion del capex del clasificador usa una anualidad con discount_rate
  (ver armar_params_escenario) - no linea recta, para que el parametro tenga uso.
- Los costos fijos (dotacion, capex, opex) se prorratean a los D dias de la corrida.
"""

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

# ---------------------------------------------------------------------------
# Rutas (resueltas relativas a este archivo, no al cwd - mismo patron que
# DATOS_DIR en tp2/codigo/simulacion.py)
#
# NOTA (paquete de entrega IO2-TP3b-Entrega, autocontenido): en el repo de
# trabajo (tp3/parte b/codigo/) este modulo lee de DOS carpetas Datos/ (la de
# parte a, para parametros_y_costos.csv y el dataset limpio; la propia de
# parte b, para los GLM calibrados) - ver tp3/parte b/CLAUDE.md. Para que este
# paquete de entrega funcione solo, sin depender de que exista una carpeta
# "parte a" hermana, se copiaron los 2 CSV de parte a tambien a este Datos/
# local y se unificaron las rutas en una sola carpeta.
# ---------------------------------------------------------------------------

_AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.normpath(os.path.join(_AQUI, "..", "Datos"))
RESULTADOS_DIR = os.path.normpath(os.path.join(_AQUI, "..", "resultados"))

PATH_PARAMETROS_COSTOS = os.path.join(DATOS_DIR, "parametros_y_costos.csv")
PATH_CASCADAS_LIMPIO = os.path.join(DATOS_DIR, "muestra_cascadas_REDBAIRES_limpio.csv")
PATH_GLM_MULTIPLICADOR = os.path.join(DATOS_DIR, "glm_negbinom_multiplicador_base.csv")
PATH_GLM_FLAG = os.path.join(DATOS_DIR, "glm_logit_flag.csv")

# ---------------------------------------------------------------------------
# Constantes que solo existen en prosa en TP3_entrega_final.docx (no hay CSV
# del que leerlas - se dejan ACA, no desperdigadas en motor.py/costos.py, para
# que quede un solo lugar donde buscar "de donde sale este numero")
# ---------------------------------------------------------------------------

# Tabla 6 (tabla maestra), fila EXITO: "umbral_cliente_historico=1.095 dias"
UMBRAL_CLIENTE_HISTORICO_DIAS = 1095

# multiplicador_base: r fijo de la Binomial Negativa condicional (GLM NB2). El
# CSV de coeficientes no incluye la fila de dispersion (se descarta antes de
# guardar) - el valor final SI quedo documentado en texto:
# alpha estimado = 1,020 -> r = 1/alpha = 0,980 (parte a/CLAUDE.md seccion 5bis,
# y TP3_entrega_final.docx 3.3). Usar ese r directamente.
HIJOS_R = 0.980

# EXITO (3.4/Anexo del .docx): sin datos para calibrar (no hay campo de resultado
# de moderacion en la muestra) - declarado como supuesto libre, misma forma
# logistica que FLAG pero con coeficientes propuestos, no ajustados.
EXITO_BASE = 0.80
EXITO_DELTA_TOX = 0.5
EXITO_DELTA_TEMA_DISARM = 0.3
EXITO_DELTA_INDULGENCIA = -0.4

# Derivada de impressions (Tabla 3): "actor_follower_count x ratio_por_superficie
# x ruido residual". No hay un ajuste de dispersion propio para ese ruido en
# ningun notebook - el unico desvio empirico disponible es el de una regresion
# log-log alternativa que el notebook de parte a exploro pero que NO quedo como
# la formula final (la final es la de ratio_por_superficie, Tabla 3). Se usa ese
# desvio (en escala log) como la mejor aproximacion disponible de la dispersion,
# con ruido multiplicativo lognormal de media 1 (no sesga el promedio).
SIGMA_RESID_IMPRESSIONS_LOG = 0.9445

# Cortes de nivel de audiencia (3.1, limpieza_datos_TP3.ipynb paso 6): bins
# right-open sobre actor_follower_count.
AUDIENCIA_BINS = [0, 10_000, 100_000, 1_000_000, np.inf]
AUDIENCIA_LABELS = ["nano", "micro", "macro", "mega"]

# Temas DISARM-* (tecnica de desinformacion) vs. ORG-* (organico) - usado tanto
# en FLAG/EXITO como en la prioridad de cola de moderacion.
TEMAS_DISARM_PREFIX = "DISARM-"

# KMAX = N x este factor (2.6 del .docx): ~10x el peor caso historico de tamano
# de cascada (~502), escala con N en vez de ser un numero fijo.
KMAX_POR_SEMILLA = 5_000

# lambda de N ~ Poisson(lambda x D): calibrado de forward_datetime en los datos
# reales (parte b/CLAUDE.md seccion 3, punto 4), excluyendo el centinela 2099.
LAMBDA_CASCADAS_POR_DIA = 27.8

# ---------------------------------------------------------------------------
# Escenarios (Tabla 1 del informe, ya cerrados - no se recalculan de ningun CSV
# porque son decisiones de politica, no datos medidos)
# ---------------------------------------------------------------------------

ESCENARIOS = {
    "conservador": dict(
        moderadores_por_turno=2,
        capex_clasificador=0.0,
        opex_anual_clasificador=0.0,
        precision_clasificador=0.72,
        recall_clasificador=0.63,
        umbral_reenvios_rate_limit=20,
        reduccion_alcance_rate_limit=0.0,
        segmento_degradado=frozenset(),
        intensidad_degradacion=0.0,
    ),
    "intermedio": dict(
        moderadores_por_turno=6,
        capex_clasificador=485_000.0,
        opex_anual_clasificador=100_000.0,
        precision_clasificador=0.84,
        recall_clasificador=0.77,
        umbral_reenvios_rate_limit=5,
        reduccion_alcance_rate_limit=0.45,
        segmento_degradado=frozenset({"mega"}),
        intensidad_degradacion=0.50,
    ),
    "pesimista": dict(
        moderadores_por_turno=12,
        capex_clasificador=900_000.0,
        opex_anual_clasificador=180_000.0,
        precision_clasificador=0.95,
        recall_clasificador=0.90,
        umbral_reenvios_rate_limit=1,
        reduccion_alcance_rate_limit=0.90,
        segmento_degradado=frozenset({"macro", "mega"}),
        intensidad_degradacion=0.90,
    ),
}


# ---------------------------------------------------------------------------
# Carga de CSVs (cacheada - una sola lectura por proceso)
# ---------------------------------------------------------------------------

@lru_cache()
def cargar_parametros_costos():
    """Dict {param: valor_base} desde parametros_y_costos.csv (Finanzas)."""
    df = pd.read_csv(PATH_PARAMETROS_COSTOS)
    return dict(zip(df["param"], df["valor_base"]))


@lru_cache()
def _cargar_cascadas_limpio():
    """Carga perezosa y cacheada del dataset limpio completo - lo usan varios
    loaders de abajo (bootstrap, frecuencias, ratio_por_superficie), asi que se
    lee una sola vez y el resto recorta columnas sobre esta copia en memoria."""
    return pd.read_csv(PATH_CASCADAS_LIMPIO)


@lru_cache()
def cargar_pool_bootstrap():
    """Perfiles unicos de actor para el bootstrap conjunto de 2.7/3.3: sample
    con reemplazo preserva las correlaciones reales entre nivel_audiencia,
    verified_flag, account_type_code (y, agregado en esta sesion,
    account_age_days - ver docstring del modulo). Devuelve una lista de tuplas
    (mas rapido de indexar por posicion con rng.integers que un DataFrame,
    dado que se van a hacer decenas de miles de sorteos por replica)."""
    df = _cargar_cascadas_limpio()
    cols = ["actor_follower_count", "verified_flag", "account_type_code",
            "nivel_audiencia", "account_age_days"]
    perfiles = df[cols].drop_duplicates()
    return list(perfiles.itertuples(index=False, name=None))


@lru_cache()
def cargar_frecuencias_categoricas():
    """Frecuencias empiricas (value_counts normalizado) de superficie, formato
    y TEMA. TEMA en particular es un gap que parte a dejo senalado y nunca
    cerro (ningun notebook lo ajusta, ver parte a/CLAUDE.md seccion 8) - se
    cierra aca, calculandolo directo de los datos reales en vez de inventar
    probabilidades.

    Devuelve un dict {columna: (labels_ndarray, probs_ndarray)} listo para
    rng.choice(labels, p=probs)."""
    df = _cargar_cascadas_limpio()
    resultado = {}
    for columna in ("platform_surface_code", "content_type_code", "topic_code"):
        vc = df[columna].value_counts(normalize=True)
        resultado[columna] = (vc.index.to_numpy(), vc.to_numpy())
    return resultado


@lru_cache()
def cargar_ratio_por_superficie():
    """Mediana de impressions/actor_follower_count por superficie (Tabla 3:
    impressions ~ actor_follower_count x ratio_por_superficie x ruido), sobre
    las filas con actor_follower_count>0. Calculado de los datos, no copiado a
    mano, para que quede trazable."""
    df = _cargar_cascadas_limpio()
    sub = df[df["actor_follower_count"] > 0].copy()
    sub["ratio"] = sub["impressions"] / sub["actor_follower_count"]
    return sub.groupby("platform_surface_code")["ratio"].median().to_dict()


def _parsear_termino_glm(term):
    """'C(factor)[T.nivel]' (con o sin 'levels=[...]' de por medio) -> (factor,
    nivel). Un termino que no matchea ese patron (Intercept, una variable
    continua como toxicity_index) devuelve (term, None)."""
    m = re.match(r"^C\(([^,)]+).*?\)\[T\.(.+)\]$", term)
    if m:
        return m.group(1), m.group(2)
    return term, None


def _cargar_coefs_glm(path, columna_coef):
    """Parsea un CSV de coeficientes GLM (formato statsmodels .params volcado a
    CSV) en un dict {factor: {nivel: coef}} + 'intercept' + terminos continuos
    sueltos por nombre. Las categorias de referencia (las que statsmodels no
    imprime por ser el nivel base) simplemente no aparecen como filas - al
    buscarlas con .get(nivel, 0.0) el coeficiente implicito es 0 (multiplicador
    1, no 0), que es el comportamiento correcto y la razon por la que se
    guardan como dict con default 0.0 en vez de, por ejemplo, un DataFrame
    pivoteado que obligaria a rellenar NaN a mano."""
    df = pd.read_csv(path, index_col=0)
    coefs = {"intercept": 0.0}
    for term, coef in df[columna_coef].items():
        if term == "Intercept":
            coefs["intercept"] = float(coef)
            continue
        factor, nivel = _parsear_termino_glm(term)
        if nivel is None:
            coefs[factor] = float(coef)  # termino continuo (toxicity_index, log1p_antig_rel)
        else:
            coefs.setdefault(factor, {})[nivel] = float(coef)
    return coefs


@lru_cache()
def cargar_coefs_multiplicador_base():
    """GLM NB2 (link log) de reshare_count_from_actor ~ superficie + formato +
    rol + nivel_audiencia + log1p(antiguedad_contenido_rel) (3.3 del .docx).
    multiplicador_base = exp(X.beta); se usa coef_log (aditivo en escala log),
    no multiplicador_parcial (multiplicativo), porque sumar logs y exponenciar
    una vez al final es mas limpio que multiplicar varios factores sueltos."""
    return _cargar_coefs_glm(PATH_GLM_MULTIPLICADOR, "coef_log")


@lru_cache()
def cargar_coefs_flag():
    """GLM logistico de moderation_flag ~ toxicity_index + topic_code (3.3 del
    .docx, Datos/ajuste_flag_logit.py). El intercepto de este CSV refleja la
    tasa de flag del clasificador ACTUAL (recall_clasificador=0.63 base) - para
    otros escenarios hay que sumarle el offset delta de calibrar_delta_flag."""
    return _cargar_coefs_glm(PATH_GLM_FLAG, "coef")


# ---------------------------------------------------------------------------
# Formulas que combinan los coeficientes cargados arriba con atributos de un
# nodo puntual - estas SI las usa motor.py directamente (no son solo carga).
# ---------------------------------------------------------------------------

def multiplicador_base(superficie, formato, rol, nivel_audiencia, antig_rel, coefs):
    """exp(X.beta) del GLM NB2 - ver cargar_coefs_multiplicador_base. antig_rel
    es (t_actual - t_semilla_cascada) / vida_util_cascada (adimensional)."""
    log_mult = coefs["intercept"]
    log_mult += coefs.get("platform_surface_code", {}).get(superficie, 0.0)
    log_mult += coefs.get("content_type_code", {}).get(formato, 0.0)
    log_mult += coefs.get("account_type_code", {}).get(rol, 0.0)
    log_mult += coefs.get("nivel_audiencia", {}).get(nivel_audiencia, 0.0)
    log_mult += coefs.get("log1p_antig_rel", 0.0) * math.log1p(max(antig_rel, 0.0))
    return math.exp(log_mult)


def prob_flag(tox, tema, delta, coefs):
    """P(FLAG=1 | TOX, TEMA) del GLM logistico, con el offset delta ya
    calibrado para el escenario (ver calibrar_delta_flag)."""
    eta = coefs["intercept"] + delta
    eta += coefs.get("toxicity_index", 0.0) * tox
    eta += coefs.get("topic_code", {}).get(tema, 0.0)
    return float(expit(eta))


def prob_exito(tox, tema, indulgencia):
    """P(EXITO=1 | TOX, TEMA, indulgencia) - supuesto libre declarado (ver
    constantes EXITO_* arriba), misma forma logistica que FLAG pero sin ajuste
    de datos (no existe campo de resultado de moderacion en la muestra)."""
    eta = logit(EXITO_BASE)
    eta += EXITO_DELTA_TOX * tox
    eta += EXITO_DELTA_TEMA_DISARM * float(tema.startswith(TEMAS_DISARM_PREFIX))
    eta += EXITO_DELTA_INDULGENCIA * float(indulgencia)
    return float(expit(eta))


# ---------------------------------------------------------------------------
# Calibracion del offset delta de FLAG (una vez por escenario, no por replica)
# ---------------------------------------------------------------------------

@lru_cache()
def calibrar_delta_flag(recall_objetivo):
    """Resuelve delta tal que el promedio de P(FLAG=1) sobre la distribucion
    REAL de TOX/TEMA de los datos historicos (no una muestra simulada, para
    que el resultado sea reproducible sin depender de una semilla) iguale
    recall_objetivo. Se cachea por valor de recall (los 3 escenarios tienen
    recall distinto, asi que se calibra una vez por escenario, nunca por
    replica/nodo - recalibrar por nodo seria carisimo y ademas no tiene
    sentido, delta es una propiedad del escenario, no del evento)."""
    df = _cargar_cascadas_limpio()
    coefs = cargar_coefs_flag()
    tox = df["toxicity_index"].to_numpy()
    tema_coef = coefs.get("topic_code", {})
    beta_tema = df["topic_code"].map(lambda t: tema_coef.get(t, 0.0)).to_numpy()
    eta_base = coefs["intercept"] + coefs.get("toxicity_index", 0.0) * tox + beta_tema

    def promedio_menos_objetivo(delta):
        return expit(eta_base + delta).mean() - recall_objetivo

    return brentq(promedio_menos_objetivo, -20.0, 20.0)


# ---------------------------------------------------------------------------
# Bundle de parametros por escenario (lo que motor.py/costos.py reciben)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EscenarioParams:
    nombre: str
    moderadores_por_turno: int
    precision_clasificador: float
    recall_clasificador: float
    umbral_reenvios_rate_limit: int
    reduccion_alcance_rate_limit: float
    segmento_degradado: frozenset
    intensidad_degradacion: float
    capex_clasificador: float
    opex_anual_clasificador: float
    delta_flag: float
    coefs_multiplicador: dict = field(repr=False)
    coefs_flag: dict = field(repr=False)
    pool_bootstrap: list = field(repr=False)
    frecuencias: dict = field(repr=False)
    ratio_por_superficie: dict = field(repr=False)
    costos_base: dict = field(repr=False)


def armar_params_custom(nombre, base_escenario="conservador", **overrides):
    """Variante NO cacheada de armar_params_escenario, para sweeps puntuales
    (stress_test.py) que necesitan variar una sola palanca VD1/VD2/VD3
    manteniendo el resto de un escenario base fijo - ej. moderar con 1, 2, 6,
    12, 24 moderadores sin tener que agregar una fila nueva a ESCENARIOS por
    cada valor que se quiera probar. Si se pasa recall_clasificador en
    overrides, delta_flag se recalibra para ese valor (cacheado aparte, por
    valor de recall, en calibrar_delta_flag)."""
    cfg = dict(ESCENARIOS[base_escenario])
    cfg.update(overrides)
    return EscenarioParams(
        nombre=nombre,
        moderadores_por_turno=cfg["moderadores_por_turno"],
        precision_clasificador=cfg["precision_clasificador"],
        recall_clasificador=cfg["recall_clasificador"],
        umbral_reenvios_rate_limit=cfg["umbral_reenvios_rate_limit"],
        reduccion_alcance_rate_limit=cfg["reduccion_alcance_rate_limit"],
        segmento_degradado=cfg["segmento_degradado"],
        intensidad_degradacion=cfg["intensidad_degradacion"],
        capex_clasificador=cfg["capex_clasificador"],
        opex_anual_clasificador=cfg["opex_anual_clasificador"],
        delta_flag=calibrar_delta_flag(cfg["recall_clasificador"]),
        coefs_multiplicador=cargar_coefs_multiplicador_base(),
        coefs_flag=cargar_coefs_flag(),
        pool_bootstrap=cargar_pool_bootstrap(),
        frecuencias=cargar_frecuencias_categoricas(),
        ratio_por_superficie=cargar_ratio_por_superficie(),
        costos_base=cargar_parametros_costos(),
    )


@lru_cache()
def armar_params_escenario(nombre_escenario):
    """Arma el bundle inmutable de parametros que necesita una replica de este
    escenario. Cacheado por nombre de escenario: la calibracion de delta_flag
    y la carga de CSVs se hacen una sola vez, nunca por replica."""
    cfg = ESCENARIOS[nombre_escenario]
    return EscenarioParams(
        nombre=nombre_escenario,
        moderadores_por_turno=cfg["moderadores_por_turno"],
        precision_clasificador=cfg["precision_clasificador"],
        recall_clasificador=cfg["recall_clasificador"],
        umbral_reenvios_rate_limit=cfg["umbral_reenvios_rate_limit"],
        reduccion_alcance_rate_limit=cfg["reduccion_alcance_rate_limit"],
        segmento_degradado=cfg["segmento_degradado"],
        intensidad_degradacion=cfg["intensidad_degradacion"],
        capex_clasificador=cfg["capex_clasificador"],
        opex_anual_clasificador=cfg["opex_anual_clasificador"],
        delta_flag=calibrar_delta_flag(cfg["recall_clasificador"]),
        coefs_multiplicador=cargar_coefs_multiplicador_base(),
        coefs_flag=cargar_coefs_flag(),
        pool_bootstrap=cargar_pool_bootstrap(),
        frecuencias=cargar_frecuencias_categoricas(),
        ratio_por_superficie=cargar_ratio_por_superficie(),
        costos_base=cargar_parametros_costos(),
    )


if __name__ == "__main__":
    # Chequeo rapido de que todo carga y calibra bien, para correr suelto
    # como primer paso de verificacion (ver plan de implementacion).
    print(f"parametros_y_costos.csv: {len(cargar_parametros_costos())} parametros")
    print(f"pool de bootstrap: {len(cargar_pool_bootstrap())} perfiles unicos")

    frecs = cargar_frecuencias_categoricas()
    for col, (labels, probs) in frecs.items():
        print(f"{col}: {len(labels)} categorias, suma de probs = {probs.sum():.6f}")

    print(f"ratio_por_superficie: {cargar_ratio_por_superficie()}")

    coefs_mult = cargar_coefs_multiplicador_base()
    coefs_flag = cargar_coefs_flag()
    print(f"coefs multiplicador_base: intercept={coefs_mult['intercept']:.4f}, "
          f"{len(coefs_mult) - 1} factores")
    print(f"coefs FLAG: intercept={coefs_flag['intercept']:.4f}")

    for nombre in ESCENARIOS:
        params = armar_params_escenario(nombre)
        print(f"escenario {nombre}: delta_flag={params.delta_flag:.4f} "
              f"(recall objetivo={params.recall_clasificador})")
