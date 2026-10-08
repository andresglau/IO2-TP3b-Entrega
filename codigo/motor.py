"""Motor de eventos discretos - RedBaires (TP3 parte B, IO2).

Implementa UNA replica completa del modelo logico de 2.6 (modelo_logico_simulador.drawio,
verificado celda por celda contra el .docx): N cascadas semilla compitiendo por un pool
compartido de moderadores a traves de una FEL (lista de eventos futuros) compartida, hasta
vaciarse o alcanzar KMAX (tope de computo).

No duplicar logica de negocio fuera de aca - parametros.py solo carga/calibra datos,
costos.py solo traduce los contadores que este motor devuelve a USD. Las decisiones de
modelado (formulas, supuestos declarados) estan documentadas con su cita de seccion del
.docx en cada funcion relevante; para la justificacion completa ver tp3/parte b/CLAUDE.md.

Dos tecnicas que van mas alla de lo visto en el curso (marcadas explicitamente, equivalente
Python de "funcionalidades no cubiertas en clase"):
1. FEL con invalidacion perezosa + purga activa dirigida, combinadas segun si el recurso en
   juego importa o no (decision del usuario, 2026-10-07; ver _contener_cascada): cuando
   CUALQUIER nodo de una cascada logra EXITO=1, la cascada entera se da por "contenida"
   (Cascada.contenida=True) y TODO lo que le quedaba pendiente se corta - no solo los
   descendientes directos del nodo removido, toda la cascada. Los eventos GENERADOR
   pendientes de esa cascada se invalidan PEREZOSAMENTE (un chequeo de un bool al
   principio de _generador, nunca hace falta tocar el heap); pero los nodos que YA tienen
   un moderador asignado (ocupando un recurso escaso de verdad) liberan ese moderador DE
   INMEDIATO, y los que estan en la cola de moderacion se purgan ACTIVAMENTE de ahi - en
   ninguno de los dos casos alcanza con esperar a que les toque el turno, porque estarian
   reteniendo un recurso compartido sin necesidad. Perezoso para lo que no cuesta nada
   dejar pasar, activo para lo que si importa liberar ya.
2. Variables aleatorias calibradas con GLM sobre datos reales (multiplicador_base, FLAG) en
   vez de distribuciones fijas - ver parametros.py.

Unidades: el reloj interno de la simulacion es en MINUTOS (coincide con como estan
calibradas vida_util/TIEMPO_REVISION/tiempo_entre_eventos en la Tabla 6). D (dias a
simular) se usa tal cual para N ~ Poisson(lambda x D) y para el horizonte de arranque de
las semillas, convertido a minutos (D x 1440) para que todo quede en una sola unidad.
"""

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import parametros as p

MINUTOS_POR_DIA = 1440


# ---------------------------------------------------------------------------
# Entidades (2.8 del .docx: "el nodo" es la entidad; "la cascada" no es una
# entidad del flujo pero agrupa el estado que varios nodos comparten - vida
# util, TEMA, RA - asi que tambien se modela como su propia clase)
# ---------------------------------------------------------------------------

@dataclass
class Cascada:
    """Agrupa el estado compartido por todos los nodos de una misma cascada:
    vida_util y TEMA se sortean una sola vez, en la semilla, y se heredan (no
    se vuelven a samplear por nodo) - ver 2.6/2.7 del .docx."""
    cascade_id: int
    t_inicio: float
    vida_util: float = 0.0       # minutos; sorteado al crear la semilla
    tema: str = ""               # sorteado al crear la semilla
    ra: int = 0                  # reenvios acumulados (VD2) - crece durante la replica
    t_ultimo_evento: float = 0.0        # VE tiempo_contencion (extincion natural o contencion)
    t_primera_remocion: Optional[float] = None  # VE tiempo_contencion (primera remocion)
    # Contencion de cascada completa (decision del usuario, 2026-10-07): se
    # activa la primera vez que CUALQUIER nodo de la cascada logra EXITO=1
    # (sea o no "a tiempo" - eso solo decide si ESE nodo acredita sus propias
    # impresiones, no si la cascada se corta). Una vez contenida, nada
    # pendiente de esta cascada vuelve a propagarse ni a acreditar
    # impresiones - ver _contener_cascada.
    contenida: bool = False
    # Nodos de esta cascada que AHORA MISMO tienen un moderador asignado
    # (TIEMPO_REVISION en curso, su evento FINALIZADOR ya esta programado en
    # la FEL) - se necesita este registro para poder liberar esos moderadores
    # de inmediato si la cascada se contiene antes de que llegue su turno.
    nodos_en_revision: set = field(default_factory=set)


@dataclass
class Nodo:
    """Una publicacion o reenvio (2.8 del .docx). Los campos coinciden 1:1 con
    el catalogo de atributos de 2.7/Tabla 2 para que se pueda leer el codigo
    al lado del documento sin tener que traducir nombres."""
    node_id: int
    cascada: Cascada
    parent: Optional["Nodo"]
    t_creacion: float
    es_semilla: bool
    # atributos del actor (bootstrap conjunto, ver parametros.cargar_pool_bootstrap)
    actor_follower_count: int
    verified_flag: bool
    account_type_code: str       # "rol_actor" en el texto del .docx
    nivel_audiencia: str
    account_age_days: int
    # atributos del contenido
    superficie: str
    formato: str
    # derivados / sorteados al procesar el evento GENERADOR de este nodo
    impressions: float = 0.0
    tox: Optional[float] = None
    flag: Optional[bool] = None
    removido: bool = False
    t_rem: Optional[float] = None
    t_encolado: Optional[float] = None  # instante en que entro a la cola de moderacion (si aplica)

    @property
    def indulgencia(self):
        """'cliente historico' o verificado (2.7/2.8) - el moderador es mas
        indulgente con estas cuentas: pesa en EXITO y en la prioridad de cola."""
        return bool(self.verified_flag) or self.account_age_days > p.UMBRAL_CLIENTE_HISTORICO_DIAS


# ---------------------------------------------------------------------------
# Estado mutable de una replica (bookkeeping del motor, no es una entidad del
# modelo de negocio - vive aca, separado de Nodo/Cascada, para no mezclar
# "que es el modelo" con "como lo corremos")
# ---------------------------------------------------------------------------

class _EstadoReplica:
    def __init__(self, params, moderadores_por_turno):
        self.params = params
        self.fel = []                      # heap: (t_evento, seq, tipo, node_id)
        self._seq = itertools.count()
        self.nodos = {}                    # node_id -> Nodo
        self.cascadas = []                 # lista de Cascada (para el cierre)
        self.cola_moderacion = []          # heap: (clave_prioridad, seq, node_id)
        self.moderadores_capacidad = moderadores_por_turno
        self.moderadores_libres = moderadores_por_turno
        self.k = 0
        self.alcance_noc = 0.0
        self.items_flageados = 0
        self.impresiones_recortadas = 0.0
        self.nodo_instancias_degradadas = 0
        # Tiempo de espera en cola por item encolado (VE diagnostica, separada
        # a proposito de "tiempo de contencion": un pool de moderadores
        # saturado (ej. escenario conservador) puede dejar items esperando
        # cientos de miles de minutos antes de ser atendidos, lo cual domina y
        # distorsiona el tiempo de contencion por cascada si no se reporta
        # aparte - ver hallazgo documentado en CLAUDE.md seccion 7.
        self.tiempos_espera_cola = []
        # Latencia de moderacion: desde que un nodo se flaguea (su propio
        # t_creacion, momento en que Centinela lo evalua) hasta que se
        # resuelve (evento FINALIZADOR) - sea cual sea el desenlace (EXITO o
        # no). Equivalente directo a moderation_latency_min en los datos
        # reales (3.1 del .docx: "minutos hasta que se actua sobre el evento
        # marcado... no implica remocion efectiva"), util para contrastar la
        # dinamica simulada contra la real - condicion no negociable del
        # encargo (memo, seccion 8: "debe reproducir lo que observamos").
        self.latencias_moderacion = []
        self._siguiente_node_id = itertools.count()
        # VE ocupacion de moderadores: area bajo la curva de "ocupados(t)",
        # actualizada SOLO en toma/liberacion (ver _registrar_cambio_ocupacion),
        # nunca en un loop periodico - asi lo exige la rubrica.
        self.ve_area_ocupacion = 0.0
        self.ve_ocupados_actual = 0
        self.ve_t_ultimo_cambio = 0.0
        self.t_actual = 0.0
        # trace opcional (graficos.py)
        self.trace = None

    def nuevo_id(self):
        return next(self._siguiente_node_id)

    def push_fel(self, t_evento, tipo, node_id):
        heapq.heappush(self.fel, (t_evento, next(self._seq), tipo, node_id))


# ---------------------------------------------------------------------------
# Generacion de variables aleatorias (envuelven los parametros calibrados de
# parametros.py con la parametrizacion que espera numpy)
# ---------------------------------------------------------------------------

def _sortear_vida_util(rng):
    """Lognormal(sigma=0.887, loc=0, escala=221.26) minutos - Tabla 6."""
    return float(rng.lognormal(mean=math.log(221.26), sigma=0.887))


def _sortear_tiempo_revision(rng):
    """Lognormal(sigma=0.627, loc=0, escala=169.03) minutos - Tabla 6."""
    return float(rng.lognormal(mean=math.log(169.03), sigma=0.627))


def _sortear_tiempo_entre_eventos(rng):
    """Weibull(c=0.674, loc=0, escala=53.09) minutos - Tabla 6. numpy.rng.weibull
    da la Weibull estandar (escala 1); se multiplica por la escala calibrada."""
    return float(53.09 * rng.weibull(0.674))


def _sortear_tox(rng):
    """Beta(a=1.319, b=2.440) - Tabla 6, acotada en [0,1] como corresponde a un
    puntaje de nocividad."""
    return float(rng.beta(1.319, 2.440))


def _elegir(rng, labels_y_probs):
    """rng.choice(labels, p=probs) - envuelto en una funcion para que el
    segundo elemento de la tupla (labels, probs) de parametros.py nunca se
    pase por posicion (ahi rng.choice lo interpretaria como `size`, no como
    `p` - bug real que aparecio al escribir este modulo, ver _generar_
    cascadas_semilla en el historial)."""
    labels, probs = labels_y_probs
    return rng.choice(labels, p=probs)


def _sortear_tema(rng, frecuencias):
    return str(_elegir(rng, frecuencias["topic_code"]))


def _sortear_superficie_formato(rng, frecuencias):
    superficie = _elegir(rng, frecuencias["platform_surface_code"])
    formato = _elegir(rng, frecuencias["content_type_code"])
    return str(superficie), str(formato)


def _bootstrap_actor(rng, pool):
    """Un perfil de actor completo, muestreado con reemplazo del pool de
    perfiles unicos observados - preserva las correlaciones reales entre
    nivel_audiencia/verificado/rol/antiguedad (3.3 del .docx: 'no se ajusta
    distribucion por atributo, se usa bootstrap empirico conjunto')."""
    idx = rng.integers(0, len(pool))
    follower_count, verified, account_type, nivel_audiencia, age_days = pool[idx]
    return follower_count, bool(verified), account_type, nivel_audiencia, int(age_days)


def _derivar_impressions(rng, follower_count, superficie, ratio_por_superficie,
                          antig_rel, coef_log1p_antig_rel):
    """impressions ~ actor_follower_count x ratio_por_superficie x ruido x
    decaimiento_por_antiguedad (Tabla 3 + decision del usuario, 2026-10-08).

    Ruido multiplicativo lognormal de media 1 (no sesga el promedio), con el
    unico desvio empirico disponible (ver SIGMA_RESID_IMPRESSIONS_LOG en
    parametros.py para la salvedad).

    El factor de decaimiento modela que la MISMA cuenta publicando temprano en
    la vida de la cascada (antig_rel chico) alcanza mas gente que publicando
    tarde (antig_rel grande) - la cascada, probabilisticamente, ya perdio
    traccion. Reutiliza EXACTAMENTE la misma transformacion y el mismo
    coeficiente calibrado que ya usa multiplicador_base para HIJOS
    (coefs_multiplicador['log1p_antig_rel'], GLM NB2 de 3.3) en vez de
    inventar un parametro libre nuevo sin dato que lo respalde: es la unica
    medida disponible de "cuanto decae la propagacion con la antiguedad
    relativa del contenido", y se aplica igual a cuanto alcance tiene un nodo
    que a cuantos hijos genera - mismo mecanismo, dos efectos. Decae suave
    (nunca llega a exactamente 0), consistente con que vida_util sigue siendo
    probabilistica, no un corte duro."""
    s = p.SIGMA_RESID_IMPRESSIONS_LOG
    ruido = rng.lognormal(mean=-0.5 * s * s, sigma=s)
    ratio = ratio_por_superficie.get(superficie, 0.0)
    decaimiento = math.exp(coef_log1p_antig_rel * math.log1p(max(antig_rel, 0.0)))
    return max(follower_count, 0) * ratio * ruido * decaimiento


def _sortear_hijos(rng, me, r=p.HIJOS_R):
    """HIJOS ~ BinomialNegativa(media=me, r fijo) - 3.3 del .docx, GLM NB2.
    numpy parametriza negative_binomial(n, prob) como cantidad de fallos antes
    de n exitos con probabilidad prob cada uno; su media es n(1-prob)/prob.
    Para que esa media sea `me` con n=r fijo: prob = r/(r+me) (bug clasico si
    se invierte esta cuenta - ver nota en el docstring del modulo)."""
    me = max(me, 0.0)
    prob = r / (r + me)
    return int(rng.negative_binomial(r, prob))


# ---------------------------------------------------------------------------
# Formulas de 2.7 (ME, prioridad de cola) que combinan sorteos + parametros
# ---------------------------------------------------------------------------

def _calcular_me(nodo, antig_rel, params):
    """ME = multiplicador_base(...) x (1 - reduccion x [RA>=umbral]) x
    (1 - intensidad x [segmento targeteado]) - formula textual de 2.7, VD2 y
    VD3 son independientes y acumulables sobre el mismo nodo (no excluyentes).

    Devuelve ademas si cada palanca disparo sobre ESTE nodo (independiente del
    resultado de FLAG mas adelante) - costos.py/_generador los usan para los
    costos de friccion y degradacion, que dependen de que la palanca haya
    actuado, no de si Centinela despues marco el nodo."""
    base = p.multiplicador_base(
        nodo.superficie, nodo.formato, nodo.account_type_code,
        nodo.nivel_audiencia, antig_rel, params.coefs_multiplicador)

    targeteado_vd2 = nodo.cascada.ra >= params.umbral_reenvios_rate_limit
    factor_vd2 = (1.0 - params.reduccion_alcance_rate_limit) if targeteado_vd2 else 1.0

    targeteado_vd3 = nodo.nivel_audiencia in params.segmento_degradado
    factor_vd3 = (1.0 - params.intensidad_degradacion) if targeteado_vd3 else 1.0

    me = max(base * factor_vd2 * factor_vd3, 0.0)
    return me, targeteado_vd2, targeteado_vd3


def clave_prioridad_moderacion(nodo):
    """Regla de prioridad no-FIFO de la cola de moderacion (2.6/2.8: 'prioriza
    segun TEMA y antiguedad/verificado', sin formula concreta en el .docx ni
    en el .drawio - decision tomada y confirmada con el usuario 2026-10-06,
    ver tp3/parte b/CLAUDE.md seccion 7):

        1) TEMA es DISARM-* (tecnica de desinformacion) se atiende ANTES que
           ORG-* (organico) - "mas severidad con temas sensibles".
        2) dentro de cada grupo, cuentas NO indulgentes (ni verificadas ni
           clientes historicos) se atienden antes que las indulgentes -
           "mas indulgentes con verificadas/historicos" se traduce en que
           esperan mas.
        3) empate -> FIFO por instante de creacion.

    Aislada en esta unica funcion a proposito: si se decide otra regla, es el
    UNICO lugar que hay que tocar - nada mas en motor.py depende de los
    detalles de esta formula, solo de que exista una clave ordenable."""
    es_disarm = nodo.cascada.tema.startswith(p.TEMAS_DISARM_PREFIX)
    return (0 if es_disarm else 1, 1 if nodo.indulgencia else 0, nodo.t_creacion)


# ---------------------------------------------------------------------------
# VE de ocupacion de moderadores: actualizada EXCLUSIVAMENTE en los eventos
# que cambian cuantos moderadores estan ocupados (toma/liberacion), nunca en
# un loop periodico - requisito explicito de la rubrica.
# ---------------------------------------------------------------------------

def _registrar_cambio_ocupacion(estado, t_actual, ocupados_nuevo):
    estado.ve_area_ocupacion += estado.ve_ocupados_actual * (t_actual - estado.ve_t_ultimo_cambio)
    estado.ve_ocupados_actual = ocupados_nuevo
    estado.ve_t_ultimo_cambio = t_actual


def _tomar_moderador(nodo, estado, rng, t_actual):
    """Toma un moderador libre (inline, duracion cero - no es un evento propio
    de la FEL, ver 2.8 del .docx) y programa el 'Fin de atencion' de este
    nodo. La usan tanto la rama GENERADOR (moderador libre al momento del
    flag) como la rama FINALIZADOR (siguiente de la cola al liberarse uno) -
    una sola funcion, nunca dos copias de esta logica."""
    if nodo.t_encolado is not None:
        estado.tiempos_espera_cola.append(t_actual - nodo.t_encolado)
    estado.moderadores_libres -= 1
    ocupados = estado.moderadores_capacidad - estado.moderadores_libres
    _registrar_cambio_ocupacion(estado, t_actual, ocupados)
    nodo.cascada.nodos_en_revision.add(nodo.node_id)
    tiempo_revision = _sortear_tiempo_revision(rng)
    estado.push_fel(t_actual + tiempo_revision, "FINALIZADOR", nodo.node_id)


def _liberar_moderador(estado, t_actual):
    estado.moderadores_libres += 1
    ocupados = estado.moderadores_capacidad - estado.moderadores_libres
    _registrar_cambio_ocupacion(estado, t_actual, ocupados)


def _servir_cola_moderacion(estado, rng, t_actual):
    """Mientras haya moderadores libres y cola, asigna el siguiente por
    prioridad. Es un `while`, no un `if`, a proposito: cuando una cascada se
    contiene (_contener_cascada) puede liberar varios moderadores de golpe
    (el del nodo que disparo la contencion + los de otros nodos que estaban
    en revision), y todos esos moderadores recien liberados tienen que poder
    tomar trabajo nuevo en la misma pasada, no uno por evento."""
    while estado.moderadores_libres > 0 and estado.cola_moderacion:
        _, _, siguiente_id = heapq.heappop(estado.cola_moderacion)
        siguiente = estado.nodos[siguiente_id]
        _tomar_moderador(siguiente, estado, rng, t_actual)


def _intentar_tomar_moderador_o_encolar(nodo, estado, rng, t_actual):
    if estado.moderadores_libres > 0:
        _tomar_moderador(nodo, estado, rng, t_actual)
    else:
        nodo.t_encolado = t_actual
        clave = clave_prioridad_moderacion(nodo)
        heapq.heappush(estado.cola_moderacion, (clave, next(estado._seq), nodo.node_id))


def _fraccion_expuesta(t_creacion, t_resolucion, vida_util):
    """Fraccion de `impressions` a acreditar para un nodo que circulo desde su
    propia creacion hasta `t_resolucion` (remocion exitosa propia, o
    contencion de su cascada disparada por OTRO nodo) - capada en 1 (decision
    del usuario, 2026-10-08/09). Se combina con, pero es independiente de, el
    decaimiento por antiguedad que ya aplica _derivar_impressions al nacer el
    nodo: aquel pone un techo mas bajo para posts tardios (publicar cuando la
    cascada ya perdio traccion alcanza menos gente, pase lo que pase despues);
    esta fraccion ademas recorta ESE techo si la moderacion actua rapido en
    relacion al mismo `vida_util` - dos efectos distintos del mismo reloj,
    uno en el nacimiento, otro en la resolucion."""
    if vida_util <= 0:
        return 1.0
    return min(1.0, max(0.0, (t_resolucion - t_creacion) / vida_util))


def _contener_cascada(cascada, estado, t):
    """Se dispara la primera vez que CUALQUIER nodo de la cascada logra
    EXITO=1 (decision del usuario, 2026-10-07 - antes la cancelacion solo
    alcanzaba a los hijos directos del nodo removido, chequeando tiempos). A
    partir de aca, TODO lo que siga pendiente de esta cascada queda cortado:
    ni los nodos cuyo propio evento GENERADOR todavia no se proceso, ni los
    que ya estan en la cola de moderacion, ni los que ya tienen un moderador
    asignado, llegan a propagarse mas - pero SI acreditan a ALCANCE_NOC la
    fraccion de sus impresiones correspondiente al tiempo que ya llevaban
    circulando (ver _fraccion_expuesta, decision del usuario 2026-10-08/09):
    ya fueron publicados y clasificados por Centinela (tienen `impressions`
    derivadas), simplemente la cascada se corto antes de que SU PROPIA
    moderacion terminara - no es lo mismo que un nodo cuyo GENERADOR nunca se
    proceso (ese nunca "existio", no acredita nada).

    Los nodos con moderador YA asignado (TIEMPO_REVISION en curso) liberan su
    moderador DE INMEDIATO, no esperan al tiempo ya sorteado (decision del
    usuario: si el contenido ya no existe, no tiene sentido seguir
    "revisando" algo que ya no importa). El nodo que disparo la contencion
    NO esta en `nodos_en_revision` en este punto: _finalizador ya lo saco de
    ahi antes de llamar a esta funcion, y su propio moderador y credito ya se
    resolvieron por separado, para no duplicarlos aca."""
    if cascada.contenida:
        return
    cascada.contenida = True
    cascada.t_ultimo_evento = t

    for node_id in cascada.nodos_en_revision:
        nodo = estado.nodos[node_id]
        estado.alcance_noc += nodo.impressions * _fraccion_expuesta(
            nodo.t_creacion, t, cascada.vida_util)
        estado.moderadores_libres += 1
        ocupados = estado.moderadores_capacidad - estado.moderadores_libres
        _registrar_cambio_ocupacion(estado, t, ocupados)
    cascada.nodos_en_revision.clear()

    # Purga activa de la cola de moderacion (no lazy, a diferencia del resto
    # del motor): acá SI hace falta buscar y sacar, porque un item encolado
    # no tiene ningun evento propio en la FEL todavia que pueda chequear
    # "soy de una cascada contenida" por su cuenta - si no se saca de la
    # cola ahora, quedaria esperando un moderador para siempre (o hasta que
    # le toque el turno y recien ahi se descarte, ocupando un lugar en la
    # cola sin necesidad). Se acredita antes de descartar, mismo criterio
    # que los nodos en revision de arriba.
    nueva_cola = []
    for entrada in estado.cola_moderacion:
        nodo = estado.nodos[entrada[2]]
        if nodo.cascada is cascada:
            estado.alcance_noc += nodo.impressions * _fraccion_expuesta(
                nodo.t_creacion, t, cascada.vida_util)
        else:
            nueva_cola.append(entrada)
    estado.cola_moderacion = nueva_cola
    heapq.heapify(estado.cola_moderacion)


# ---------------------------------------------------------------------------
# Rama GENERADOR (nace o se propaga un nodo) - 2.6 del .docx / .drawio
# ---------------------------------------------------------------------------

def _generador(nodo, estado, rng):
    params = estado.params
    t = nodo.t_creacion
    estado.t_actual = t

    # Paso 1: invalidacion perezosa por contencion de cascada (ver
    # _contener_cascada). Reemplaza el chequeo anterior, que solo miraba si
    # el padre INMEDIATO habia sido removido antes del propio instante de
    # este nodo: ahora CUALQUIER nodo de la cascada que logre EXITO=1 corta
    # toda la cascada de una, asi que alcanza con este chequeo (mas simple
    # y, de paso, estrictamente mas amplio que el anterior - lo subsume).
    if nodo.cascada.contenida:
        return
    nodo.cascada.t_ultimo_evento = t

    # Paso 2: solo la semilla sortea vida_util y TEMA (se heredan - 2.6/2.7).
    if nodo.es_semilla:
        nodo.cascada.vida_util = _sortear_vida_util(rng)
        nodo.cascada.tema = _sortear_tema(rng, params.frecuencias)

    # Paso 3-4: antiguedad relativa (ANTES de impressions, ahora que impressions
    # tambien depende de ella - decision del usuario, 2026-10-08) + impressions
    # propias de este nodo.
    vida_util = nodo.cascada.vida_util or 1.0  # guarda defensiva, siempre >0 en la practica
    antig_rel = (nodo.t_creacion - nodo.cascada.t_inicio) / vida_util
    coef_antig = params.coefs_multiplicador.get("log1p_antig_rel", 0.0)
    nodo.impressions = _derivar_impressions(
        rng, nodo.actor_follower_count, nodo.superficie, params.ratio_por_superficie,
        antig_rel, coef_antig)

    # Paso 5-7: ME, HIJOS, actualizar RA (con el valor de RA ANTES de sumar
    # los hijos de este nodo - el chequeo de VD2 es sobre lo acumulado hasta
    # ahora, no incluye la propia contribucion de este nodo).
    me, targeteado_vd2, targeteado_vd3 = _calcular_me(nodo, antig_rel, params)
    if targeteado_vd2:
        # Friccion: impresiones de ESTE nodo que no se generaron porque VD2
        # le recorto el alcance - se contabiliza apenas se sabe que VD2
        # disparo, independiente de si Centinela despues lo marca (FLAG) o no.
        estado.impresiones_recortadas += nodo.impressions * params.reduccion_alcance_rate_limit
    if targeteado_vd3:
        estado.nodo_instancias_degradadas += 1
    hijos = _sortear_hijos(rng, me)
    nodo.cascada.ra += hijos

    # Paso 8: crear cada hijo y programar su propio evento GENERADOR.
    for _ in range(hijos):
        follower_count, verified, account_type, nivel_audiencia, age_days = \
            _bootstrap_actor(rng, params.pool_bootstrap)
        superficie, formato = _sortear_superficie_formato(rng, params.frecuencias)
        dt = _sortear_tiempo_entre_eventos(rng)
        hijo = Nodo(
            node_id=estado.nuevo_id(), cascada=nodo.cascada, parent=nodo,
            t_creacion=t + dt, es_semilla=False,
            actor_follower_count=follower_count, verified_flag=verified,
            account_type_code=account_type, nivel_audiencia=nivel_audiencia,
            account_age_days=age_days, superficie=superficie, formato=formato,
        )
        estado.nodos[hijo.node_id] = hijo
        estado.push_fel(hijo.t_creacion, "GENERADOR", hijo.node_id)

    # Paso 9: Centinela evalua ESTE nodo (no los hijos recien creados).
    nodo.tox = _sortear_tox(rng)
    p_flag = p.prob_flag(nodo.tox, nodo.cascada.tema, params.delta_flag, params.coefs_flag)
    nodo.flag = rng.random() < p_flag

    # Paso 10-11: sin flag, impresiones se acreditan ya; con flag, quedan
    # pendientes del desenlace en el FINALIZADOR (ver mas abajo).
    if not nodo.flag:
        estado.alcance_noc += nodo.impressions
    else:
        estado.items_flageados += 1
        _intentar_tomar_moderador_o_encolar(nodo, estado, rng, t)


# ---------------------------------------------------------------------------
# Rama FINALIZADOR (resuelve un "Fin de atencion") - 2.6 del .docx / .drawio
# ---------------------------------------------------------------------------

def _finalizador(nodo, estado, rng):
    t = estado.t_actual  # ya seteado al tiempo de este evento por el loop principal

    # Si la cascada ya fue contenida por OTRO nodo mientras este esperaba su
    # turno, no hay nada que hacer: su moderador ya se liberó en el momento
    # de la contención (_contener_cascada), no de nuevo aca.
    if nodo.cascada.contenida:
        return

    nodo.cascada.t_ultimo_evento = t
    nodo.cascada.nodos_en_revision.discard(nodo.node_id)
    estado.latencias_moderacion.append(t - nodo.t_creacion)

    p_exito = p.prob_exito(nodo.tox, nodo.cascada.tema, nodo.indulgencia)
    exito = rng.random() < p_exito

    if exito:
        # EXITO=1 => el moderador removio el nodo, punto. "a_tiempo" (si la
        # remocion fue antes de que pasara vida_util) ya NO es una condicion
        # para nada aca - existia en versiones previas del motor para decidir
        # todo-o-nada en el credito de impresiones, pero desde que el credito
        # es proporcional (_fraccion_expuesta, mas abajo) esa distincion dejo
        # de hacer falta: la fraccion ya captura continuamente cuanto circulo
        # el nodo, sin el salto artificial que tenia la version vieja.
        # Corregido 2026-10-09 (antes nodo.removido solo se marcaba si
        # a_tiempo, inconsistente con _contener_cascada, que ya dispara con
        # CUALQUIER exito=True). nodo.removido/t_rem son puramente
        # diagnosticos (nada los lee en el resto del motor, verificado), asi
        # que este fix no cambia ningun resultado numerico.
        nodo.removido = True
        nodo.t_rem = t
        if nodo.cascada.t_primera_remocion is None:
            nodo.cascada.t_primera_remocion = t - nodo.cascada.t_inicio

        # Credito proporcional al tiempo que circulo este nodo en relacion a
        # vida_util, capado en 1 (decision del usuario, 2026-10-08/09) - NO
        # es todo-o-nada segun "a_tiempo": una remocion exitosa que tarda
        # mucho mas que vida_util naturalmente da fraccion~1 (credito casi
        # completo, el nodo ya habia circulado su vida natural igual), y una
        # remocion rapida sobre un nodo nacido tarde da una fraccion chica
        # aunque no sea "a tiempo" segun el reloj de la cascada - un solo
        # calculo, sin el salto artificial que tenia la version todo-o-nada.
        fraccion = _fraccion_expuesta(nodo.t_creacion, t, nodo.cascada.vida_util)
        estado.alcance_noc += nodo.impressions * fraccion
    else:
        estado.alcance_noc += nodo.impressions

    _liberar_moderador(estado, t)

    # Cualquier EXITO=1 corta el resto de la cascada, sea o no "a tiempo" -
    # "a tiempo" (arriba) solo decidio si ESTE nodo acredita sus propias
    # impresiones, no si la cascada sigue viva (decision del usuario,
    # 2026-10-07). _contener_cascada puede liberar moderadores adicionales
    # (otros nodos de esta cascada que estaban en revision), por eso servir
    # la cola va DESPUES y en loop, no un solo pop.
    if exito:
        _contener_cascada(nodo.cascada, estado, t)

    _servir_cola_moderacion(estado, rng, t)


# ---------------------------------------------------------------------------
# Preambulo: sortear las N cascadas semilla y cargar sus eventos iniciales
# ---------------------------------------------------------------------------

def _generar_cascadas_semilla(rng, estado, D):
    """N ~ Poisson(lambda x D); los N instantes de inicio se reparten
    Uniforme(0, D) i.i.d. (propiedad estandar de un proceso de Poisson
    homogeneo condicionado a la cantidad de llegadas en la ventana - Ross,
    citado en 2.4 del .docx). D esta en dias para el sorteo de N (asi se
    calibro lambda); los instantes se convierten a minutos para el reloj
    interno de la simulacion."""
    n = int(rng.poisson(p.LAMBDA_CASCADAS_POR_DIA * D))
    horizonte_min = D * MINUTOS_POR_DIA
    instantes = rng.uniform(0, horizonte_min, size=n) if n > 0 else []

    for t_inicio in instantes:
        cascada = Cascada(cascade_id=len(estado.cascadas), t_inicio=float(t_inicio),
                           t_ultimo_evento=float(t_inicio))
        estado.cascadas.append(cascada)
        semilla = Nodo(
            node_id=estado.nuevo_id(), cascada=cascada, parent=None,
            t_creacion=float(t_inicio), es_semilla=True,
            **dict(zip(
                ("actor_follower_count", "verified_flag", "account_type_code",
                 "nivel_audiencia", "account_age_days"),
                _bootstrap_actor(rng, estado.params.pool_bootstrap))),
            superficie=str(_elegir(rng, estado.params.frecuencias["platform_surface_code"])),
            formato=str(_elegir(rng, estado.params.frecuencias["content_type_code"])),
        )
        estado.nodos[semilla.node_id] = semilla
        estado.push_fel(semilla.t_creacion, "GENERADOR", semilla.node_id)

    return n


# ---------------------------------------------------------------------------
# Punto de entrada: una replica completa
# ---------------------------------------------------------------------------

def correr_replica(escenario, D, seed, devolver_trace=False, params=None, kmax_override=None):
    """Corre UNA replica completa (las N cascadas semilla conviviendo en la
    FEL compartida hasta vaciarse o alcanzar KMAX) - ver 2.4/2.6 del .docx.
    No es una corrida de M replicas (eso es monte_carlo.py, que llama a esta
    funcion M veces con semillas distintas).

    `escenario` es el nombre con el que se etiqueta el resultado; por defecto
    tambien se usa para buscar los parametros en parametros.ESCENARIOS. Si se
    pasa `params` (un EscenarioParams ya armado - ver
    parametros.armar_params_custom), se usa ese en vez de buscarlo por
    nombre, para poder correr sweeps de una sola palanca VD1/VD2/VD3 sin
    tener que agregar una fila nueva a ESCENARIOS (ver stress_test.py).
    `kmax_override`, si se pasa, reemplaza el KMAX=N*5000 calculado - solo
    para el caso de prueba que verifica el corte limpio por KMAX
    (stress_test.py); en una corrida normal se deja en None."""
    rng = np.random.default_rng(seed)
    if params is None:
        params = p.armar_params_escenario(escenario)
    estado = _EstadoReplica(params, params.moderadores_por_turno)
    if devolver_trace:
        estado.trace = dict(ocupacion=[])

    n = _generar_cascadas_semilla(rng, estado, D)
    kmax = kmax_override if kmax_override is not None else max(n, 1) * p.KMAX_POR_SEMILLA

    while estado.fel and estado.k < kmax:
        t_evento, _, tipo, node_id = heapq.heappop(estado.fel)
        estado.k += 1
        estado.t_actual = t_evento
        nodo = estado.nodos[node_id]
        if tipo == "GENERADOR":
            _generador(nodo, estado, rng)
        else:
            _finalizador(nodo, estado, rng)
        if devolver_trace:
            estado.trace["ocupacion"].append(
                (t_evento, estado.moderadores_capacidad - estado.moderadores_libres))

    truncado_por_kmax = bool(estado.fel) and estado.k >= kmax

    # Flush final del VE de ocupacion (cierra la integral en el ultimo instante
    # procesado, o en 0 si la replica no proceso ningun evento - N=0).
    _registrar_cambio_ocupacion(estado, estado.t_actual, estado.ve_ocupados_actual)
    t_cierre = max(estado.t_actual, 1e-9)
    ocupacion_promedio = estado.ve_area_ocupacion / (t_cierre * estado.moderadores_capacidad)

    tiempos_extincion = [c.t_ultimo_evento - c.t_inicio for c in estado.cascadas]
    tiempos_primera_remocion = [c.t_primera_remocion for c in estado.cascadas
                                 if c.t_primera_remocion is not None]

    resultado = dict(
        escenario=escenario, D=D, seed=seed,
        n_semillas=n, k=estado.k, kmax=kmax, truncado_por_kmax=truncado_por_kmax,
        alcance_noc=estado.alcance_noc,
        items_flageados=estado.items_flageados,
        impresiones_recortadas=estado.impresiones_recortadas,
        nodo_instancias_degradadas=estado.nodo_instancias_degradadas,
        ocupacion_moderadores_promedio=ocupacion_promedio,
        latencia_moderacion_promedio=(
            float(np.mean(estado.latencias_moderacion)) if estado.latencias_moderacion else 0.0),
        latencia_moderacion_mediana=(
            float(np.median(estado.latencias_moderacion)) if estado.latencias_moderacion else 0.0),
        items_encolados=len(estado.tiempos_espera_cola),
        espera_cola_promedio=(
            float(np.mean(estado.tiempos_espera_cola)) if estado.tiempos_espera_cola else 0.0),
        espera_cola_mediana=(
            float(np.median(estado.tiempos_espera_cola)) if estado.tiempos_espera_cola else 0.0),
        tiempo_contencion_extincion_promedio=(
            float(np.mean(tiempos_extincion)) if tiempos_extincion else 0.0),
        tiempo_contencion_primera_remocion_promedio=(
            float(np.mean(tiempos_primera_remocion)) if tiempos_primera_remocion else None),
    )
    if devolver_trace:
        resultado["trace"] = estado.trace
        resultado["trace"]["tiempos_extincion"] = tiempos_extincion
        resultado["trace"]["tiempos_primera_remocion"] = tiempos_primera_remocion
    return resultado


if __name__ == "__main__":
    # Chequeo de humo (checkpoint 2 del plan de implementacion): una replica
    # suelta del escenario conservador, sin excepciones, con numeros sensatos.
    r = correr_replica("conservador", D=30, seed=42)
    print(f"N semillas: {r['n_semillas']}")
    print(f"K eventos procesados: {r['k']} / KMAX={r['kmax']} "
          f"(truncado={r['truncado_por_kmax']})")
    print(f"ALCANCE_NOC: {r['alcance_noc']:,.0f} impresiones")
    print(f"items flageados: {r['items_flageados']}")
    print(f"impresiones recortadas por rate-limit: {r['impresiones_recortadas']:,.0f}")
    print(f"nodo-instancias degradadas (VD3): {r['nodo_instancias_degradadas']}")
    print(f"ocupacion promedio de moderadores: {r['ocupacion_moderadores_promedio']:.2%}")
    print(f"items que pasaron por la cola: {r['items_encolados']} "
          f"(espera promedio={r['espera_cola_promedio']:.0f} min, "
          f"mediana={r['espera_cola_mediana']:.0f} min)")
    print(f"latencia de moderacion (flag->resolucion), equivalente a "
          f"moderation_latency_min real: promedio={r['latencia_moderacion_promedio']:.0f} "
          f"min, mediana={r['latencia_moderacion_mediana']:.0f} min "
          f"(real: promedio=238 min, mediana=140 min)")
    print(f"tiempo de contencion (extincion), promedio: "
          f"{r['tiempo_contencion_extincion_promedio']:.1f} min")
    print(f"tiempo de contencion (primera remocion), promedio: "
          f"{r['tiempo_contencion_primera_remocion_promedio']}")
    assert 0.0 <= r["ocupacion_moderadores_promedio"] <= 1.0
    assert r["k"] <= r["kmax"]
    print("\nOK - sin excepciones, invariantes basicos verificados.")
