"""Stress test y debugging - RedBaires (TP3 parte B, IO2).

Barre las tres palancas de decision (VD1/VD2/VD3) una por vez, ejercita los dos casos
limite que la rubrica pide poder mostrar (KMAX forzado chico, cola de moderacion
saturada) y corre un chequeo de invariantes sobre cada punto del barrido. Es el
"stress test y debugging, documentar que se encontro" que pide la rubrica - no es un
test automatizado en el sentido de pytest, es un script que se corre y deja un registro
legible de lo que se probo y que broke/se confirmo en el camino.

Uso: python3 stress_test.py (corre todo, imprime una tabla de hallazgos y la guarda en
resultados/stress_test_hallazgos.csv)

--- Hallazgo real documentado durante el desarrollo (2026-10-06) ---
Al probar el motor por primera vez con el escenario conservador (2 moderadores) y
D=30 dias, el tiempo de contencion promedio por cascada dio ~175 dias - a primera
vista parecia un bug (vida_util tiene mediana ~221 MINUTOS, no dias). La investigacion
(ver tp3/parte b/CLAUDE.md seccion 7 para el detalle completo) encontro que NO es
un bug: el tamano de las cascadas es razonable (mediana 2 nodos, maximo 681, comparable
al maximo real de ~502), pero con solo 2 moderadores compitiendo por el flag-rate que
generan ~860 cascadas en 30 dias, la cola de moderacion queda saturada casi todo el
tiempo (occupancy ~99.99%) y los items encolados esperan cientos de miles de minutos
antes de ser atendidos - eso es lo que dominaba el promedio. Se separo "tiempo de
contencion" en dos metricas (extincion natural / primera remocion exitosa) y se agrego
una tercera, "espera en cola", que aisla exactamente este fenomeno (ver motor.py,
resultado de correr_replica). El sweep de VD1 de abajo reproduce esto a proposito,
variando moderadores_por_turno, para dejarlo documentado como comportamiento esperado
del modelo (un pool de moderadores insuficiente SE SATURA, no es un error de calculo).
"""

import csv
import os

import numpy as np

import motor
import parametros as p

D_SWEEP = 14  # dias mas cortos que una corrida real, alcanza para ver el efecto y corre rapido
SEEDS_POR_PUNTO = 5  # un puñado de replicas por punto de barrido, no un Monte Carlo completo

ARCHIVO_SALIDA = os.path.join(p.RESULTADOS_DIR, "stress_test_hallazgos.csv")


def _correr_puntos(params, D, n_replicas=SEEDS_POR_PUNTO, seed_base=9_000_000):
    """Corre n_replicas con los mismos params y devuelve la lista de
    resultados crudos de motor.correr_replica - para promediar/inspeccionar."""
    return [motor.correr_replica(params.nombre, D, seed=seed_base + i, params=params)
            for i in range(n_replicas)]


def _chequear_invariantes(resultados):
    """Los invariantes que tienen que cumplirse SIEMPRE, sea cual sea la
    combinacion de VD - si alguno falla, hay un bug real, no una
    particularidad de un escenario."""
    for r in resultados:
        assert 0.0 <= r["ocupacion_moderadores_promedio"] <= 1.0, \
            f"ocupacion fuera de [0,1]: {r['ocupacion_moderadores_promedio']}"
        assert r["alcance_noc"] >= 0.0, f"ALCANCE_NOC negativo: {r['alcance_noc']}"
        assert r["k"] <= r["kmax"], f"K={r['k']} supero KMAX={r['kmax']}"
        assert r["items_flageados"] >= 0
        assert r["nodo_instancias_degradadas"] >= 0
        assert r["impresiones_recortadas"] >= 0.0


def _resumen(resultados):
    return dict(
        alcance_noc_medio=np.mean([r["alcance_noc"] for r in resultados]),
        ocupacion_media=np.mean([r["ocupacion_moderadores_promedio"] for r in resultados]),
        items_flageados_medio=np.mean([r["items_flageados"] for r in resultados]),
        espera_cola_media=np.mean([r["espera_cola_promedio"] for r in resultados]),
        k_medio=np.mean([r["k"] for r in resultados]),
    )


def sweep_vd1_moderadores(hallazgos):
    """VD1: a mas moderadores, se espera que la ocupacion y la espera en cola
    bajen (mas capacidad para el mismo flag-rate) - VD2/VD3 quedan en 0
    (como en el escenario conservador) para aislar el efecto de VD1 solo."""
    print("\n=== Sweep VD1 (moderadores_por_turno) ===")
    for moderadores in (1, 2, 6, 12, 24):
        params = p.armar_params_custom(f"vd1_{moderadores}", base_escenario="conservador",
                                        moderadores_por_turno=moderadores)
        resultados = _correr_puntos(params, D_SWEEP)
        _chequear_invariantes(resultados)
        resumen = _resumen(resultados)
        print(f"moderadores={moderadores:3d}  ocupacion={resumen['ocupacion_media']:6.1%}  "
              f"espera_cola={resumen['espera_cola_media']:10.0f} min  "
              f"items_flageados={resumen['items_flageados_medio']:.0f}")
        hallazgos.append(dict(
            sweep="VD1_moderadores", valor=moderadores, **resumen,
            nota="ocupacion y espera en cola deberian bajar monotonamente con mas moderadores"))


def sweep_vd2_rate_limit(hallazgos):
    """VD2: a mayor reduccion (y menor umbral, dispara mas seguido), ME baja
    mas para las cascadas largas -> se espera menos ALCANCE_NOC."""
    print("\n=== Sweep VD2 (umbral/reduccion de rate-limit) ===")
    puntos = [(20, 0.0), (10, 0.25), (5, 0.45), (1, 0.90)]
    for umbral, reduccion in puntos:
        params = p.armar_params_custom(
            f"vd2_{umbral}_{reduccion}", base_escenario="conservador",
            umbral_reenvios_rate_limit=umbral, reduccion_alcance_rate_limit=reduccion)
        resultados = _correr_puntos(params, D_SWEEP)
        _chequear_invariantes(resultados)
        resumen = _resumen(resultados)
        recortadas = np.mean([r["impresiones_recortadas"] for r in resultados])
        print(f"umbral={umbral:3d} reduccion={reduccion:.0%}  "
              f"ALCANCE_NOC={resumen['alcance_noc_medio']:,.0f}  "
              f"impresiones_recortadas={recortadas:,.0f}")
        hallazgos.append(dict(
            sweep="VD2_rate_limit", valor=f"umbral={umbral},reduccion={reduccion}",
            **resumen, nota="impresiones_recortadas>0 solo si reduccion>0 (chequeado)"))
        if reduccion == 0.0:
            assert recortadas == 0.0, "reduccion=0 pero se recortaron impresiones igual (bug)"


def sweep_vd3_degradacion(hallazgos):
    """VD3: solo nodos en el segmento targeteado deberian generar
    nodo_instancias_degradadas>0; con segmento vacio, siempre 0."""
    print("\n=== Sweep VD3 (segmento degradado / intensidad) ===")
    puntos = [(frozenset(), 0.0), (frozenset({"mega"}), 0.50),
              (frozenset({"macro", "mega"}), 0.90)]
    for segmento, intensidad in puntos:
        params = p.armar_params_custom(
            f"vd3_{segmento}_{intensidad}", base_escenario="conservador",
            segmento_degradado=segmento, intensidad_degradacion=intensidad)
        resultados = _correr_puntos(params, D_SWEEP)
        _chequear_invariantes(resultados)
        resumen = _resumen(resultados)
        degradadas = np.mean([r["nodo_instancias_degradadas"] for r in resultados])
        etiqueta_segmento = ", ".join(sorted(segmento)) or "ninguno"
        print(f"segmento={etiqueta_segmento:20} intensidad={intensidad:.0%}  "
              f"nodo_instancias_degradadas={degradadas:.1f}")
        hallazgos.append(dict(
            sweep="VD3_degradacion", valor=f"segmento={sorted(segmento)},intensidad={intensidad}",
            **resumen, nota="segmento vacio -> degradadas debe ser exactamente 0 (chequeado)"))
        if not segmento:
            assert degradadas == 0.0, "segmento vacio pero hubo nodos degradados (bug)"


def caso_kmax_forzado(hallazgos):
    """Fuerza un KMAX artificialmente chico y verifica que el motor corta
    limpio (sin excepcion, sin loop infinito) y que lo reporta en
    truncado_por_kmax - evidencia directa de que el acotamiento computacional
    (condicion no negociable del encargo) funciona."""
    print("\n=== Caso limite: KMAX forzado chico ===")
    params = p.armar_params_escenario("pesimista")  # el mas agresivo en generar eventos
    r = motor.correr_replica("pesimista", D=30, seed=123, params=params, kmax_override=10)
    print(f"KMAX forzado=10 -> K={r['k']}, truncado_por_kmax={r['truncado_por_kmax']}")
    assert r["k"] == 10, f"deberia haber procesado exactamente 10 eventos, proceso {r['k']}"
    assert r["truncado_por_kmax"], "deberia reportar truncado_por_kmax=True"
    hallazgos.append(dict(sweep="caso_limite", valor="KMAX=10", alcance_noc_medio=r["alcance_noc"],
                           ocupacion_media=r["ocupacion_moderadores_promedio"],
                           items_flageados_medio=r["items_flageados"],
                           espera_cola_media=r["espera_cola_promedio"], k_medio=r["k"],
                           nota="corte limpio confirmado, sin excepcion"))


def caso_cola_saturada(hallazgos):
    """1 solo moderador + escenario pesimista (recall=0.90, flagea casi todo
    lo nocivo) - el caso mas extremo de saturacion posible. Verifica que el
    motor sigue terminando (FEL se vacia o llega a KMAX) sin colgarse."""
    print("\n=== Caso limite: cola de moderacion saturada (1 moderador, escenario pesimista) ===")
    params = p.armar_params_custom("saturacion", base_escenario="pesimista", moderadores_por_turno=1)
    r = motor.correr_replica("saturacion", D=D_SWEEP, seed=456, params=params)
    print(f"K={r['k']}/{r['kmax']} (truncado={r['truncado_por_kmax']}), "
          f"ocupacion={r['ocupacion_moderadores_promedio']:.1%}, "
          f"items_encolados={r['items_encolados']}, "
          f"espera_cola_mediana={r['espera_cola_mediana']:,.0f} min")
    assert r["k"] <= r["kmax"]
    hallazgos.append(dict(sweep="caso_limite", valor="1 moderador + pesimista",
                           alcance_noc_medio=r["alcance_noc"],
                           ocupacion_media=r["ocupacion_moderadores_promedio"],
                           items_flageados_medio=r["items_flageados"],
                           espera_cola_media=r["espera_cola_promedio"], k_medio=r["k"],
                           nota="termina sin colgarse incluso en saturacion extrema"))


def main():
    os.makedirs(p.RESULTADOS_DIR, exist_ok=True)
    hallazgos = []

    sweep_vd1_moderadores(hallazgos)
    sweep_vd2_rate_limit(hallazgos)
    sweep_vd3_degradacion(hallazgos)
    caso_kmax_forzado(hallazgos)
    caso_cola_saturada(hallazgos)

    with open(ARCHIVO_SALIDA, "w", newline="") as f:
        columnas = list(hallazgos[0].keys())
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(hallazgos)

    print(f"\n{len(hallazgos)} puntos de barrido/casos limite, todos los invariantes OK.")
    print(f"Guardado en {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
