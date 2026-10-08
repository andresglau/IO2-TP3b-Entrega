"""Monte Carlo definitivo - RedBaires (TP3 parte B, IO2).

Corre M replicas por escenario con el procedimiento SECUENCIAL de Law (2015) - a
diferencia de tp2/codigo/simulacion_montecarlo.py, que hace una sola pasada (piloto de
N0 fijo -> formula cerrada de N -> una corrida final), esto es genuinamente iterativo:

    1. Correr un lote inicial de M0 replicas.
    2. Calcular el half-width HW = Z_(1-alpha/2) * S / sqrt(M) de cada VR de interes
       (ALCANCE_NOC y COSTO).
    3. Si HW no baja del HW0 objetivo para AMBAS metricas, agregar DELTA_M replicas mas
       (reusando las ya corridas, no se descartan) y volver a 2.
    4. Repetir hasta cumplir con las dos precisiones a la vez, o hasta M_MAX (valvula de
       seguridad para no colgarse si el objetivo es irrealista).

HW0_ALCANCE/HW0_COSTO/NIVEL_CONFIANZA son PROVISORIOS (constantes de modulo, no un valor
final hardcodeado) - la eleccion definitiva de precision es una decision del informe,
todavia pendiente (ver tp3/parte b/CLAUDE.md seccion 3, punto 6: "no tiene sentido
elegirlo en el vacio, antes de ver la escala/dispersion real de los VR" - ahora que el
motor ya corre, se puede revisar con datos reales en mano al escribir el informe).

Uso: python3 monte_carlo.py (corre los 3 escenarios, guarda
resultados/resultados_montecarlo.csv)
"""

import csv
import os
import zlib

import numpy as np
from scipy import stats

import costos
import motor
import parametros as p

# Precision objetivo (half-width) - PROVISORIO, ver docstring del modulo.
HW0_ALCANCE = 1_000_000.0   # impresiones
HW0_COSTO = 2_000.0         # USD
NIVEL_CONFIANZA = 0.95

M0_PILOTO = 30
DELTA_M = 20
M_MAX = 2_000
SEMILLA_BASE = 3_000_000

ARCHIVO_SALIDA = os.path.join(p.RESULTADOS_DIR, "resultados_montecarlo.csv")


def _half_width(valores, confianza=NIVEL_CONFIANZA):
    """HW = Z_(1-alpha/2) * S / sqrt(M) - mismo criterio que la formula de la
    catedra citada en tp2 (foto "Calculo del tamano de la muestra"), aca
    aplicada de forma iterativa en vez de en una sola pasada."""
    n = len(valores)
    if n < 2:
        return float("inf")
    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    return z * valores.std(ddof=1) / np.sqrt(n)


def _semilla(escenario, D, m):
    """zlib.crc32 (no hash() built-in, que se randomiza entre procesos salvo
    PYTHONHASHSEED fijo) para que la semilla de cada replica sea siempre la
    misma dado (escenario, D, indice de replica) - mismo idioma que
    tp2/codigo/simulacion_montecarlo.py."""
    clave = f"{escenario}_{D}".encode()
    offset = zlib.crc32(clave) % 100_000
    return SEMILLA_BASE + offset + m


def correr_monte_carlo(escenario, D, hw0_alcance=HW0_ALCANCE, hw0_costo=HW0_COSTO,
                        confianza=NIVEL_CONFIANZA, m0=M0_PILOTO, delta_m=DELTA_M,
                        m_max=M_MAX):
    """Procedimiento secuencial de Law (2015) sobre ALCANCE_NOC y COSTO a la
    vez: agrega replicas hasta que AMBAS metricas cumplan su precision
    objetivo, o hasta m_max. Devuelve un dict con medias/IC de las dos
    metricas, mas M final y los half-width realmente alcanzados."""
    params = p.armar_params_escenario(escenario)
    muestras_alcance, muestras_costo = [], []
    m = 0
    hw_alcance = hw_costo = float("inf")

    while True:
        faltan = m0 if m == 0 else delta_m
        for _ in range(faltan):
            r = motor.correr_replica(escenario, D, seed=_semilla(escenario, D, m))
            muestras_alcance.append(r["alcance_noc"])
            muestras_costo.append(costos.costo_total_replica(r, params, D))
            m += 1

        arr_alcance = np.array(muestras_alcance)
        arr_costo = np.array(muestras_costo)
        hw_alcance = _half_width(arr_alcance, confianza)
        hw_costo = _half_width(arr_costo, confianza)

        if hw_alcance <= hw0_alcance and hw_costo <= hw0_costo:
            break
        if m >= m_max:
            print(f"ADVERTENCIA [{escenario}, D={D}]: no se alcanzo la precision "
                  f"objetivo (HW alcance={hw_alcance:,.0f} > {hw0_alcance:,.0f} o "
                  f"HW costo={hw_costo:,.2f} > {hw0_costo:,.2f}) - corte por M_MAX={m_max}.")
            break

    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    media_alcance, std_alcance = arr_alcance.mean(), arr_alcance.std(ddof=1)
    media_costo, std_costo = arr_costo.mean(), arr_costo.std(ddof=1)

    return dict(
        escenario=escenario, D=D, m=m,
        hw_alcance_objetivo=hw0_alcance, hw_alcance_final=hw_alcance,
        alcance_noc_media=media_alcance, alcance_noc_std=std_alcance,
        alcance_noc_ic_inf=media_alcance - z * std_alcance / np.sqrt(m),
        alcance_noc_ic_sup=media_alcance + z * std_alcance / np.sqrt(m),
        hw_costo_objetivo=hw0_costo, hw_costo_final=hw_costo,
        costo_media=media_costo, costo_std=std_costo,
        costo_ic_inf=media_costo - z * std_costo / np.sqrt(m),
        costo_ic_sup=media_costo + z * std_costo / np.sqrt(m),
    )


def main(D=30):
    os.makedirs(p.RESULTADOS_DIR, exist_ok=True)
    filas = [correr_monte_carlo(escenario, D) for escenario in p.ESCENARIOS]

    with open(ARCHIVO_SALIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"Precision objetivo (provisoria): HW0_alcance={HW0_ALCANCE:,.0f} impresiones, "
          f"HW0_costo=${HW0_COSTO:,.2f}, confianza={NIVEL_CONFIANZA:.0%}\n")
    print(f"{'Escenario':12}{'M':>6}{'ALCANCE_NOC (IC95%)':>34}{'COSTO (IC95%)':>26}")
    for fi in filas:
        ic_a = f"[{fi['alcance_noc_ic_inf']:,.0f}, {fi['alcance_noc_ic_sup']:,.0f}]"
        ic_c = f"[{fi['costo_ic_inf']:,.0f}, {fi['costo_ic_sup']:,.0f}]"
        print(f"{fi['escenario']:12}{fi['m']:6d}{ic_a:>34}{ic_c:>26}")
    print(f"\nGuardado en {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()
