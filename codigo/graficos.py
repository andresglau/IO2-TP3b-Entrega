"""Graficos - RedBaires (TP3 parte B, IO2).

Tres figuras, mismo patron que tp2/codigo/grafico_stock.py (usa devolver_trace=True
para sacar series dia-a-dia/evento-a-evento de una replica puntual, sin afectar la
reproducibilidad del resto del pipeline):

1. Ocupacion de moderadores en el tiempo (VE) - una replica representativa por
   escenario.
2. Tiempo de contencion por escenario - en escala log, porque la distribucion es
   heavy-tailed (ver stress_test.py y CLAUDE.md seccion 7 para el porque: un pool
   de moderadores saturado puede dejar items esperando muchisimo en cola, lo que
   domina el tiempo de "extincion" si no se mira junto con "primera remocion").
3. ALCANCE_NOC vs. COSTO por escenario - el grafico que de verdad sostiene la
   recomendacion de politica (trade-off entre las dos VR).

Uso: python3 graficos.py (guarda los 3 PNG en resultados/)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import costos
import motor
import parametros as p

D_DEMO = 30
REPLICAS_SCATTER = 20  # alcanza para un scatter ilustrativo, no es el Monte Carlo final
COLOR_ESCENARIO = {"conservador": "#4C72B0", "intermedio": "#DD8452", "pesimista": "#C44E52"}


def _guardar(fig, nombre):
    os.makedirs(p.RESULTADOS_DIR, exist_ok=True)
    ruta = os.path.join(p.RESULTADOS_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"guardado: {ruta}")


def grafico_ocupacion_moderadores():
    """VE de ocupacion: una replica con trace por escenario, step plot de
    moderadores ocupados en el tiempo. Actualizado en los eventos de toma/
    liberacion (motor.py), no por muestreo periodico - este grafico solo
    dibuja los puntos que YA calculo el motor, no recalcula nada."""
    fig, ejes = plt.subplots(len(p.ESCENARIOS), 1, figsize=(9, 8), sharex=False)
    for eje, escenario in zip(ejes, p.ESCENARIOS):
        params = p.armar_params_escenario(escenario)
        r = motor.correr_replica(escenario, D_DEMO, seed=1, devolver_trace=True)
        puntos = r["trace"]["ocupacion"]
        if puntos:
            t = np.array([x[0] for x in puntos]) / motor.MINUTOS_POR_DIA
            ocupados = np.array([x[1] for x in puntos])
            eje.step(t, ocupados / params.moderadores_por_turno, where="post",
                     color=COLOR_ESCENARIO[escenario])
        eje.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        eje.set_ylim(-0.05, 1.3)
        eje.set_ylabel("ocupacion")
        eje.set_title(f"{escenario} ({params.moderadores_por_turno} moderadores)")
    ejes[-1].set_xlabel("dias desde el inicio de la corrida")
    fig.suptitle("VE: ocupacion de moderadores en el tiempo (1 replica por escenario)")
    fig.tight_layout()
    _guardar(fig, "ve_ocupacion_moderadores.png")


def grafico_tiempo_contencion():
    """Distribucion de tiempo de contencion por escenario, en escala log.
    Dos paneles: extincion natural (dominada por espera en cola si el
    escenario esta saturado) y primera remocion exitosa (mas interpretable
    como "que tan rapido actua la moderacion cuando funciona")."""
    fig, (eje_ext, eje_rem) = plt.subplots(1, 2, figsize=(11, 5))
    datos_ext, datos_rem, etiquetas = [], [], []
    for escenario in p.ESCENARIOS:
        params = p.armar_params_escenario(escenario)
        r = motor.correr_replica(escenario, D_DEMO, seed=2, devolver_trace=True,
                                  params=params)
        ext = [max(x, 1e-6) for x in r["trace"]["tiempos_extincion"]]
        rem = r["trace"]["tiempos_primera_remocion"]
        datos_ext.append(ext)
        datos_rem.append(rem if rem else [0])
        etiquetas.append(escenario)

    for eje, datos, titulo in ((eje_ext, datos_ext, "Extincion natural de la cascada"),
                                (eje_rem, datos_rem, "Primera remocion exitosa")):
        eje.boxplot(datos, tick_labels=etiquetas, showfliers=False)
        eje.set_yscale("log")
        eje.set_ylabel("minutos (escala log)")
        eje.set_title(titulo)
    fig.suptitle("VE: tiempo de contencion por escenario\n"
                  "(extincion puede estar dominada por espera en cola si el escenario "
                  "esta saturado - ver stress_test.py)")
    fig.tight_layout()
    _guardar(fig, "ve_tiempo_contencion.png")


def grafico_tradeoff_alcance_costo():
    """ALCANCE_NOC vs. COSTO: un punto por replica (REPLICAS_SCATTER por
    escenario, corridas directo aca - no es el Monte Carlo con precision
    garantizada de monte_carlo.py, es solo para visualizar la nube de puntos
    y la media de cada escenario)."""
    fig, eje = plt.subplots(figsize=(8, 6))
    for escenario in p.ESCENARIOS:
        params = p.armar_params_escenario(escenario)
        alcances, costos_ = [], []
        for i in range(REPLICAS_SCATTER):
            r = motor.correr_replica(escenario, D_DEMO, seed=5_000 + i, params=params)
            alcances.append(r["alcance_noc"])
            costos_.append(costos.costo_total_replica(r, params, D_DEMO))
        color = COLOR_ESCENARIO[escenario]
        eje.scatter(costos_, alcances, alpha=0.4, color=color, label=None)
        eje.scatter(np.mean(costos_), np.mean(alcances), marker="X", s=200,
                   color=color, edgecolor="black", label=escenario, zorder=3)
    eje.set_xlabel("COSTO total (USD)")
    eje.set_ylabel("ALCANCE_NOC (impresiones)")
    eje.set_title(f"Tradeoff costo-beneficio por escenario (D={D_DEMO} dias, "
                 f"{REPLICAS_SCATTER} replicas c/u)")
    eje.legend(title="media por escenario (X)")
    fig.tight_layout()
    _guardar(fig, "tradeoff_alcance_costo.png")


def main():
    grafico_ocupacion_moderadores()
    grafico_tiempo_contencion()
    grafico_tradeoff_alcance_costo()


if __name__ == "__main__":
    main()
