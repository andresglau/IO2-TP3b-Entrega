"""Corrida de demostracion ("condiciones iniciales") - RedBaires (TP3 parte B, IO2).

Deja lista, a nivel de codigo, la corrida que pide el bloque obligatorio del informe
(rubrica, seccion "Informe"): "elegir un set de VD inicial, calcular N, elegir cantidad
de dias a simular, mostrar resultados de 1 corrida, y mostrar resultados con intervalo
de confianza (multiples replicas)". Este script no ES el informe (el usuario pidio
explicitamente no escribirlo todavia) - solo corre y muestra los numeros para que el
informe los pueda citar cuando se escriba.

Uso: python3 corrida_demo.py
"""

import costos
import monte_carlo
import motor
import parametros as p

ESCENARIO_DEMO = "conservador"  # set de VD inicial elegido para esta demo
D_DEMO = 30                      # dias a simular
SEED_DEMO = 42
M_MAX_DEMO = 100                 # tope corto para que la demo termine rapido -
                                  # el Monte Carlo "de verdad" con precision objetivo
                                  # completa es monte_carlo.py


def main():
    params = p.armar_params_escenario(ESCENARIO_DEMO)

    print("=" * 70)
    print(f"Condiciones iniciales: escenario={ESCENARIO_DEMO!r}, D={D_DEMO} dias")
    print("=" * 70)
    print(f"VD1 moderadores_por_turno = {params.moderadores_por_turno}")
    print(f"VD1 precision/recall clasificador = {params.precision_clasificador}/"
          f"{params.recall_clasificador}")
    print(f"VD2 umbral/reduccion rate-limit = {params.umbral_reenvios_rate_limit}/"
          f"{params.reduccion_alcance_rate_limit:.0%}")
    print(f"VD3 segmento/intensidad degradacion = "
          f"{sorted(params.segmento_degradado) or 'ninguno'}/"
          f"{params.intensidad_degradacion:.0%}")
    print(f"delta_flag calibrado para este escenario = {params.delta_flag:.4f}")

    print(f"\n--- 1 corrida (seed={SEED_DEMO}) ---")
    r = motor.correr_replica(ESCENARIO_DEMO, D_DEMO, seed=SEED_DEMO)
    print(f"N (cascadas semilla sorteadas) = {r['n_semillas']} "
          f"(esperado ~ lambda*D = {p.LAMBDA_CASCADAS_POR_DIA}*{D_DEMO} = "
          f"{p.LAMBDA_CASCADAS_POR_DIA * D_DEMO:.1f})")
    print(f"K (eventos procesados) = {r['k']} / KMAX = {r['kmax']} "
          f"(truncado={r['truncado_por_kmax']})")
    print(f"ALCANCE_NOC = {r['alcance_noc']:,.0f} impresiones")
    costo = costos.costo_total_replica(r, params, D_DEMO)
    print(f"COSTO = ${costo:,.2f}")
    print(f"ocupacion promedio de moderadores = {r['ocupacion_moderadores_promedio']:.1%}")

    print(f"\n--- Monte Carlo corto (M<= {M_MAX_DEMO}, procedimiento secuencial de "
          f"Law 2015) ---")
    resumen = monte_carlo.correr_monte_carlo(ESCENARIO_DEMO, D_DEMO, m_max=M_MAX_DEMO)
    print(f"M replicas corridas = {resumen['m']}")
    print(f"ALCANCE_NOC: media={resumen['alcance_noc_media']:,.0f}, "
          f"IC95%=[{resumen['alcance_noc_ic_inf']:,.0f}, "
          f"{resumen['alcance_noc_ic_sup']:,.0f}] "
          f"(HW={resumen['hw_alcance_final']:,.0f}, objetivo={resumen['hw_alcance_objetivo']:,.0f})")
    print(f"COSTO: media=${resumen['costo_media']:,.2f}, "
          f"IC95%=[${resumen['costo_ic_inf']:,.2f}, ${resumen['costo_ic_sup']:,.2f}] "
          f"(HW=${resumen['hw_costo_final']:,.2f}, objetivo=${resumen['hw_costo_objetivo']:,.2f})")


if __name__ == "__main__":
    main()
