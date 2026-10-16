# TP3 Parte B — IO2 (RedBaires) — Motor de simulación

Repositorio de código, datos y resultados de la entrega de parte B: implementación,
calibración y experimentos del simulador de eventos discretos descripto en el Modelo
Conceptual de Parte A. El informe (lógicas/bloques, cambios vs. Parte A, cálculo de N
y resultados) se entrega por afuera de este repositorio, como documento aparte.

## Estructura

```
IO2-TP3b-Entrega/
├── README.md                 # este archivo
├── modelo_logico_simulador.drawio  # modelo lógico (flowchart), actualizado para
│                                     # coincidir exactamente con el motor final
│                                     # (ver tp3/parte b/CLAUDE.md sección 9)
├── codigo/                   # motor de eventos discretos (Python)
│   ├── parametros.py         # carga de datos, calibración, escenarios
│   ├── motor.py               # motor de eventos discretos (nucleo)
│   ├── costos.py              # calculo de la VR COSTO
│   ├── corrida_demo.py        # "condiciones iniciales" del informe (N, 1 corrida, IC)
│   ├── monte_carlo.py         # M réplicas por escenario (procedimiento de Law 2015)
│   ├── stress_test.py         # stress test y debugging (barridos de VD + casos límite)
│   └── graficos.py            # gráficos de Variables de Estado y tradeoff costo-beneficio
├── Datos/                     # parámetros, dataset limpio y coeficientes GLM calibrados
└── resultados/                # CSV/PNG generados por el código (reproducibles)
```

Paquete autocontenido: todo lo que el código necesita está en `Datos/` (no depende de
ninguna otra carpeta del repositorio de trabajo de la cátedra).

## Cómo correr

Desde `codigo/`, con Python 3 + numpy/pandas/scipy/matplotlib instalados:

```bash
python3 parametros.py    # chequeo de carga/calibración (opcional, standalone)
python3 motor.py         # 1 réplica suelta de prueba (opcional, standalone)
python3 costos.py        # chequeo de costeo (opcional, standalone)
python3 corrida_demo.py  # bloque "condiciones iniciales" del informe
python3 stress_test.py   # stress test (barridos VD1/VD2/VD3 + casos límite) -> resultados/
python3 graficos.py      # gráficos VE + tradeoff -> resultados/
python3 monte_carlo.py   # Monte Carlo (M réplicas, Law 2015) de los 3 escenarios -> resultados/
```

No hay un orden estricto entre los últimos 4 (cada uno es independiente), salvo que
todos requieren que `Datos/` esté poblado (ya lo está en este paquete).

## Alcance de esta entrega

Implementa y valida el motor (entidades, colas, recursos, VR, VE, stress test) para los
3 escenarios de política ya definidos (conservador/intermedio/pesimista). **No incluye**
la experimentación completa (Parameter Variation sobre todo el espacio de VD,
comparación estadística formal entre escenarios, conclusión y recomendación final) —
eso es alcance de parte C.
