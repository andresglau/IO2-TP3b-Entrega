"""Costeo de la VR COSTO - RedBaires (TP3 parte B, IO2).

Traduce los contadores que devuelve una replica de motor.py a USD, siguiendo la formula
de 2.2 del .docx:

    COSTO = costo_dotacion + costo_clasificador + costo_falsos_positivos
            + costo_friccion + costo_degradacion

Dos partes bien separadas (y por que):
- `costo_fijo_escenario`: dotacion de moderadores + capex/opex del clasificador. Son
  costos de POLITICA del escenario, no dependen de lo que pase en una replica puntual -
  se calculan una sola vez por escenario (no por replica), igual que tp2 separa el costo
  de ordenar (fijo por pedido) del costo de almacenar (depende de la trayectoria
  simulada).
- `costo_variable_replica`: falsos positivos, friccion, degradacion. Dependen de los
  contadores que produjo ESA replica puntual (items flageados, impresiones recortadas,
  nodo-instancias degradadas) - se recalculan por cada una.

Parametros sociales NO incluidos aca (deliberado, no un olvido): costo_social_por_
impresion_nociva, valor_marca_por_usuario_activo y tasa_churn_por_exposicion_nociva NO
entran en COSTO - el propio 2.2 del .docx dice explicitamente que esos se usan despues,
en la sintesis de costo-beneficio final del informe (todavia no se escribe), para
traducir ALCANCE_NOC a un costo comparable - no son parte del VR COSTO del motor.

Decisiones de esta sesion (confirmadas con el usuario 2026-10-06, ver CLAUDE.md seccion
7 para el detalle completo):
- Prorrateo a D dias: dotacion x (D/30) [costo_moderador_por_hora/horas_operativas_mes
  son mensuales], capex/opex x (D/365) [son anuales].
- Amortizacion de capex: anualidad estandar con discount_rate (no linea recta - si no,
  discount_rate quedaria sin ningun uso en todo el modelo).
- Costo de degradacion (VD3): 1 cuenta-dia por cada nodo-instancia que sufrio el factor
  VD3 en la corrida (no hay identidad persistente de cuenta en el bootstrap i.i.d.).
"""

import parametros as p

DIAS_POR_MES = 30
DIAS_POR_ANIO = 365


def _anualidad(capex, discount_rate, vida_util_years):
    """Cuota anual que amortiza `capex` a lo largo de `vida_util_years` a una
    tasa `discount_rate` (formula estandar de anualidad/prestamo):

        cuota = capex * r / (1 - (1+r)^-n)

    Con r=0 (caso borde, no ocurre con los parametros actuales pero se cubre
    por las dudas) cae a linea recta (capex/n) para evitar division por cero."""
    if capex == 0:
        return 0.0
    if discount_rate == 0:
        return capex / vida_util_years
    r, n = discount_rate, vida_util_years
    return capex * r / (1 - (1 + r) ** (-n))


def costo_fijo_escenario(params, D):
    """Dotacion de moderadores + capex/opex del clasificador, prorateados a
    los D dias de la corrida. Se calcula UNA VEZ por escenario (no por
    replica) - 2.2 del .docx."""
    costos_base = params.costos_base

    dotacion_mensual = (params.moderadores_por_turno
                         * costos_base["costo_moderador_por_hora"]
                         * costos_base["horas_operativas_mes"])
    dotacion = dotacion_mensual * (D / DIAS_POR_MES)

    capex_anual = _anualidad(params.capex_clasificador,
                              costos_base["discount_rate"],
                              costos_base["vida_util_clasificador_years"])
    capex = capex_anual * (D / DIAS_POR_ANIO)
    opex = params.opex_anual_clasificador * (D / DIAS_POR_ANIO)

    return dotacion + capex + opex


def _falsos_positivos(items_flageados, precision_clasificador):
    """FP = items_flageados x (1-precision)/precision - despejado de la
    definicion de precision (precision=TP/(TP+FP)), tratando cada item
    flageado dentro de la cascada como un verdadero positivo (ver 2.2 del
    .docx para el razonamiento completo: toda la cascada es nociva por
    construccion, no hay contenido legitimo ahi para marcar por error)."""
    if precision_clasificador <= 0:
        return 0.0
    return items_flageados * (1 - precision_clasificador) / precision_clasificador


def costo_variable_replica(resultado, params):
    """Falsos positivos + friccion + degradacion - dependen de los contadores
    que produjo ESTA replica (ver motor.correr_replica). costo_revision_por_
    item_flag tambien es variable (depende de cuantos items se flagearon en
    esta corrida puntual), asi que vive aca y no en costo_fijo_escenario pese
    a que el resto de "Moderacion (dotacion humana)" si es fijo."""
    costos_base = params.costos_base
    items_flageados = resultado["items_flageados"]

    revision = costos_base["costo_revision_por_item_flag"] * items_flageados

    fp = _falsos_positivos(items_flageados, params.precision_clasificador)
    falsos_positivos = costos_base["costo_reputacional_por_falso_positivo"] * fp

    friccion = (costos_base["perdida_engagement_por_rate_limit"]
                * costos_base["valor_engagement_por_impresion"]
                * resultado["impresiones_recortadas"])

    degradacion = (costos_base["costo_degradar_influencer_por_dia"]
                   * resultado["nodo_instancias_degradadas"])

    return dict(
        revision_items=revision,
        falsos_positivos=falsos_positivos,
        friccion=friccion,
        degradacion=degradacion,
        total=revision + falsos_positivos + friccion + degradacion,
    )


def costo_total_replica(resultado, params, D):
    """COSTO completo de una replica: fijo del escenario (prorateado a D) +
    variable de esta corrida puntual. Es lo que se compara contra ALCANCE_NOC
    en el tradeoff costo-beneficio."""
    fijo = costo_fijo_escenario(params, D)
    variable = costo_variable_replica(resultado, params)
    return fijo + variable["total"]


if __name__ == "__main__":
    # Checkpoint 3 del plan: el costo fijo escala linealmente con D, y el
    # costo variable responde a los items flageados de una replica real.
    import motor

    params = p.armar_params_escenario("conservador")

    fijo_30 = costo_fijo_escenario(params, D=30)
    fijo_60 = costo_fijo_escenario(params, D=60)
    print(f"costo fijo D=30: ${fijo_30:,.2f}")
    print(f"costo fijo D=60: ${fijo_60:,.2f}")
    assert abs(fijo_60 / fijo_30 - 2.0) < 1e-9, "el costo fijo deberia duplicarse con D"
    print("OK - escala linealmente con D\n")

    r = motor.correr_replica("conservador", D=30, seed=42)
    variable = costo_variable_replica(r, params)
    print(f"items flageados en la replica: {r['items_flageados']}")
    print(f"costo variable: {variable}")
    assert variable["revision_items"] > 0, "deberia haber costo de revision si hubo flags"
    assert variable["total"] == sum(v for k, v in variable.items() if k != "total")
    print("OK - el costo variable responde a los contadores de la replica\n")

    total = costo_total_replica(r, params, D=30)
    print(f"COSTO total de la replica (escenario conservador, D=30): ${total:,.2f}")
