"""Ajuste del GLM logístico de FLAG (moderation_flag ~ toxicity_index + topic_code).

Reproduce el fit citado en 3.3/3.4 y Tabla 6 de TP3_entrega_final.docx (parte b).
Fuente de datos: ../../parte a/Datos/muestra_cascadas_REDBAIRES_limpio.csv (66.021 filas).
Corrida original: 2026-10-06. Salida: glm_logit_flag.csv (coeficientes, std_err, p_value).
"""
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("../../parte a/Datos/muestra_cascadas_REDBAIRES_limpio.csv")

model = smf.logit("moderation_flag ~ toxicity_index + C(topic_code)", data=df).fit(disp=0)

coefs = model.params.rename("coef").to_frame()
coefs["std_err"] = model.bse
coefs["p_value"] = model.pvalues
coefs.index.name = "term"
coefs.to_csv("glm_logit_flag.csv")

print(coefs)
print("\nPseudo R2:", model.prsquared)
print("N:", model.nobs)
print("Tasa de flag observada (toda la muestra):", df.moderation_flag.mean())
