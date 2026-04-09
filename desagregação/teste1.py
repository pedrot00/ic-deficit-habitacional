import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. Carregamento dos bancos
# ============================================================

df_ap = pd.read_csv("../BDD/BD_MG_DEFICIT_AP.csv")
df_sc = pd.read_csv("../BDD/BD_MG_SETORES1.csv")

# ============================================================
# 2. Definições
# ============================================================

COL_AP_AP = "AP"          # código da AP no banco AP
COL_AP_SC = "COD_AP"      # código da AP no banco SC
COL_SC = "COD_SC"
COL_TARGET = "DEFICIT_TOTAL"

SAIDAS = [
    "DOMICILIOS_PRECARIOS",
    "COABITACAO",
    "ONUS_EXCESSIVO",
    "ADENSAMENTO",
    "DEFICIT_TOTAL"
]
# Entradas = tudo menos AP e saída
FEATURES = [
    col for col in df_ap.columns
    if col not in (["AP"] + SAIDAS)
]


# ============================================================
# 3. Preparação do banco de AP (IGUAL ao seu padrão)
# ============================================================

# Remove APs sem DEFICIT_TOTAL (igual ao seu dropna)
df_ap_modelo = df_ap.dropna(subset=[COL_TARGET]).copy()

X = df_ap_modelo[FEATURES]
y = df_ap_modelo[COL_TARGET]

# ============================================================
# 4. Treinamento do modelo
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_train, y_train)

# ============================================================
# 5. Métricas (exatamente como você faz)
# ============================================================

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 MÉTRICAS – DEFICIT_TOTAL (AP)")
print(f"MAE :  {mae:.3f}")
print(f"RMSE:  {rmse:.3f}")
print(f"R²  :  {r2:.3f}")

# ============================================================
# 6. Predição em TODAS as APs + resíduo
# ============================================================

df_ap["PRED_AP"] = modelo.predict(df_ap[FEATURES])
df_ap["RESIDUO_ABS"] = np.abs(df_ap[COL_TARGET] - df_ap["PRED_AP"])

# Limiar de resíduo (90% piores erros)
LIMIAR_RESIDUO = df_ap["RESIDUO_ABS"].quantile(0.90)

# ============================================================
# 7. Predição BRUTA nos setores censitários
# ============================================================

df_sc["PRED_SC_BRUTO"] = modelo.predict(df_sc[FEATURES])

# ============================================================
# 8. Processo de propensão (desagregação condicional)
# ============================================================

resultados = []

for cod_ap, grupo_sc in df_sc.groupby(COL_AP_SC):

    linha_ap = df_ap[df_ap[COL_AP_AP] == cod_ap]

    if linha_ap.empty:
        continue

    linha_ap = linha_ap.iloc[0]

    valor_real = linha_ap[COL_TARGET]
    valor_pred = linha_ap["PRED_AP"]
    residuo = linha_ap["RESIDUO_ABS"]

    # -------------------------
    # Regra de decisão
    # -------------------------
    if not pd.isna(valor_real):
        valor_base = valor_real
        fonte = "real"

    else:
        if pd.isna(residuo) or residuo > LIMIAR_RESIDUO:
            continue
        valor_base = valor_pred
        fonte = "predito"

    soma_pred = grupo_sc["PRED_SC_BRUTO"].sum()

    if soma_pred <= 0:
        continue

    for _, row in grupo_sc.iterrows():

        peso = row["PRED_SC_BRUTO"] / soma_pred
        deficit_sc = valor_base * peso

        resultados.append({
            "COD_SC": row[COL_SC],
            "COD_AP": cod_ap,
            "DEFICIT_TOTAL_PROPENSO": deficit_sc,
            "PESO": peso,
            "FONTE_VALOR_AP": fonte,
            "RESIDUO_AP": residuo
        })

# ============================================================
# 9. Geração do CSV final
# ============================================================

df_resultado = pd.DataFrame(resultados)

df_resultado.to_csv(
    "../RESULTADOS/SC_DEFICIT_TOTAL_PROPENSAO.csv",
    index=False
)

print("\nCSV de propensão gerado com sucesso!")
# soma por AP dos setores
# ============================================================
# 10. AUDITORIA DE FECHAMENTO DA PROPENSÃO (AP x SC)
# ============================================================

print("\n🔍 AUDITORIA DE FECHAMENTO – PROPENSÃO POR AP\n")

auditoria = (
    df_resultado
    .groupby("COD_AP", as_index=False)
    .agg(
        SOMA_SC=("DEFICIT_TOTAL_PROPENSO", "sum")
    )
)

# Junta com o valor real do déficit da AP
auditoria = auditoria.merge(
    df_ap[[COL_AP_AP, COL_TARGET]],
    left_on="COD_AP",
    right_on=COL_AP_AP,
    how="left"
)

auditoria["DIFERENCA"] = auditoria["SOMA_SC"] - auditoria[COL_TARGET]

# Ordena pela maior diferença absoluta
auditoria["DIF_ABS"] = auditoria["DIFERENCA"].abs()
auditoria = auditoria.sort_values("DIF_ABS", ascending=False)

# Impressão linha a linha (bonita e legível)
for _, row in auditoria.iterrows():
    print(
        f"AP: {int(row['COD_AP'])} | "
        f"DEFICIT_AP: {row[COL_TARGET]:.6f} | "
        f"SOMA_SC: {row['SOMA_SC']:.6f} | "
        f"DIFERENCA: {row['DIFERENCA']:.6e}"
    )

print("\n📌 Diferença máxima absoluta:",
      auditoria["DIF_ABS"].max())