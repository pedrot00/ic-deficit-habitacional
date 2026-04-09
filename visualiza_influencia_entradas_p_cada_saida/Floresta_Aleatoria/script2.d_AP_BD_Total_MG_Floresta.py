import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# === CARREGA OS DADOS ===
dados = pd.read_csv("../../BDD/BD_MG_DEFICIT.csv")

# remove qualquer linha que tenha NaN nas colunas de saída ou entrada
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"])

# === DEFINE SAÍDAS E VARIÁVEIS A EXCLUIR ===
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
dados_indesejados = [
    "Soma_V154", "Soma_V160", "Soma_V041.x", "Soma_V113", "Soma_V206",
    "Soma_V039", "Soma_V119", "Soma_V131", "Soma_V149", "Soma_V052"
]

# === SELECIONA ENTRADAS ===
entradas = [col for col in dados.columns if col not in saidas + dados_indesejados]
X = dados[entradas]

# === MOSTRA INFORMAÇÕES DAS COLUNAS ===
print("\n============================================================")
print("✅ COLUNAS UTILIZADAS COMO ENTRADAS NO MODELO")
print("============================================================")
print(f"Total de colunas de entrada: {len(entradas)}")
print("Lista completa de colunas (em ordem alfabética):\n")
print(sorted(entradas))  # mostra todas as colunas
print("\n------------------------------------------------------------")
print("❌ Colunas removidas (dados indesejados):")
print(dados_indesejados)
print("------------------------------------------------------------\n")

# === FUNÇÃO PARA TREINAR E MOSTRAR IMPORTÂNCIAS ===
def importancia_random_forest(y_col):
    print(f"\n{'='*60}")
    print(f"Treinando modelo para saída: {y_col}")
    print(f"{'='*60}")

    y = dados[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=200, random_state=None)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas de Desempenho:")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.3f}")
    print()

    importancias = modelo.feature_importances_
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(y="Variável", x="Importância", data=df_import.head(15), palette="viridis")
    plt.title(f"Importância das variáveis para {y_col}\nR²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f}")
    plt.tight_layout()
    plt.show()

    return df_import, {"MAE": mae, "RMSE": rmse, "R2": r2}

# === TREINA PARA CADA SAÍDA ===
importancias_dict = {}
metricas_dict = {}

for saida in saidas:
    df_import, metricas = importancia_random_forest(saida)
    importancias_dict[saida] = df_import
    metricas_dict[saida] = metricas

# === RESUMO DAS MÉTRICAS ===
print(f"\n{'='*60}")
print("RESUMO DAS MÉTRICAS PARA TODAS AS SAÍDAS")
print(f"{'='*60}\n")

df_metricas = pd.DataFrame(metricas_dict).T
df_metricas = df_metricas.round(3)
print(df_metricas.to_string())
