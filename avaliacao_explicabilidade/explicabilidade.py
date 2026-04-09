"""
EXPERIMENTO: Análise de Importância de Variáveis e Explicabilidade com SHAP
Objetivo: Identificar as variáveis preditoras mais impactantes para cada uma das 5 dimensões
do déficit habitacional em Minas Gerais, utilizando Random Forest e SHAP values para
explicar as contribuições individuais das features nas previsões do modelo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import shap

# Carrega o dataset completo do déficit habitacional de MG
dados = pd.read_csv("../BDD/BD_MG_DEFICIT1.csv")

# Remove registros incompletos nas variáveis de resposta
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"])
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]

# Define automaticamente as variáveis preditoras (todas exceto as de saída)
entradas = [col for col in dados.columns if col not in saidas]
X = dados[entradas]

def importancia_random_forest(y_col):
    """
    Treina um modelo Random Forest para uma variável de resposta específica,
    calcula métricas de desempenho e aplica análise SHAP para explicabilidade.
    
    SHAP (SHapley Additive exPlanations): Método baseado na teoria dos jogos que
    atribui a cada variável uma importância para cada predição individual,
    mostrando tanto o impacto quanto a direção (positiva/negativa) do efeito.
    """
    print(f"\n{'='*60}")
    print(f"Treinando modelo para saída: {y_col}")
    print(f"{'='*60}")

    y = dados[y_col]

    # Divisão estratificada em conjuntos de treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configuração do Random Forest com 200 árvores (ensemble robusto)
    modelo = RandomForestRegressor(n_estimators=200, random_state=None)
    modelo.fit(X_train, y_train)

    # Previsões no conjunto de teste
    y_pred = modelo.predict(X_test)

    # Avaliação do modelo com métricas padrão
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas de Desempenho:")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.3f}")
    print()
    
    # ====================== ANÁLISE SHAP ======================
    print("Calculando SHAP values... isso pode levar alguns segundos.")

    # Explicador específico para modelos baseados em árvores (eficiente)
    # TreeExplainer calcula valores SHAP de forma otimizada para Random Forests
    explainer = shap.TreeExplainer(modelo)

    # Calcula valores SHAP para todas as instâncias do conjunto de teste
    # Cada valor SHAP representa a contribuição de uma variável para afastar
    # a previsão do valor base (expectativa do modelo)
    shap_values = explainer.shap_values(X_test)

    # 1. SHAP Summary Plot (beeswarm) - VISUALIZAÇÃO PRINCIPAL
    # Mostra a distribuição do impacto de cada variável em todas as predições
    # Posição no eixo X = valor SHAP (impacto na predição)
    # Cor = valor real da variável (alto/baixo)
    # Sobreposição de pontos mostra densidade de observações
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot – {y_col}")
    plt.tight_layout()
    plt.show()

    # 2. SHAP Bar Plot - IMPORTÂNCIA MÉDIA ABSOLUTA
    # Ordena variáveis pela média dos valores SHAP absolutos
    # (quantifica impacto geral ignorando direção positiva/negativa)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP – Importância média das variáveis ({y_col})")
    plt.tight_layout()
    plt.show()

    # Importância tradicional do Random Forest (baseada na redução de impureza)
    # Complementar ao SHAP, mas menos explicativa em nível individual
    importancias = modelo.feature_importances_
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)

    # Visualização das 15 variáveis mais importantes (método tradicional)
    plt.figure(figsize=(8, 5))
    sns.barplot(y="Variável", x="Importância", data=df_import.head(15), palette="viridis")
    plt.title(f"Importância das variáveis para {y_col}\nR²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f}")
    plt.tight_layout()
    plt.show()

    return df_import, {"MAE": mae, "RMSE": rmse, "R2": r2}

# Executa a análise para cada uma das 5 dimensões do déficit
importancias_dict = {}
metricas_dict = {}

for saida in saidas:
    df_import, metricas = importancia_random_forest(saida)
    importancias_dict[saida] = df_import
    metricas_dict[saida] = metricas

# Consolidação final das métricas de desempenho
print(f"\n{'='*60}")
print("RESUMO DAS MÉTRICAS PARA TODAS AS SAÍDAS")
print(f"{'='*60}\n")

df_metricas = pd.DataFrame(metricas_dict).T
df_metricas = df_metricas.round(3)
print(df_metricas.to_string())