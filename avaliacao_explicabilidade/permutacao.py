import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================
try:
    dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")
except FileNotFoundError:
    print("ERRO: Arquivo não encontrado. Verifique o caminho '../BDD/BD_MG_DEFICIT.csv'")
    # Cria dados fictícios apenas para o código não quebrar se você testar sem o arquivo
    dados = pd.DataFrame(np.random.rand(100, 15), columns=[f'Var_{i}' for i in range(10)] + ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"])

# Remove linhas com NaN nas saídas
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
dados = dados.dropna(subset=saidas)

# Define as entradas (todas as colunas menos as saídas e identificadores se houver)
# Importante: O Random Forest do Scikit-Learn só aceita números. Se tiver coluna de texto (Nome Cidade), remova.
entradas = [col for col in dados.columns if col not in saidas and dados[col].dtype in ['float64', 'int64']]
X = dados[entradas]

# ==========================================
# 2. FUNÇÃO PRINCIPAL DE TREINO E ANÁLISE
# ==========================================
def analisar_variavel(y_col):
    print(f"\n{'='*80}")
    print(f">>> ANALISANDO SAÍDA: {y_col}")
    print(f"{'='*80}")

    y = dados[y_col]

    # Divide base em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria e treina o modelo Random Forest
    # n_jobs=-1 usa todos os núcleos do processador para ser mais rápido
    modelo = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)

    # Faz previsões
    y_pred = modelo.predict(X_test)

    # Calcula métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Métricas de Desempenho:")
    print(f"   R²:   {r2:.3f} (Quanto mais perto de 1.0, melhor)")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")

    # ---------------------------------------------------------
    # ANÁLISE 1: FEATURE IMPORTANCE PADRÃO (GINI)
    # ---------------------------------------------------------
    importancias = modelo.feature_importances_
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(y="Variável", x="Importância", data=df_import.head(10), palette="viridis")
    plt.title(f"Importância Padrão (Gini) - {y_col}\n(O que a árvore usou para dividir os nós)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # ANÁLISE 2: PERMUTATION IMPORTANCE (PROVA DOS NOVE)
    # ---------------------------------------------------------
    print(f"   ...Calculando Permutation Importance (pode demorar um pouco)...")
    
    result = permutation_importance(
        modelo, 
        X_test,     
        y_test, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Organizando
    sorted_idx = result.importances_mean.argsort()
    # Pegamos apenas as top 15 para o gráfico não ficar gigante
    top_15_idx = sorted_idx[-15:] 

    plt.figure(figsize=(10, 6))
    plt.barh(X_test.columns[top_15_idx], result.importances_mean[top_15_idx], color='purple')
    plt.xlabel("Queda no R² (Importância)")
    plt.title(f"Permutation Importance - {y_col}\n(Se remover essa variável, o modelo piora X)")
    plt.tight_layout()
    plt.show()

    # Retorna métricas e importâncias para o resumo final
    metricas = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return df_import, metricas

# ==========================================
# 3. LOOP DE EXECUÇÃO
# ==========================================
metricas_dict = {}

for saida in saidas:
    # Chama a função para cada variável alvo
    _, metricas = analisar_variavel(saida)
    metricas_dict[saida] = metricas

# ==========================================
# 4. RESUMO FINAL
# ==========================================
print(f"\n{'='*60}")
print("TABELA RESUMO FINAL (Para copiar para o artigo)")
print(f"{'='*60}\n")

df_metricas = pd.DataFrame(metricas_dict).T
df_metricas = df_metricas.round(3)
print(df_metricas)