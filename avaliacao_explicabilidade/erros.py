import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CARREGAMENTO E CONFIGURAÇÃO
# ==========================================
# Certifique-se de que o caminho do arquivo está correto
dados = pd.read_csv("../BDD/BD_MG_DEFICIT_COD_AP.csv")

# Colunas alvo
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]

# Limpeza: remove linhas com NaN nas saídas ou no código da AP
dados = dados.dropna(subset=saidas + ["AP"])

# Definição de Entradas (X)
# Removemos as saídas e a coluna 'AP' para não vazar a resposta
dados_indesejados = saidas + ["AP"]
entradas = [col for col in dados.columns if col not in dados_indesejados]
X = dados[entradas]

# Dicionário para armazenar os Top 10 Erros de cada saída para plotar no final
top_erros_por_saida = {}
metricas_dict = {}  

# ==========================================
# 2. FUNÇÃO DE TREINAMENTO
# ==========================================
def processar_modelo(y_col):
    print(f"Processando: {y_col}...")
    y = dados[y_col]

    # Divisão Treino/Teste
    # Importante: O índice é preservado para recuperarmos o código da AP depois
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo
    modelo = RandomForestRegressor(n_estimators=200, random_state=None, n_jobs=-1)
    modelo.fit(X_train, y_train)
    
    # Predição
    y_pred = modelo.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metricas_dict[y_col] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # --- CÁLCULO DOS RESÍDUOS ---
    # 1. Recupera códigos das APs do teste
    ids_ap_teste = dados.loc[X_test.index, 'AP']
    
    # 2. Calcula Erro Absoluto
    erros_abs = np.abs(y_test - y_pred)
    
    # 3. Cria DataFrame auxiliar
    df_erros = pd.DataFrame({
        'CODIGO_AP': ids_ap_teste.astype(str), # Converte para texto para o gráfico não achar que é número
        'ERRO_ABSOLUTO': erros_abs
    })
    
    # 4. Pega os Top 10 Maiores Erros
    df_top10 = df_erros.sort_values(by='ERRO_ABSOLUTO', ascending=False).head(10)
    
    return df_top10

# ==========================================
# 3. LOOP DE EXECUÇÃO
# ==========================================
print(f"{'='*60}")
print("INICIANDO TREINAMENTO E CÁLCULO DE ERROS")
print(f"{'='*60}")

for saida in saidas:
    top_10_df = processar_modelo(saida)
    top_erros_por_saida[saida] = top_10_df

# ==========================================
# 4. PLOTAGEM CONJUNTA (5 GRÁFICOS)
# ==========================================
print(f"\n{'='*60}")
print("GERANDO GRÁFICO CONJUNTO DOS ERROS...")
print(f"{'='*60}")

# Cria uma figura com 2 colunas e 3 linhas (total 6 espaços, usaremos 5)
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten() # Transforma a matriz 3x2 em uma lista linear para facilitar o loop

# Cores para cada gráfico ficar bonito
cores = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8F33']

for i, saida in enumerate(saidas):
    ax = axes[i]
    df_plot = top_erros_por_saida[saida]
    
    # Plota barras horizontais
    # Invertemos o .iloc[::-1] para o maior erro ficar no topo do gráfico
    sns.barplot(
        x='ERRO_ABSOLUTO', 
        y='CODIGO_AP', 
        data=df_plot, 
        ax=ax, 
        color=cores[i]
    )
    
    # Estilização
    ax.set_title(f"Top 10 Erros: {saida}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Erro Absoluto (Resíduo)")
    ax.set_ylabel("Código da AP")
    
    # Adiciona o valor do erro na frente da barra
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)

# Remove o 6º gráfico (vazio) pois só temos 5 saídas
fig.delaxes(axes[5])

plt.tight_layout()
plt.suptitle("Áreas de Ponderação com Maiores Erros de Predição por Variável", fontsize=16, y=1.02)
plt.show()

# ==========================================
# 5. RESUMO DE MÉTRICAS (OPCIONAL)
# ==========================================
print("\nRESUMO DAS MÉTRICAS GERAIS:")
print(pd.DataFrame(metricas_dict).T.round(3))