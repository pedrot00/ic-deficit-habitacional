# carrega a bdd BD_MG_DEFICIT conta a qtd de zeros em cada entrada e mostra o top 10 das que possuem mais 0
# bem como a porcentagem do quanto daqueles dados eh composto por zeros

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === ETAPA 1: Cálculo de zeros por variável ===
dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")

# considerar apenas colunas numéricas
dados_numericos = dados.select_dtypes(include='number')

# total de linhas
total_linhas = len(dados_numericos)

# contagem e porcentagem de zeros
zeros_por_coluna = (dados_numericos == 0).sum()
porcentagem_zeros = (zeros_por_coluna / total_linhas) * 100

# cria DataFrame resumo e ordena
df_zeros = pd.DataFrame({
    "Variável": dados_numericos.columns,
    "Qtd_Zeros": zeros_por_coluna.values,
    "Percentual_Zeros(%)": porcentagem_zeros.values
}).sort_values("Percentual_Zeros(%)", ascending=False).reset_index(drop=True)

# === ETAPA 2: Exibir e visualizar as 10 variáveis mais afetadas ===
top10 = df_zeros.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10,
    y="Variável",
    x="Percentual_Zeros(%)",
    palette="crest"
)

plt.title("Top 10 Variáveis com Maior Percentual de Zeros", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Percentual de Zeros (%)", fontsize=12, fontweight='bold')
plt.ylabel("Variável", fontsize=12, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
