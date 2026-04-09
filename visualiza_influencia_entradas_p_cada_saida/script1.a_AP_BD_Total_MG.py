# script1 etapa 1 - visualiza influência de todas as entradas para cada saída individualmente

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# carregar dados
dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")

# calcular a matriz de correlação
corr = dados.corr()

# exibir somente correlações das saídas com as entradas
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
corr_saidas = corr[saidas].drop(saidas)  # remove correlação entre as saídas

# VISUALIZAÇÃO COM SEABORN
# calcula tamanho automático com base na quantidade de entradas
altura = max(8, len(corr_saidas) * 0.4)
largura = max(10, len(saidas) * 1.2) 

estatisticas = pd.DataFrame({
    "Média": corr_saidas.mean(),
    "Desvio Padrão": corr_saidas.std()
})

print(estatisticas.round(3))

plt.figure(figsize=(largura, altura))
sns.heatmap(
    corr_saidas,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Correlação"}
)
plt.title("Correlação entre Entradas e Saídas", fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()