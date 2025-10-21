#script1 etapa 1 - visualiza influencia de todas as entradas p cada saida individualmente
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")

# calcular a matriz de correlação
corr = dados.corr()

# exibir somente corr das saidas com as entradas
saidas = ["DOMICILIOS_PRECARIOS","COABITACAO","ONUS_EXCESSIVO","ADENSAMENTO","DEFICIT_TOTAL"]
corr_saidas = corr[saidas].drop(saidas)  # remove correlação entre as saídas


# VISUALIZACAO
plt.figure(figsize=(20, 20))
plt.imshow(corr_saidas, cmap='coolwarm', aspect=0,1, vmin=-1, vmax=1)
plt.colorbar(label='Correlação', shrink=0.3)

plt.xticks(range(len(saidas)), saidas, rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(range(len(corr_saidas.index)), corr_saidas.index, fontsize=10)

plt.title("Correlação entre Entradas e Saídas", fontsize=16, fontweight='bold', pad=20)

# adicionar os valores de correlação no mapa com fonte menor
for i in range(len(corr_saidas.index)):
    for j in range(len(saidas)):
        valor = corr_saidas.iloc[i, j]
        # mudar cor do texto baseado no valor da correlação
        cor_texto = 'white' if abs(valor) > 0.5 else 'black'
        plt.text(j, i, f'{valor:.2f}', 
                ha='center', va='center', color=cor_texto, fontsize=8)
plt.tight_layout()
plt.show()