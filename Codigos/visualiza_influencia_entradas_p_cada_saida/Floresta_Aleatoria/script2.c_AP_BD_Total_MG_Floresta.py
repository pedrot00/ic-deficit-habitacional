# o seguinte teste visa analisar o impacto da quantidade de zeros na base de dados
# para isso é feito uma comparação entre medias
# de inicio pegamos a media e o desvio padrao de cada uma das saidas do banco BD_MG_DEFICIT
# em sequencia retiramos os zeros e pegamos a media e o desvio padrao dos valores para cada componente

# as medias visao mostrar a variavao da predicao do deficit quando o banco possui os zeros e quando nao possui
# o desvio padrao ajuda a identificar a dispersao das predicoes
# se a mudanca de desvio de um para outro for alto ent os zeros tem papel relevante no equilibrio da variabilidade
# se nao for alto significa que o equilibro da dispersao de valores se mantem

# ======================================
# impacto dos zeros nas previsoes - com medias e desvios padrao
# ======================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === 1. carregar base e definir saidas ===
dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")

saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
entradas = [col for col in dados.columns if col not in saidas]

# remove linhas com valores nulos
dados = dados.dropna(subset=entradas + saidas)

X = dados[entradas]
resultados = []

# === 2. treinar modelo e comparar previsoes ===
for saida in saidas:
    y = dados[saida]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # modelo random forest
    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    # previsoes com zeros
    y_pred_original = modelo.predict(X_test)

    # previsoes sem zeros (substitui zeros pela media global das nao-zero)
    X_test_sem_zeros = X_test.copy()
    media_global = X_test_sem_zeros[X_test_sem_zeros != 0].mean().mean()
    X_test_sem_zeros = X_test_sem_zeros.applymap(lambda x: media_global if x == 0 else x)
    y_pred_sem_zeros = modelo.predict(X_test_sem_zeros)

    # calcula medias e desvios
    media_original = np.mean(y_pred_original)
    media_sem_zeros = np.mean(y_pred_sem_zeros)
    desvio_original = np.std(y_pred_original)
    desvio_sem_zeros = np.std(y_pred_sem_zeros)
    diferenca = media_sem_zeros - media_original

    resultados.append({
        "Saida": saida,
        "Media original": media_original,
        "Desvio original": desvio_original,
        "Media sem zeros": media_sem_zeros,
        "Desvio sem zeros": desvio_sem_zeros,
        "Diferenca (sem zeros - original)": diferenca
    })

# === 3. exibir tabela de resultados ===
df_resultados = pd.DataFrame(resultados)
pd.set_option("display.float_format", "{:.3f}".format)

print("\nResultados comparando medias e desvios com e sem zeros:\n")
print(df_resultados.to_string(index=False))

# faz previsões
y_pred = modelo.predict(X_test)

# calcula métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Valores reais")
plt.ylabel("Valores preditos")
plt.title("Relação entre valores reais e preditos (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()


#      Saida      |Media original|Desvio original|Media sem zeros |Desvio sem zeros|  Diferenca (sem zeros - original)
#   DOMIC_PRECAR  |     14.689   |     12.516    |     50.580     |    27.186      |   35.891
#    COABITACAO   |    217.231   |     184.076   |    217.492     |   184.022      |    0.260
#  ONUS_EXCESSIVO |    132.706   |     136.476   |    133.805     |   136.406      |    1.098
#    ADENSAMENTO  |     16.221   |     14.400    |     18.981     |    15.126      |    2.760
#   DEFICIT_TOTAL |    381.483   |     330.100   |    383.532     |   327.702      |    2.049

# MAE: 53.708   -> indica que o modelo erra em medida 53,7 unidades sofre com outlines
# RMSE: 85.618  -> indica que o modelo tem dificuldade c valores extremos e la ele erra mais
# R²: 0.937     -> otimo, indica que o modelo esta aprendendo bem


# domicilios -> aumento expressivo da media, aumento do desvio -> indica que os zeros tem papel relevante no 
# equilibrio da variabilidade dos dados, ao remover o modelo superestima o componente e perde estabilidade

# coabitacao/onus/adensamento -> diferenca pequena em media e desvio -> zeros n impactam tanto na predicao
# deficit total -> leve reducao no desvio -> apesar dos efeitos em domicio precario n impac tanto na predicao



# CONCLUSOES
# 1.
# podemos concluir que a presença dos zeros possui uma importancia significativa para a predição de valores 
# feita pela floresta aleatória, especialmente o componente de domicilios precários, no entanto essa  
# diferença não tem um impacto geral relevante na precisão do deficit total quando tratando das áreas de  
# ponderação. Isso significa dizer que o modelo de aprendizado da floresta aleatoria não é afetado 
# drasticamente pela presença dos zeros, porém ao retira-los teriamos perda de precisão em outros  
# componentes que compõe o déficit total. O estudo do desvio padrão ajuda a identificar se a  
#  variabilidade/dispersao dos dados eh alterado quando esses zeros sao retirados, a tabela mostra que 
#  #esse impacto eh significativo dentro do componente de domicilios precarios, pois teriamos uma 
# distribuição de valores na média completamente desajustado, e isso impactaria tanto na predição quando a 
# interepretabilidade do componente. Portanto, não devemos eliminar os zeros da base de dados.

# 2.
# O modelo de Random Forest é muito robusto a presença  de zeros, valores faltantes (depois de preenchidos 
# com médias ou medianas), e escalas diferentes entre variáveis. Porém ele sofre bastante com outlines
# e sofre bastante com valores extremos, no entanto essa caracteristica de outline + valores extremos
# esta presente somente no componente de domicilios precarios
# esse defeito sera significativo nas areas de ponderacao que tem valores grandes, em setores censitarios
# o modelo vai cair como uma luva