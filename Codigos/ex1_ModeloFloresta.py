import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# carregando diretamente os dados, mesmo bdd ex1_ModeloRegressao.py
dados = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")

dados = dados.dropna()     #limpeza eliminando os NaN (linhas val ausentes, 0 sao mantidos)
dados = pd.get_dummies(dados, drop_first=True) #converte dados categoricos para numericos
#no geral RandonFlorest lidam bem com dados categoricos, mas essa biblioteca n aceita string por isso converter
#isso n gera um impacto significativo

X = dados.drop('median_house_value', axis=1) #elimina a saida, sobra as entradas
Y = dados['median_house_value'] #seleciona saida que eh justamente o que foi eliminado acima

# quebrando etapa teste e treino em 80 p 20 (padrao)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)
          
# processo escalanomanete (OPCIONAL EM ARVORES!!) , ajuda a padronizar os dados                                           
scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)
# todo esse processo eh util para COMPARAR MODELOS
# quando varivaeis maginitudes mt diferentes auxilia analise visual e estabilidade
# NAO USAREMOS POIS ELE ELIMINARIA OS PESOS DOS COMPONENTES, nao daria para ver a influencia real


# criando e treinando modelo, repare que ate agora so muda essa linha em relacao a regressao
# note que o random_state =42 fixa uma ordem aleatoria -> tonra-se deterministica
# eh util fixar para comparar o resultado c outros modelos pois se rodar x vezes vai dar sempre o mesmo resultado
# no entanto eh interessante explorar de forma nao deterministica tmb
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_treino, Y_treino)  # aprende padroes <-> ajusta as arvores
#n_estimators -> quant. arvores, +arvore -> +estabilidade, +lento
#random_state -> fixa

# modelo.predict -> aplica o que aprendeu com os dados X_teste e devolve as previsoes para Y_pred
Y_pred = modelo.predict(X_teste) 

# avalia desempenho
mse = mean_squared_error(Y_teste, Y_pred) # MSE-> Erro Quad. Medio, mede erro medio ao quadrado, quanto menor melhor
r2 = r2_score(Y_teste, Y_pred) # R2 -> Coef. determinacao, mede qt model explica variacao dos dados, intevalo [0,1]


#VISUALIZACAO RESULTADOS - usando random Forest eh muito facil analisar cada componente
importancias = modelo.feature_importances_  #carrega importancia de cada feature p arvore (feature = componente)
nomes = X.columns

# organiza em ordem decrescente
indices = np.argsort(importancias)[::-1]

plt.figure(figsize=(8,8))
plt.title("Importancia das Variaveis (Random Forest)")
plt.bar(range(len(importancias)), importancias[indices], align="center")
plt.xticks(range(len(importancias)), [nomes[i] for i in indices], rotation=90)
plt.ylabel("Importancia relativa")
plt.show()

# PONTOS IMPORTANTES IMPLEMENTACAO BDD
#1.caso 0 seja ausencia real dos componentes -> valor legitimo, devemos manter, n faz mal
#2.caso 0 seja dados faltantes mascasros -> valores falsos, apagar para n enganar modelo
#3.caso 0 sejam concentrados em um determinado ponto 0 -> pode gerar viés, devemos tratar de alguma forma

# Arvores e florestas lidam bem mesmo com bastantes zeros, sera util utilizar

# Modelos especificos so para os zeros:
# se quant. de 0 forem extremas da p/ usar MODELO BINÁRIO, para prever "zero/n-zero" e outro modelo regressivo p valores n 0
# tecnica chamada Two-Stage Model ou Zero-Inflated Model -> PESQUISAR