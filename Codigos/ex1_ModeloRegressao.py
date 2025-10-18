import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#carregamento dos dados
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
dados = pd.read_csv(url)

dados = dados.dropna() #remove valores nulos

#print(dados.shape)  #mostra quantidade(linhas, colunas)
#print(dados.info())  #mostra tipos das colunas e valores nulos
#print(dados.describe()) # estatisticas importante das variaveis, count, max, min, media, desvio, mediana, percentil de 50 e 75
#print(dados.head()) # mostra as 5 primeiras linhas


# CRIANDO TREINAMENTO 

#converte atributos categoricos em numericos -> se isso aq tiver comentado o modelo quebra se tiver strings, melhor usar
dados = pd.get_dummies(dados, drop_first=True) #drop_first evita redundancia, ne muito interessante n

#print(dados.head())

#separando entradas de saida, nesse caso estamos dropando esse atributo ai pois ele sera a saida, o resto entrada
X = dados.drop('median_house_value', axis = 1) #aqui sao as entradas (componentes) - chamaremos de Features
Y = dados['median_house_value']

#print(Y.head())
#print(X.describe())

#dividindo o treino e o teste, o train_test_split() divide os dados aleatoriamente em 80/20
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()     #usado para normalizar todas as features, media= 0, dp = 1
#nem sempre necessario mas faz diferença; se uma coluna varia de 1 a 10 e outra de 1 a 1 milhao, a de maior escala domina a otimização do modelo
#por isso normalizamos
#melhor usar do que não usar quando há dúvida sobre escalas muito diferentes, porque ajuda estabilidade e interpretação sem riscos.
#no caso a ideia é que sempre que eu tiver disparidades muito grandes entre colunas devemos utilizar isso para equilibrar otimização
#sempre usaremos em redes neurais

X_treino = scaler.fit_transform(X_treino) #fit calcula a media e dp, e transform aplica normalizacao usando esse valor
X_teste =  scaler.transform(X_teste) #aplica mesma transf, mas sem calc media e dp dnv, evita vazamento de dados isso ai


modelo = LinearRegression() #cria modelo de regressao
modelo.fit(X_treino, Y_treino) #treina o modelo passando como parametro o treino da entrada e da saida

Y_pred = modelo.predict(X_teste) #pega os coeficiente aprendido e aplica os dados de teste para gerar predicoes
#isso ai eh o q iremos comparar com os resultados verdadeiros

mae = mean_absolute_error(Y_teste, Y_pred) #MAE (Mean Absolute Error): média dos erros absolutos. Indica o quanto, em média, a previsão erra:
mse = mean_squared_error(Y_teste, Y_pred) #MSE (Mean Squared Error): média dos erros ao quadrado. Penaliza mais erros grandes
r2 = r2_score(Y_teste, Y_pred) #R2 (Coeficiente de determinação): mede quão bem o modelo explica a variação dos dados

#visualizacao padrao dos resultados
plt.scatter(Y_teste, Y_pred, alpha=0.5)
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.title("Relação entre Real e Previsto (Regressão Linear)")
plt.show()