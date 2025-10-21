import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dados = pd.read_csv('https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv')

#etapa 1
#print(dados.describe())
#print(dados.info())
#print(dados.head())

dados = dados.dropna() #dropando null
dados = pd.get_dummies(dados, drop_first=True) #util ja que temos muita string no bdd

#a url desse banco n sera interessante no momento (n tem como definir saidas)
