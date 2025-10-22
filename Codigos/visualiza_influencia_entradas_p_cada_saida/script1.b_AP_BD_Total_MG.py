# o codigo a seguir eh uma floresta aleatoria que cria 200 arvores em cada uma eh medido as entradas que 
# possuem maior influencia para as 5 saidas da base de dados BD_MG_DEFICIT, mostrando as 15 mais influentes
# para cada saida individualmente, no final eh feito uma ponderacao das 200 arvores p saber os componentes mais
# impactantes. a ordem que as avores pegam 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns


dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")

# remove qualquer linha que tenha NaN nas colunas de saída ou entrada
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"])
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]

# as entradas serão todas as outras colunas que não são saídas
entradas = [col for col in dados.columns if col not in saidas]
X = dados[entradas]

# função para treinar e plotar importâncias
def importancia_random_forest(y_col):
    print(f"\nTreinando modelo para saída: {y_col}")

    y = dados[y_col]

    # divide base em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # cria o modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    # rxtrai importâncias
    importancias = modelo.feature_importances_
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)

    # plota as 15 mais importantes
    plt.figure(figsize=(8, 5))
    sns.barplot(y="Variável", x="Importância", data=df_import.head(15), palette="viridis")
    plt.title(f"Importância das variáveis para {y_col}")
    plt.tight_layout()
    plt.show()

    return df_import

# treina para cada uma das 5 saídas
importancias_dict = {}
for saida in saidas:
    df_import = importancia_random_forest(saida)
    importancias_dict[saida] = df_import
