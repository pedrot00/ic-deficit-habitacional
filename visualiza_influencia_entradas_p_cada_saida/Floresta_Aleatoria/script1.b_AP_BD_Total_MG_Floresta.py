# o codigo a seguir eh uma floresta aleatoria que cria 200 arvores em cada uma eh medido as entradas que 
# possuem maior influencia para as 5 saidas da base de dados BD_MG_DEFICIT, mostrando as 15 mais influentes
# para cada saida individualmente, no final eh feito uma ponderacao das 200 arvores p saber os componentes mais
# impactantes. a ordem que as avores pegam 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns


dados = pd.read_csv("../../BDD/BD_MG_DEFICIT.csv")

# remove qualquer linha que tenha NaN nas colunas de saída ou entrada
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"])
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]

# as entradas serão todas as outras colunas que não são saídas
entradas = [col for col in dados.columns if col not in saidas]
X = dados[entradas]

# função para treinar e plotar importâncias
def importancia_random_forest(y_col):
    print(f"\n{'='*60}")
    print(f"Treinando modelo para saída: {y_col}")
    print(f"{'='*60}")

    y = dados[y_col]

    # divide base em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # cria o modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=200, random_state=None)
    modelo.fit(X_train, y_train)

    # faz previsões
    y_pred = modelo.predict(X_test)

    # calcula métricas de avaliação
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas de Desempenho:")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.3f}")
    print()

    # extrai importâncias
    importancias = modelo.feature_importances_
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)

    
    # plota as 15 mais importantes
    plt.figure(figsize=(8, 5))
    sns.barplot(y="Variável", x="Importância", data=df_import.head(15), palette="viridis")
    plt.title(f"Importância das variáveis para {y_col}\nR²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f}")
    plt.tight_layout()
    plt.show()

    return df_import, {"MAE": mae, "RMSE": rmse, "R2": r2}

# treina para cada uma das 5 saídas
importancias_dict = {}
metricas_dict = {}

for saida in saidas:
    df_import, metricas = importancia_random_forest(saida)
    importancias_dict[saida] = df_import
    metricas_dict[saida] = metricas

# resumo final das métricas
print(f"\n{'='*60}")
print("RESUMO DAS MÉTRICAS PARA TODAS AS SAÍDAS")
print(f"{'='*60}\n")

df_metricas = pd.DataFrame(metricas_dict).T
df_metricas = df_metricas.round(3)
print(df_metricas.to_string())

#                         MAE    RMSE     R2
# DOMICILIOS_PRECARIOS  13.519  31.039  0.152
# COABITACAO            30.687  45.125  0.944
# ONUS_EXCESSIVO        33.531  58.292  0.844
# ADENSAMENTO           10.090  15.059  0.487
# DEFICIT_TOTAL         54.106  85.387  0.938

# CONCLUSÃO
#
# Os resultados obtidos foram analisados através das métricas MAE, RMSE e R² que 
# verificam a qualidade do aprendizado e detectam overfitting.
#
# A floresta aleatória apresentou predição e interpretabilidade EXCELENTE para 
# déficit total (R²=0.937) e coabitação (R²=0.945), aprendizado MODERADO para 
# adensamento (R²=0.496), SUSPEITA DE DATA LEAKAGE em ônus excessivo (R²=0.837) 
# e predição PÉSSIMA para domicílios precários (R²=0.158).
#
# Analisando as causas:
#
# 1) ÔNUS EXCESSIVO: A variável Soma_V008.x domina com 80% de importância e 
# correlação de 0.93 com o target, indicando possível vazamento de informação 
# (data leakage) ao invés de aprendizado genuíno. Isso compromete a validade 
# do modelo para este componente.
#
# 2) COABITAÇÃO: Duas variáveis dominantes (Soma_V008.y com 44% e Soma_V011.y) 
# explicam bem o fenômeno. Apesar da alta correlação (0.964), o coeficiente de 
# variação da razão (48.9%) sugere que pode ser relação legítima, mas requer 
# investigação no dicionário do Censo para confirmar.
#
# 3) ADENSAMENTO: Performance moderada (R²=0.496) com importância distribuída 
# entre várias variáveis (Soma_V007.y lidera com 21%). O modelo captura parte 
# dos padrões, mas faltam features importantes para melhor precisão.
#
# 4) DOMICÍLIOS PRECÁRIOS: Péssimo desempenho (R²=0.158) causado pela alta 
# concentração de zeros na base, poucos valores moderados e alguns outliers 
# extremos. A importância muito dispersa (nenhuma variável domina) indica que 
# o modelo não consegue identificar padrões claros. Este é um fenômeno 
# genuinamente difícil de prever com as variáveis disponíveis.
#
# 5) DÉFICIT TOTAL: Excelente resultado com importância bem distribuída entre 
# múltiplas variáveis (maior tem 23%). Herda a previsibilidade dos componentes 
# bem modelados (coabitação e ônus).
#
# LIMITAÇÕES PARA SETORES CENSITÁRIOS:
#
# É importante ressaltar que os dados analisados são de áreas de ponderação de 
# Minas Gerais. Se aplicado a setores censitários, os resultados seriam 
# provavelmente piores por dois motivos principais:
#
# 1) EXCESSO DE ZEROS: Setores inteiros teriam múltiplas colunas zeradas, o que 
# é péssimo para Random Forest que não lida bem com dados extremamente 
# desbalanceados (70-90% de zeros esperados vs 40-60% atual).
#
# 2) VALORES PEQUENOS E DISCRETOS: Ao invés de prever valores como 0, 50, 200, 
# 4002 (áreas de ponderação), o modelo precisaria distinguir entre 0, 1, 2, 5, 
# 100 (setores censitários). Random Forest não é ideal para contagens pequenas 
# e discretas, pois:
#    - Prevê valores contínuos (ex: 2.3, 4.7) quando queremos inteiros (2, 5)
#    - MAE de 5 que é 1.3% de erro em AP vira 33% de erro em setor
#    - A variância total menor resulta em R² drasticamente reduzido
#
# RECOMENDAÇÕES FINAIS:
#
# 1) USAR Random Forest para áreas de ponderação focando em: déficit total, 
# coabitação e adensamento, onde o modelo demonstrou capacidade de aprendizado.
#
# 2) NÃO USAR para domicílios precários e investigar data leakage em ônus 
# excessivo antes de confiar nas predições (verificar V008.x no dicionário 
# do Censo).
#
# 3) NÃO USAR Random Forest para setores censitários. Alternativas recomendadas:
#    - Modelos de contagem (Zero-Inflated Poisson ou Negative Binomial) - ideais 
#      para dados com muitos zeros e valores pequenos
#    - Gradient Boosting com objective='poisson' (LightGBM/XGBoost) - mais 
#      robusto que Random Forest para contagens
#    - Modelo híbrido: classificador (zero vs não-zero) + regressor (valores 
#      positivos)
#
# 4) Antes de decidir o modelo final para setores, fazer teste simulando a 
# redução de escala (dividir valores atuais por 30-50) e retreinar para avaliar 
# degradação de performance.