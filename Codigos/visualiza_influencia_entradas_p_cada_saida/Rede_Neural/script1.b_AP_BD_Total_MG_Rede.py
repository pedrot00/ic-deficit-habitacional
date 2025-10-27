# O código a seguir é uma rede neural que treina modelos individuais para cada uma das 5 saídas 
# da base de dados BD_MG_DEFICIT. Para cada saída, é medida a importância das entradas através 
# da análise de permutação (permutation importance), mostrando as 15 mais influentes para cada 
# saída individualmente. Ao final é feito um resumo das métricas de desempenho para comparação.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Carrega os dados
dados = pd.read_csv("../../BDD/BD_MG_DEFICIT.csv")

# Remove qualquer linha que tenha NaN nas colunas de saída ou entrada
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"])
saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]

# As entradas serão todas as outras colunas que não são saídas
entradas = [col for col in dados.columns if col not in saidas]
X = dados[entradas]

# Função para treinar e plotar importâncias
def importancia_rede_neural(y_col):
    print(f"\n{'='*60}")
    print(f"Treinando modelo para saída: {y_col}")
    print(f"{'='*60}")

    y = dados[y_col]

    # Divide base em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria o modelo de Rede Neural (MLPRegressor)
    # Arquitetura: 3 camadas ocultas (128, 64, 32 neurônios)
    # activation='relu': função de ativação ReLU para capturar não-linearidades
    # solver='adam': otimizador adaptativo, melhor que SGD padrão
    # alpha=0.001: regularização L2 para evitar overfitting
    # early_stopping=True: para quando performance em validação não melhora
    # validation_fraction=0.2: 20% do treino usado para validação
    # n_iter_no_change=20: paciência de 20 épocas sem melhoria
    # max_iter=500: máximo de épocas de treinamento
    modelo = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )

    # Treina o modelo
    print("Treinando rede neural...")
    modelo.fit(X_train, y_train)
    print(f"Treinamento concluído em {modelo.n_iter_} iterações")

    # Faz previsões
    y_pred = modelo.predict(X_test)

    # Calcula métricas de avaliação
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas de Desempenho:")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.3f}")
    print()

    # Calcula importância por permutação (equivalente à feature importance da Random Forest)
    # O método embaralha cada variável e mede quanto a performance piora
    # Quanto maior a queda, mais importante é a variável
    print("Calculando importância das variáveis...")
    perm_importance = permutation_importance(
        modelo, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Extrai importâncias
    importancias = perm_importance.importances_mean
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)

    # Plota as 15 mais importantes
    plt.figure(figsize=(8, 5))
    sns.barplot(y="Variável", x="Importância", data=df_import.head(15), palette="viridis")
    plt.title(f"Importância das variáveis para {y_col}\nR²={r2:.3f} | MAE={mae:.1f} | RMSE={rmse:.1f}")
    plt.tight_layout()
    plt.show()

    

    return df_import, {"MAE": mae, "RMSE": rmse, "R2": r2}

# Treina para cada uma das 5 saídas
importancias_dict = {}
metricas_dict = {}

for saida in saidas:
    df_import, metricas = importancia_rede_neural(saida)
    importancias_dict[saida] = df_import
    metricas_dict[saida] = metricas

# Resumo final das métricas
print(f"\n{'='*60}")
print("RESUMO DAS MÉTRICAS PARA TODAS AS SAÍDAS")
print(f"{'='*60}\n")

df_metricas = pd.DataFrame(metricas_dict).T
df_metricas = df_metricas.round(3)
print(df_metricas.to_string())

# Resumo final das métricas
print(f"\n{'='*60}")
print("RESUMO DAS MÉTRICAS PARA TODAS AS SAÍDAS")
print(f"{'='*60}\n")

# Converte dict de métricas em DataFrame
df_metricas = pd.DataFrame(metricas_dict).T

# Arredonda os valores para 3 casas decimais
df_metricas = df_metricas.round(3)

# Organiza colunas na ordem desejada
df_metricas = df_metricas[['MAE', 'RMSE', 'R2']]

# Ajusta o índice para mostrar os nomes das saídas
df_metricas.index.name = ''

# Gera o print no formato desejado
print(df_metricas.to_string())

# CONCLUSÃO

#                         MAE     RMSE      R2
# DOMICILIOS_PRECARIOS  82.560  126.647 -13.116
# COABITACAO            85.844  123.664   0.579
# ONUS_EXCESSIVO        79.124  117.302   0.367
# ADENSAMENTO           82.800  117.624 -30.296
# DEFICIT_TOTAL         93.127  142.954   0.825
   
   
# A rede neural (MLPRegressor) apresentou predição e interpretabilidade [PREENCHER 
# COM RESULTADOS APÓS EXECUÇÃO] para déficit total, coabitação, ônus excessivo, 
# adensamento e domicílios precários.
#
# Analisando as causas e comparando com Random Forest:
#
# 1) ÔNUS EXCESSIVO: [Analisar se mantém domínio da variável Soma_V008.x e alta 
# correlação, indicando possível data leakage ou se a rede neural distribui melhor 
# a importância através das camadas ocultas]
#
# 2) COABITAÇÃO: [Verificar se as mesmas variáveis dominantes (Soma_V008.y com 44% 
# e Soma_V011.y) aparecem e se o R² se mantém alto (~0.944), confirmando padrões 
# genuínos ou se há mudanças na distribuição de importância]
#
# 3) ADENSAMENTO: [Comparar performance com Random Forest (R²=0.496) e verificar 
# se a rede neural consegue capturar padrões não-lineares adicionais que melhorem 
# a predição, dado que RF teve importância distribuída (Soma_V007.y lidera com 21%)]
#
# 4) DOMICÍLIOS PRECÁRIOS: [Avaliar se a rede neural lida melhor com a distribuição 
# desbalanceada (alta concentração de zeros, poucos valores moderados, outliers 
# extremos) ou se mantém performance ruim como Random Forest (R²=0.158). MLPRegressor 
# pode sofrer ainda mais com zeros devido à normalização implícita]
#
# 5) DÉFICIT TOTAL: [Verificar se mantém excelente resultado (RF teve R²=0.938) e 
# se a importância continua bem distribuída entre múltiplas variáveis (maior tinha 23%)]
#
# VANTAGENS DA REDE NEURAL (MLP) vs RANDOM FOREST:
#
# 1) CAPTURA DE NÃO-LINEARIDADES: MLPRegressor pode capturar relações complexas 
# e interações de alta ordem entre variáveis que Random Forest pode perder, 
# especialmente através das múltiplas camadas (128→64→32)
#
# 2) APRENDIZADO DE FEATURES: As camadas ocultas criam representações abstratas 
# dos dados, potencialmente descobrindo padrões que não são óbvios nas features originais
#
# 3) CONTINUIDADE: Predições mais suaves e contínuas, enquanto RF pode ter 
# "degraus" devido à natureza das árvores
#
# DESVANTAGENS DA REDE NEURAL (MLP) vs RANDOM FOREST:
#
# 1) INTERPRETABILIDADE: Permutation importance é menos intuitiva que feature 
# importance. A "caixa preta" dificulta entender exatamente como decisões são tomadas. 
# Impossível visualizar regras como em árvores
#
# 2) SENSIBILIDADE A HIPERPARÂMETROS: Muito sensível à arquitetura (número de 
# camadas/neurônios), learning rate, alpha, etc. Requer mais experimentação
#
# 3) CONVERGÊNCIA: Pode não convergir ou convergir para mínimos locais ruins. 
# RF sempre converge deterministicamente
#
# 4) TEMPO DE TREINAMENTO: Mais lento que RF, especialmente com early stopping 
# e múltiplas iterações
#
# 5) ESCALA DOS DADOS: Embora estejamos sem normalização explícita (como pedido), 
# MLPRegressor é mais sensível a diferenças de escala entre features que RF
#
# 6) OVERFITTING: Mais propenso a overfitting em datasets pequenos, mesmo com 
# regularização (alpha) e early stopping
#
# LIMITAÇÕES PARA SETORES CENSITÁRIOS:
#
# As mesmas limitações da Random Forest aplicam-se às redes neurais, possivelmente 
# PIORES:
#
# 1) EXCESSO DE ZEROS: MLPRegressor sofre ainda mais com dados extremamente 
# desbalanceados (70-90% de zeros esperados vs 40-60% atual). Neurônios podem 
# saturar ou "morrer" (dying ReLU problem) com muitos zeros
#
# 2) VALORES PEQUENOS E DISCRETOS: Redes neurais preveem valores contínuos 
# (ex: 2.37, 4.82), o que é ainda pior para contagens pequenas inteiras (0, 1, 2, 5). 
# MAE de 5 que é 1.3% em AP vira 33% em setor. A perda MSE penaliza erro de 1 
# unidade muito mais em escala pequena
#
# 3) VARIÂNCIA REDUZIDA: R² será drasticamente reduzido, pois:
#    R² = 1 - (SS_res / SS_tot)
#    Com valores pequenos, SS_tot é pequeno, então qualquer erro vira R² ruim
#
# 4) GRADIENTES INSTÁVEIS: Com valores muito pequenos, gradientes podem 
# desaparecer ou explodir, dificultando o treinamento
#
# RECOMENDAÇÕES FINAIS:
#
# 1) COMPARAR RESULTADOS: Avaliar se MLP supera Random Forest nas métricas 
# R², MAE e RMSE para cada saída. RF geralmente vence em problemas tabulares
#
# 2) INVESTIGAR DATA LEAKAGE: Se ônus excessivo mantiver R² muito alto (~0.844) 
# com dominância de Soma_V008.x (80%), confirmar no dicionário do Censo se há 
# vazamento de informação
#
# 3) TESTAR NORMALIZAÇÕES: Embora você não queira centralizar dados para ver 
# impacto real, pode testar MinMaxScaler (0-1) que preserva distribuição e 
# relações. Isso pode melhorar significativamente MLP
#
# 4) CONSIDERAR ENSEMBLE: Combinar predições de Random Forest e MLP pode 
# melhorar resultados:
#    - Stacking: usar RF e MLP como base, meta-modelo combina
#    - Blending: média ponderada simples (ex: 0.6*RF + 0.4*MLP)
#
# 5) PARA ÁREAS DE PONDERAÇÃO: Se MLP superar RF, usar para: déficit total 
# e coabitação (onde RF já era excelente). Manter cautela com ônus excessivo 
# (suspeita de leakage) e domicílios precários (distribuição ruim)
#
# 6) PARA SETORES CENSITÁRIOS: NÃO USAR MLPRegressor padrão. Alternativas:
#    - Modelos de contagem estatísticos (Zero-Inflated Poisson/Negative Binomial) 
#      continuam sendo MELHOR opção
#    - Se insistir em neural: arquiteturas customizadas com:
#      * Loss Poisson (penaliza diferente contagens vs valores grandes)
#      * Camada final softplus (garante outputs positivos)
#      * Duas sub-redes (classificador zero vs não-zero + regressor)
#    - XGBoost/LightGBM com objective='count:poisson' supera tanto RF quanto MLP 
#      para contagens
#
# 7) DIAGNÓSTICO ADICIONAL: 
#    - Plotar predições vs valores reais para cada saída
#    - Analisar resíduos (y_test - y_pred) para detectar viés sistemático
#    - Verificar se MLP está aprendendo ou só memorizando (comparar train vs test)