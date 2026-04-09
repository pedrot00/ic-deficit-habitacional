import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES
N_TRIALS = 100  # Número de tentativas de otimização - numero de combinacoes dos parametros
CV_FOLDS = 3    # cross validation - do dataset de treino estaremos repartindo em 4 e fazendo N trials ali de cima
TEST_SIZE = 0.2 # tamanho dos dados para treino e teste (80/20)
RANDOM_STATE = 42 #setta a ordem de randomização, padrao para reprodutividade dos testes

# CARREGA E PREPARA DADOS
dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", 
                               "ADENSAMENTO", "DEFICIT_TOTAL"])

saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
entradas = [col for col in dados.columns if col not in saidas]
X = dados[entradas]

# FUNÇÃO OBJETIVO PARA OPTUNA
def objetivo_optuna(trial, X_train, y_train, target_name):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),    # num de arvores, sugerida entre 100 e 200
        'max_depth': trial.suggest_int('max_depth', None, 40),             # controle de complexidade da arvore (qtd. folhas) - 5 a 30
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), #num min de amostrar para dividir o no em 2, 
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 10),   #num min de moastrar necessarias para ser folha
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  #num de atributos entrada que cada tree pode olhar, forca arvores difrentes explorarem atributos diferentes
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]), #sorteia amostras para treino com reposicao, isto eh algumas amostras aparecem
        #na arvore e outras nao, aumenta SIGNIFICAMENTE a diversidade das trees impedindo qu tenhamos arvores iguais, sem boots trap quanto maior num de arvores maior numero de arvores identicas
        'random_state': RANDOM_STATE,
        'n_jobs': -1  # Usa todos os cores disponíveis
    }
    
    modelo = RandomForestRegressor(**params)
    
    # Usa cross-validation para métrica mais robusta
    # Negative MAE porque sklearn sempre maximiza, mas queremos minimizar MAE
    scores = cross_val_score(modelo, X_train, y_train, 
                            cv=CV_FOLDS, 
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1)
    
    return -scores.mean()  # Retorna MAE positivo (Optuna minimiza por padrão)


# OTIMIZAÇÃO PARA CADA TARGET
def otimizar_modelo(y_col, n_trials=N_TRIALS):
    """Otimiza hiperparâmetros para um target específico."""
    
    print(f"\n{'='*70}")
    print(f"🎯 OTIMIZANDO MODELO PARA: {y_col}")
    print(f"{'='*70}")
    
    y = dados[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                          random_state=RANDOM_STATE)
    
    # Cria estudo Optuna
    study = optuna.create_study(
        direction='minimize',  # Minimizar MAE
        study_name=f"RF_{y_col}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    # Executa otimização
    print(f"\n🔍 Iniciando {n_trials} trials de otimização...")
    study.optimize(
        lambda trial: objetivo_optuna(trial, X_train, y_train, y_col),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Melhores hiperparâmetros
    print(f"\n✨ MELHORES HIPERPARÂMETROS:")
    for param, valor in study.best_params.items():
        print(f"   {param}: {valor}")
    print(f"\n   MAE (CV): {study.best_value:.3f}")
    
    # Treina modelo final com melhores params
    best_params = study.best_params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    
    modelo_final = RandomForestRegressor(**best_params)
    modelo_final.fit(X_train, y_train)
    
    # Avalia no conjunto de teste
    y_pred = modelo_final.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 MÉTRICAS NO CONJUNTO DE TESTE:")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.3f}")
    
    # Extrai importâncias
    importancias = modelo_final.feature_importances_
    df_import = pd.DataFrame({"Variável": entradas, "Importância": importancias})
    df_import = df_import.sort_values("Importância", ascending=False)
    
    # Visualizações
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Importância das variáveis
    sns.barplot(y="Variável", x="Importância", data=df_import.head(15), 
                palette="viridis", ax=axes[0])
    axes[0].set_title(f"Top 15 Variáveis - {y_col}\nR²={r2:.3f} | MAE={mae:.1f}")
    
    
    plt.tight_layout()
    plt.show()
    
    return {
        'modelo': modelo_final,
        'study': study,
        'importancias': df_import,
        'metricas': {'MAE': mae, 'RMSE': rmse, 'R2': r2},
        'best_params': study.best_params
    }

# EXECUTA OTIMIZAÇÃO PARA TODOS OS TARGETS
resultados = {}
for saida in saidas:
    resultado = otimizar_modelo(saida, n_trials=N_TRIALS)
    resultados[saida] = resultado


# RESUMO COMPARATIVO
print(f"\n{'='*70}")
print("📈 RESUMO COMPARATIVO - MODELO OTIMIZADO vs BASELINE")
print(f"{'='*70}\n")

# Métricas otimizadas
df_metricas_otimizado = pd.DataFrame({
    saida: resultados[saida]['metricas'] 
    for saida in saidas
}).T

print("MÉTRICAS APÓS OTIMIZAÇÃO:")
print(df_metricas_otimizado.round(3).to_string())

# Melhores hiperparâmetros por target
print(f"\n{'='*70}")
print("🎛️  MELHORES HIPERPARÂMETROS POR TARGET")
print(f"{'='*70}\n")

for saida in saidas:
    print(f"\n{saida}:")
    for param, valor in resultados[saida]['best_params'].items():
        print(f"  {param}: {valor}")

# ANÁLISE DE IMPORTÂNCIA DOS HIPERPARÂMETROS
print(f"\n{'='*70}")
print("🔬 ANÁLISE: Quais hiperparâmetros mais impactam cada target?")
print(f"{'='*70}\n")

for saida in saidas:
    study = resultados[saida]['study']
    try:
        fig = plot_param_importances(study)
        fig.update_layout(title=f"Importância dos Hiperparâmetros - {saida}")
        fig.show()
    except:
        print(f"Não foi possível gerar gráfico de importância para {saida}")

print("\n✅ OTIMIZAÇÃO CONCLUÍDA!")
print("\n💡 DICAS:")
print("   - Se quiser mais precisão, aumente N_TRIALS (100→200)")
print("   - Se quiser mais robustez, aumente CV_FOLDS (3→5)")
print("   - Os modelos otimizados estão em resultados[target]['modelo']")