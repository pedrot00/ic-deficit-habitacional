import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES OTIMIZADAS
CV_FOLDS = 5
N_TRIALS = 50  # Aumentei um pouco para compensar busca mais inteligente
TEST_SIZE = 0.2
RANDOM_STATE = 42

# CARREGA DADOS
dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")
dados = dados.dropna(subset=["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", 
                               "ADENSAMENTO", "DEFICIT_TOTAL"])

saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
entradas = [col for col in dados.columns if col not in saidas]
X = dados[entradas]

# ----------------------- NOVA FUNÇÃO OBJETIVO MULTI-MÉTRICA -----------------------
def objetivo_optuna_multimetric(trial, X_train, y_train, target_name):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # Aumentei range superior
        'max_depth': trial.suggest_categorical('max_depth', [None, 15, 25, 35, 50]),  # Mais opções
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7, 0.8]),  # Novas opções
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # Deixei optuna decidir
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    modelo = RandomForestRegressor(**params)
    
    # AVALIAÇÃO MULTI-MÉTRICA na validação cruzada
    scores_mae = cross_val_score(modelo, X_train, y_train, cv=CV_FOLDS, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    scores_r2 = cross_val_score(modelo, X_train, y_train, cv=CV_FOLDS, 
                               scoring='r2', n_jobs=-1)
    
    mae_cv = -scores_mae.mean()
    r2_cv = scores_r2.mean()
    
    # FUNÇÃO OBJETIVO: Minimizar MAE e Maximizar R²
    # Peso maior no R² pois você mencionou que é importante
    objetivo = mae_cv * 0.3 + (1 - r2_cv) * 0.7  # Queremos minimizar esta combinação
    
    # Guarda métricas para análise
    trial.set_user_attr('r2_cv', r2_cv)
    trial.set_user_attr('mae_cv', mae_cv)
    
    return objetivo

# ----------------------- OTIMIZAÇÃO POR TARGET -----------------------
def otimizar_modelo(y_col):
    print(f"\n🎯 Otimizando: {y_col}")
    
    y = dados[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    study = optuna.create_study(
        direction='minimize',  # Minimizamos nossa função objetivo combinada
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(
        lambda trial: objetivo_optuna_multimetric(trial, X_train, y_train, y_col),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    # ----------------------- ANÁLISE DO MELHOR MODELO -----------------------
    print(f"Melhor trial: {study.best_value:.4f}")
    print(f"Melhores parâmetros: {study.best_params}")
    
    # Treina modelo final com melhores parâmetros
    best_params = study.best_params.copy()
    best_params['n_jobs'] = -1
    best_params['random_state'] = RANDOM_STATE

    modelo_final = RandomForestRegressor(**best_params)
    modelo_final.fit(X_train, y_train)

    # Predições e métricas no TESTE
    y_pred = modelo_final.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # ----------------------- COMPARAÇÃO COM PADRÃO SKLEARN -----------------------
    modelo_padrao = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    modelo_padrao.fit(X_train, y_train)
    y_pred_padrao = modelo_padrao.predict(X_test)
    
    mae_padrao = mean_absolute_error(y_test, y_pred_padrao)
    r2_padrao = r2_score(y_test, y_pred_padrao)
    
    print(f"→ R²: {r2:.3f} (Padrão: {r2_padrao:.3f}) | {'✅' if r2 > r2_padrao else '❌'}")
    print(f"→ MAE: {mae:.3f} (Padrão: {mae_padrao:.3f}) | {'✅' if mae < mae_padrao else '❌'}")
    
    return modelo_final, {'MAE': mae, 'RMSE': rmse, 'R2': r2}, best_params

# ----------------------- EXECUÇÃO PRINCIPAL -----------------------
resultados = {}
melhores_parametros = {}

print("=" * 60)
print("🚀 INICIANDO OTIMIZAÇÃO PARA MELHORAR R², MAE E RMSE")
print("=" * 60)

for col in saidas:
    modelo, metricas, params = otimizar_modelo(col)
    resultados[col] = metricas
    melhores_parametros[col] = params
    print("-" * 50)

# ----------------------- RELATÓRIO FINAL -----------------------
df = pd.DataFrame(resultados).T

plt.figure(figsize=(12, 6))

# Subplot 1: R² e MAE
plt.subplot(1, 2, 1)
plt.plot(df.index, df["R2"], marker='o', linewidth=2, markersize=8, label='R²')
plt.plot(df.index, df["MAE"], marker='s', linewidth=2, markersize=8, label='MAE')
plt.title("R² e MAE por Variável Alvo")
plt.xlabel("Variável")
plt.ylabel("Valor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Subplot 2: RMSE
plt.subplot(1, 2, 2)
plt.bar(df.index, df["RMSE"], color='lightcoral', alpha=0.7)
plt.title("RMSE por Variável Alvo")
plt.xlabel("Variável")
plt.ylabel("RMSE")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("📊 RELATÓRIO FINAL - COMPARAÇÃO DAS MÉTRICAS")
print("=" * 60)
print(df.round(4))

# Salva resultados detalhados
df.to_csv("resultados_otimizacao_rf.csv")
print(f"\n💾 Resultados salvos em 'resultados_otimizacao_rf.csv'")

print("\n⭐ DICAS PARA MELHORIAS ADICIONAIS:")
print("• Se R² ainda baixo: Considere feature engineering")
print("• Se MAE/RMSE altos: Verifique outliers na variável alvo")  
print("• Para mais ganhos: Teste outros algoritmos (XGBoost, LightGBM)")