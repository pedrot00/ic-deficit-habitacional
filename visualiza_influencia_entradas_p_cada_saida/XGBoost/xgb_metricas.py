import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import optuna.visualization as vis

# Carregar os dados
dados = pd.read_csv("../../BDD/BD_MG_DEFICIT.csv")

# Definir colunas alvo
target_columns = ['DOMICILIOS_PRECARIOS', 'COABITACAO', 'ONUS_EXCESSIVO', 'ADENSAMENTO', 'DEFICIT_TOTAL']

# Remover linhas com valores ausentes nas colunas alvo
dados = dados.dropna(subset=target_columns)

# As entradas serão todas as outras colunas que não são saídas
feature_columns = [col for col in dados.columns if col not in target_columns]

# Separar X e y
X = dados[feature_columns]
y_all = dados[target_columns]

# Normalizar as features (IMPORTANTE: pode melhorar muito o desempenho!)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Configuração: escolha o método de otimização
USAR_OPTUNA = True  # True: usa Optuna (recomendado) | False: usa RandomizedSearchCV
N_TRIALS_OPTUNA = 100  # Número de tentativas do Optuna (quanto mais, melhor)
N_ITER_RANDOM = 50     # Número de iterações do RandomizedSearchCV

# Dicionários para armazenar resultados
importancias_dict = {}
metricas_dict = {}
melhores_params_dict = {}

# Loop para cada variável alvo
for target in target_columns:
    print(f"\n{'='*60}")
    print(f" Otimizando XGBoost para: {target}")
    print(f"{'='*60}\n")

    y_clean = y_all[target]

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_clean, test_size=0.2, random_state=42
    )

    # =========================================================================
    # MÉTODO 1: OPTUNA (Otimização Bayesiana - MAIS EFICIENTE)
    # =========================================================================
    if USAR_OPTUNA:
        print("🔍 Executando otimização com OPTUNA (Bayesian Optimization)...")
        
        def objective(trial):
            param = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            }
            model = xgb.XGBRegressor(**param, random_state=42, n_jobs=-1, objective='reg:squarederror')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)

        # Criar estudo Optuna (silencioso para não poluir o output)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)

        best_params = study.best_params
        print(f"\n✅ Melhores parâmetros encontrados (Optuna):")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"\n📊 Melhor R² durante otimização: {study.best_value:.4f}")

    # =========================================================================
    # MÉTODO 2: RANDOMIZEDSEARCHCV (Busca Aleatória)
    # =========================================================================
    else:
        print("🔍 Executando otimização com RANDOMIZEDSEARCHCV...")
        
        param_dist = {
            'max_depth': range(3, 8),
            'learning_rate': np.logspace(-2, -1, num=10),
            'n_estimators': range(100, 1500, 100),
            'min_child_weight': range(1, 7),
            'subsample': np.linspace(0.6, 1.0, num=5),
            'colsample_bytree': np.linspace(0.6, 1.0, num=5),
            'gamma': np.logspace(-8, 0, num=10),
            'reg_alpha': np.logspace(-5, 0, num=10),
            'reg_lambda': np.logspace(-5, 0, num=10),
        }

        model = xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=N_ITER_RANDOM,
            cv=5,
            scoring='r2',
            refit=True,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        
        print(f"\n✅ Melhores parâmetros encontrados (RandomizedSearchCV):")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"\n📊 Melhor R² durante validação cruzada: {random_search.best_score_:.4f}")

    # =========================================================================
    # TREINAR MODELO FINAL COM OS MELHORES PARÂMETROS
    # =========================================================================
    final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, objective='reg:squarederror')
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # Calcular métricas finais
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas do modelo final no conjunto de TESTE:")
    print(f"   MAE:  {mae:.3f}")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²:   {r2:.4f}")

    # Armazenar resultados
    metricas_dict[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    melhores_params_dict[target] = best_params

    # =========================================================================
    # IMPORTÂNCIA DAS FEATURES
    # =========================================================================
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importancias_dict[target] = feature_importance

    # Plot da importância (top 30)
    top_features = feature_importance.head(30)
    plt.figure(figsize=(10, 0.35 * len(top_features) + 1))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue', edgecolor='navy')
    plt.title(f'Top 30 Features Mais Importantes - {target}\nR²={r2:.4f} | MAE={mae:.1f} | RMSE={rmse:.1f}')
    plt.xlabel('Importância')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.gca().invert_yaxis()
    
    # Adicionar valores nas barras
    for i, v in enumerate(top_features['Importance']):
        plt.text(v, i, f'{v:.4f}', va='center', fontsize=8, ha='left', color='darkblue')
    
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # VISUALIZAÇÕES DO OPTUNA (se foi usado)
    # =========================================================================
    if USAR_OPTUNA:
        print("\n📈 Gerando visualizações do Optuna...")
        
        # Histórico de otimização
        fig1 = vis.plot_optimization_history(study)
        fig1.update_layout(title=f"Histórico de Otimização - {target}")
        fig1.show()
        
        # Importância dos hiperparâmetros
        fig2 = vis.plot_param_importances(study)
        fig2.update_layout(title=f"Importância dos Hiperparâmetros - {target}")
        fig2.show()

print(f"\n{'='*60}")
print("RESUMO FINAL DAS MÉTRICAS - XGBOOST OTIMIZADO")
print(f"{'='*60}\n")

# Criar DataFrame com as métricas
df_metricas = pd.DataFrame(metricas_dict).T
df_metricas = df_metricas.round(3)
print("XGBOOST OTIMIZADO:")
print(df_metricas.to_string())

print("\n\n" + "="*60)
print("COMPARAÇÃO COM RANDOM FOREST (baseline do código original):")
print("="*60)
print("                         MAE    RMSE     R²")
print("DOMICILIOS_PRECARIOS  13.519  31.039  0.152")
print("COABITACAO            30.687  45.125  0.944")
print("ONUS_EXCESSIVO        33.531  58.292  0.844")
print("ADENSAMENTO           10.090  15.059  0.487")
print("DEFICIT_TOTAL         54.106  85.387  0.938")

print("\n\n" + "="*60)
print("MELHORES HIPERPARÂMETROS ENCONTRADOS POR VARIÁVEL:")
print("="*60)
for target, params in melhores_params_dict.items():
    print(f"\n{target}:")
    for param, value in params.items():
        print(f"  {param}: {value}")