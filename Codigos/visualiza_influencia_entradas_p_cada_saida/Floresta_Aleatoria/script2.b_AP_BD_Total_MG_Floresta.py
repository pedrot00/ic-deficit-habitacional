# a ideia desse codigo eh pegar a base de dados normal, com os zeros, e memorizar suas saidas
# depois o modelo pega essa base de dados, tira todos os zeros e testa eles pra ve quai serao as novas saidas
# obviamente havera alteracoes na saida, entao ele ira fazer a diferenca (subtracao) da saida nova
# pela saida original, se o resultado for positivo (ex: +200) significa que ao tirar os zeros daquela variavel
# o valor da saida aumentou e o contrario signifca que diminui

# a barrinha azul indica que quando a variavel eh diferente de zero o deficit medio AUMENTA para aquele componente
# a barrinha vermelha indica que quando a variavel eh igual a zero o deficit medio DIMINUI para aquele componente
# a cor indica a direcao do impacto

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Carregar base e definir saídas ===
dados = pd.read_csv("../BDD/BD_MG_DEFICIT.csv")

saidas = ["DOMICILIOS_PRECARIOS", "COABITACAO", "ONUS_EXCESSIVO", "ADENSAMENTO", "DEFICIT_TOTAL"]
dados_numericos = dados.select_dtypes(include='number')

# === 2. Calcular o impacto dos zeros em cada variável ===
impacto_zeros = {}

for col in dados_numericos.columns:
    if col not in saidas:
        grupo_zero = dados[dados[col] == 0][saidas].mean()
        grupo_nao_zero = dados[dados[col] != 0][saidas].mean()

        # diferença média entre quando há zero e quando não há
        impacto_zeros[col] = (grupo_nao_zero - grupo_zero)

# === 3. Organizar resultados em DataFrame ===
df_impacto = pd.DataFrame(impacto_zeros).T
df_impacto.columns = [f"Diferença média em {s}" for s in saidas]
df_impacto = df_impacto.sort_values(by=df_impacto.columns[-1], ascending=False)  # ordena pela última saída
print("\nImpacto médio dos zeros sobre as saídas:\n")
print(df_impacto.head(10).to_string(index=True))

# === 4. Visualização gráfica (exemplo para uma saída específica) ===
saida_ref = "DEFICIT_TOTAL"  # escolha da saída
coluna_ref = f"Diferença média em {saida_ref}"

df_plot = df_impacto[coluna_ref].sort_values(ascending=False).head(10).reset_index()
df_plot.columns = ["Variável", "Impacto médio (não zero - zero)"]

# === 5. Plot aprimorado ===
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Paleta diverging centrada no zero
palette = sns.diverging_palette(240, 10, as_cmap=False, n=10, s=80, l=50, center="light")

ax = sns.barplot(
    data=df_plot,
    y="Variável",
    x="Impacto médio (não zero - zero)",
    palette=palette
)

# Linha de referência no zero
plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)

# Rótulos de valor em cada barra
for i, (valor, var) in enumerate(zip(df_plot["Impacto médio (não zero - zero)"], df_plot["Variável"])):
    plt.text(
        valor + (5 if valor >= 0 else -5), i,
        f"{valor:.1f}",
        color='black',
        va='center',
        ha='left' if valor >= 0 else 'right',
        fontsize=9
    )

# Títulos e rótulos mais claros
plt.title(f"Top 10 variáveis com maior impacto dos zeros sobre {saida_ref}", fontsize=14, weight='bold', pad=15)
plt.xlabel("Impacto médio sobre o déficit (não zero - zero)", fontsize=12, weight='bold')
plt.ylabel("Variável", fontsize=12, weight='bold')

plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
