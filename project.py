import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# DIAGNÓSTICO ESTRATÉGICO DE DADOS

"""
OBJETIVO: Identificar o papel dos dados no contexto de negócios e aplicar Big Data.

1. APLICAÇÃO DOS 5Vs:
   - Volume: Grande base de dados California Housing com milhares de registros.
   - Velocidade: Necessidade de processamento ágil para acompanhar o mercado imobiliário.
   - Variedade: Integração de dados numéricos (tabelas) e qualitativos (feedbacks).
   - Veracidade: Limpeza de dados para garantir previsões imobiliárias precisas.
   - Valor: Inteligência para precificação e decisões de investimento.

2. CLASSIFICAÇÃO:
   - Estruturados: Preços, número de quartos, localização (latitude/longitude).
   - Não Estruturados: Feedbacks de clientes em texto, imagens das fachadas das casas.

3. USO ESTRATÉGICO: 
   Suporte à tomada de decisão para campanhas de marketing segmentadas e análise de 
   risco para crédito imobiliário.
"""

# ANÁLISE EXPLORATÓRIA

print("--- Análise Exploratória ---")

# Importação da base
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# Estatísticas Descritivas
print(df.describe())

# Matriz de Correlação [cite: 2, 5]
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação - California Housing")
plt.show()

# MODELAGEM

print("\n--- Modelagem (PCA, K-means, Regressão) ---")

# Normalização (Essencial para PCA e K-means)[cite: 2, 5]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('MedHouseVal', axis=1))

# 1. PCA (Redução de Dimensionalidade)[cite: 2, 5]
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
print(f"Variância explicada pelo PCA: {pca.explained_variance_ratio_.sum():.2f}")

# 2. K-means (Clusterização)[cite: 2, 5]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)
print("Clusters gerados com sucesso.")

# 3. Regressão Linear Múltipla [cite: 2, 5]
X = df.drop(['MedHouseVal', 'Cluster'], axis=1)
y = df['MedHouseVal']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"R² do Modelo: {r2_score(y, y_pred):.4f}")
print(f"Erro Quadrático Médio (MSE): {mean_squared_error(y, y_pred):.4f}")

# ANÁLISE DE FLUXO (Cenário E-commerce)

print("\n--- Análise de Fluxo ---")

# Simulação de dados de funil
fluxo = {
    'Etapa': ['Visitas', 'Carrinho', 'Checkout', 'Pagamento'],
    'Usuarios': [10000, 4500, 1200, 1000]
}
df_fluxo = pd.DataFrame(fluxo)
df_fluxo['Perda_Percentual'] = df_fluxo['Usuarios'].pct_change().fillna(0) * -1

taxa_conversão = (df_fluxo.iloc[-1]['Usuarios'] / df_fluxo.iloc[0]['Usuarios']) * 100
print(df_fluxo)
print(f"Taxa de Conversão Final: {taxa_conversão}%")

# MINERAÇÃO DE TEXTO

print("\n--- Mineração de Texto ---")

# Exemplo de comentários de clientes
comentarios = [
    "Ótima localização, mas o preço está muito alto",
    "Casa velha e mal cuidada, não gostei",
    "Excelente oportunidade de investimento, amei",
    "O processo de compra foi muito lento e burocrático",
    "Adorei o atendimento e a clareza dos dados"
]

# Simulação de análise de sentimentos (Palavras-chave)
palavras_positivas = ['ótima', 'excelente', 'amei', 'adorei', 'clareza']
palavras_negativas = ['alto', 'velha', 'lento', 'burocrático', 'não gostei']

def analisar_sentimento(texto):
    texto = texto.lower()
    pos = sum(1 for p in palavras_positivas if p in texto)
    neg = sum(1 for p in palavras_negativas if p in texto)
    if pos > neg: return 'Positivo'
    if neg > pos: return 'Negativo'
    return 'Neutro'

sentimentos = [analisar_sentimento(c) for c in comentarios]
for c, s in zip(comentarios, sentimentos):
    print(f"Comentário: {c} | Sentimento: {s}")