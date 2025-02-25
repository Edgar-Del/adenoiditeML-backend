#CÓDIGO DO MODELO ACTUALIZADO
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


np.random.seed(42)  # Para reprodutibilidade
n_registros = 150

# Features
data = {
    'id_paciente': range(1, n_registros + 1),
    'idade_mes': np.random.randint(6, 145, n_registros),
    'genero': np.random.choice([0, 1], n_registros),
    'obstrução_nasal_persistente': np.random.binomial(1, 0.7, n_registros),
    'secreção_nasal_purulenta': np.random.binomial(1, 0.4, n_registros),
    'dor_de_garganta_e_dificuldade_para_engolir': np.random.binomial(1, 0.6, n_registros),
    'Febre_e_mal_estar_geral': np.random.binomial(1, 0.5, n_registros),
    'linfodenopatia_cervical': np.random.binomial(1, 0.3, n_registros),
    'alteração_na_voz': np.random.binomial(1, 0.4, n_registros),
    'problemas_auditivos': np.random.binomial(1, 0.25, n_registros),
    'sintomas_de_infecção_recorrente': np.random.binomial(1, 0.5, n_registros),
    'disturbios_de_sono_e_desenvolvimento': np.random.binomial(1, 0.6, n_registros),
    'tamanho_adenoide': np.random.choice([1, 2, 3, 4], n_registros, p=[0.1, 0.3, 0.4, 0.2])    
}

# Diagnóstico baseado em critérios clínicos (ex: tamanho da adenoide + sintomas)
prob_diagnostico = (
    0.4 * (data['tamanho_adenoide'] / 4) + 
    0.3 * data['obstrução_nasal_persistente'] +
    0.2 * data['secreção_nasal_purulenta'] +
    0.1 * data['Febre_e_mal_estar_geral']
)

# Ajustar a probabilidade para garantir equilíbrio natural
# Definimos um limiar que garante ~50% de casos positivos e negativos
limiar = np.percentile(prob_diagnostico, 50)  # Limiar no percentil 50
data['diagnostico_adenoidite'] = (prob_diagnostico > limiar).astype(int)

# Criar DataFrame
df = pd.DataFrame(data)

# Verificar o equilíbrio do target
print(df['diagnostico_adenoidite'].value_counts(normalize=True))

#data['diagnostico_adenoidite'] = (prob_diagnostico + np.random.normal(0, 0.1, n_registros) > 0.5).astype(int)

# Criar DataFrame
df = pd.DataFrame(data)

# Exibir as primeiras linhas
print(df.head())

# Salvar em CSV
df.to_csv(r'data/raw/dataset.csv',index=False )

df.describe()

# Balanceamento do target
df.groupby('diagnostico_adenoidite').size()

# Separação dos features e dos targets

feature_columns = ['id_paciente', 'idade_mes', 'genero', 'obstrução_nasal_persistente', 'secreção_nasal_purulenta', 'dor_de_garganta_e_dificuldade_para_engolir', 'Febre_e_mal_estar_geral', 'linfodenopatia_cervical', 'alteração_na_voz', 'problemas_auditivos', 'sintomas_de_infecção_recorrente	', 'disturbios_de_sono_e_desenvolvimento']
#X = df[feature_columns].values
#Y = df['diagnostico_adenoidite'].values

from sklearn.model_selection import train_test_split

X = df.drop(['id_paciente', 'diagnostico_adenoidite'], axis=1)  # Remover ID e target
Y = df['diagnostico_adenoidite']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

# Visualização dos dados

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# na bibleoteca seaborn encontramos pairplot
plt.figure()
sns.pairplot(df.drop("id_paciente", axis=1), hue = "diagnostico_adenoidite", size=3, markers=["o", "s", "D"])
plt.show()

# Fit ou " Treino " do modelo
knn = classifier.fit(X_train, y_train,)
y_pred = classifier.predict(X_test)

# Aplicar um modelo para validação
y_predict = knn.predict(X_test)

from sklearn.metrics import ConfusionMatrixDisplay

# Supondo que você já tenha:
# - knn: modelo KNN treinado
# - X_test, y_test: dados de teste

# Gerar a matriz de confusão
ConfusionMatrixDisplay.from_estimator(
    knn, 
    X_test, 
    y_test,
    display_labels=['Sem Adenoidite', 'Com Adenoidite'], 
    cmap='Blues'
)

plt.title('Matriz de Confusão - KNN')
plt.show()

# Matriz de Confusão

#cm = confusion_matrix(y_test, y_pred)
#cm

print('Métricas de classificação: \n\n', classification_report(y_test, y_predict))
# Acuracia

accuracy = accuracy_score(y_test, y_pred)*100
print('A Acuracia do nosso modelo é igual a ' + str(round(accuracy, 2)) + '%.')

# Dicionário para visualizar os acertos dos errados


actual_vs_predict = pd.DataFrame({'actual: ':y_test,
                                 'prediction: ': y_predict})
actual_vs_predict.head(10)

# Normalizar os dados (importante para KNN)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)
X_padronizado = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, Y, test_size=0.2, random_state=42)

# Padronizado

# X_train, X_test, y_train, y_test = train_test_split(X_padronizado, Y, test_size=0.2, random_state=42)

# Fit ou " Treino " do modelo
knn = classifier.fit(X_train, y_train,)
y_pred = classifier.predict(X_test)
# Aplicar um modelo para validação
y_predict = knn.predict(X_test)

# Supondo que você já tenha:
# - knn: modelo KNN treinado
# - X_test, y_test: dados de teste

# Gerar a matriz de confusão
ConfusionMatrixDisplay.from_estimator(
    knn, 
    X_test, 
    y_test,
    display_labels=['Sem Adenoidite', 'Com Adenoidite'], 
    cmap='Blues'
)

plt.title('Matriz de Confusão - KNN')
plt.show()

# Métricas da Matriz de Confusão depois na normalização

print('Métricas de classificação: \n\n', classification_report(y_test, y_predict))

# Acuracia

accuracy = accuracy_score(y_test, y_pred)*100
print('A Acuracia do nosso modelo é igual a ' + str(round(accuracy, 2)) + '%.')

# Dicionário para visualizar os acertos dos errados


actual_vs_predict = pd.DataFrame({'actual: ':y_test,
                                 'prediction: ': y_predict})
actual_vs_predict.head(10)

# Salvar em CSV
df.to_csv(r'data/raw/dataset_normalizado.csv',index=False )

# outra maneira do PCA


# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA e reter 95% da variância
pca = PCA(n_components=0.95)  # Retém 95% da variância automaticamente
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Determinar o melhor valor de k para o KNN
best_k = None
best_score = 0

for k in range(1, 21, 2):  # Testa valores ímpares de k de 1 a 19
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    y_val_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_val_pred)

    if acc > best_score:
        best_score = acc
        best_k = k

print(f"Melhor k encontrado: {best_k} com acurácia de {best_score:.2f}")

# Treinar o KNN com o melhor k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_pca, y_train)

# Avaliar o modelo
y_pred = knn.predict(X_test_pca)
acc_final = accuracy_score(y_test, y_pred)
print(f"Acurácia final: {acc_final:.2f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()

# Calcular as métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Para classes desbalanceadas
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Exibir as métricas
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Exibir relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# APLICAR A VALIDAÇÃO CRUZADA
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

# Definir o pipeline: escalonamento → PCA → KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Padronização (substitua por MinMaxScaler se preferir)
    ('pca', PCA(n_components=3)), # Redução dimensional
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Definir o número de folds (ex.: 5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Executar validação cruzada
scores = cross_val_score(
    estimator=pipeline,
    X=X,  # Dados originais (não divididos em treino/teste)
    y=Y,
    cv=kfold,
    scoring='accuracy'  # Métrica de avaliação
)

print(f"Acurácia média: {scores.mean():.2f} (± {scores.std():.2f})")

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, Y, cv=skfold, scoring='accuracy')

from sklearn.model_selection import GridSearchCV

# Definir os hiperparâmetros a serem testados
param_grid = {
    'pca__n_components': [5, 10, 15],  # Número de componentes do PCA
    'knn__n_neighbors': [3, 5, 7]       # Valores de k para o KNN
}

# Criar o GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # Número de folds
    scoring='accuracy'
)

# Treinar o modelo
grid_search.fit(X, Y)

# Melhores parâmetros e acurácia
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor acurácia: {grid_search.best_score_:.2f}")

# Fazer previsões com o melhor modelo encontrado
y_pred = grid_search.best_estimator_.predict(X)

# Calcular a matriz de confusão
cm = confusion_matrix(Y, y_pred)

# Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()

k_list = list(range(1,50,2))
k_list

# APLICAR O CALCULO DO PCA

from sklearn.cluster import KMeans

# Aplicar K-Means (ex: 2 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_normalizado)
                              
# Redução para 2D (PCA) para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalizado)

# Plotar clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Clusters de Adenoidite (K-Means)')
plt.colorbar(label='Cluster')
plt.show()

df['cluster'] = clusters
cluster_stats = df.groupby('cluster').mean().T  # Médias por cluster
print(cluster_stats)

from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_normalizado)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=clusters, cmap='viridis')
plt.title('Clusters em 3D')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')
plt.colorbar(scatter, label='Cluster')
plt.show()

plt.figure(figsize=(12, 6))

# Subplot 1: Clusters do K-Means
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Clusters do K-Means')
plt.colorbar(label='Cluster')

# Subplot 2: Diagnóstico Real
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['diagnostico_adenoidite'], cmap='plasma', alpha=0.6)
plt.title('Diagnóstico Real')
plt.colorbar(label='Diagnóstico')

plt.tight_layout()
plt.show()

# Mapear features aos componentes principais
features = X.columns  # Substitua X pelo DataFrame original das features
componentes = pd.DataFrame(pca.components_, columns=features)

# Componente 1
print("Componente Principal 1 (maior influência):")
print(componentes.iloc[0].sort_values(ascending=False).head(3))

# Componente 2
print("\nComponente Principal 2 (maior influência):")
print(componentes.iloc[1].sort_values(ascending=False).head(3))

# Inicialização da lista que vai guardar os scores

cv_scores = []

# execução do KNN com diferentes valores de K

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, scoring='accuracy')
    cv_scores.append(scores.mean())

# Convertendo para o "error"
ERR = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title(' Melhor Valor de K', fontsize=20, fontweight='bold')
plt.xlabel('Numeros de vizinhos K', fontsize=15)
plt.ylabel('Erro de classificação', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, ERR)

plt.show()
best_k = k_list[ERR.index(min(ERR))]
print(" Valor Optimo para K é %d." % best_k)

# USANDO A CLUSTERIZAÇÃO COM K_MEANS ( IMPORTANDO O K_MEANS)
# A = A.loc[:, ['frequência_respiração_bucal','intensidade_ressoal','tamanho_adenoidite_mm']]
# Utilizando o algorítimo
kmeans2 = KMeans(n_clusters=2, random_state=42).fit(X)
# Verificando Quais foram os labels determinados pelo K_means
kmeans2.labels_

# Visualização em 3D

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Gráfico 3D com x, y, z:
ax.scatter(
    X['tamanho_adenoide'],  # Eixo x
    X['obstrução_nasal_persistente'],           # Eixo y
    X['secreção_nasal_purulenta'],         # Eixo z
    c=kmeans2.labels_                   # Cores baseadas nos clusters
)

ax.set_xlabel('tamanho_adenoide')
ax.set_ylabel('obstrução_nasal_persistente')
ax.set_zlabel('secreção_nasal_purulenta')
plt.show()

# Aplicar KMeans (exemplo):
kmeans4 = KMeans(n_clusters=4, random_state=42).fit(X)

# Plotar gráfico 2D:
fig, ax = plt.subplots()
scatter = ax.scatter(
    X['tamanho_adenoide'], 
    X['obstrução_nasal_persistente'], 
    c=kmeans2.labels_, 
    cmap='viridis'
)

plt.colorbar(scatter, label='Cluster')
plt.xlabel('tamanho_adenoide')
plt.ylabel('obstrução_nasal_persistente')
plt.show()

# Percorrendo diferentes valores de K
valores_k = []
inercias = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    valores_k.append(i)
    inercias.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(valores_k, inercias)
plt.show()

# AVALIAR OS ModeloS DE K SUGERIDOS
#  Visualização dos targets e os labels em um mesmo Gráfico


fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 1 linha, 2 colunas

# Gráfico 1: Clusters do KMeans
scatter1 = ax[0].scatter(
    X['frequência_respiração_bucal'], 
    X['intensidade_ressoal'], 
    c=kmeans2.labels_,
    cmap='viridis'
)
ax[0].set_title('Clusters (KMeans)')
ax[0].set_xlabel('Frequência Respiração Bucal')
ax[0].set_ylabel('Intensidade do Ronco')

# Gráfico 2: Diagnóstico Real (y)
scatter2 = ax[1].scatter(
    X['frequência_respiração_bucal'], 
    X['intensidade_ressoal'], 
    c=Y,
    cmap='plasma'
)
ax[1].set_title('Diagnóstico Real')
ax[1].set_xlabel('Frequência Respiração Bucal')
ax[1].set_ylabel('Intensidade do Ronco')

plt.colorbar(scatter1, ax=ax[0], label='Cluster')
plt.colorbar(scatter2, ax=ax[1], label='Diagnóstico')
plt.tight_layout()
plt.show()

from sklearn import metrics
metrics.adjusted_rand_score(Y,kmeans.labels_)

# k = 4
metrics.adjusted_rand_score(Y,kmeans4.labels_)

# Percorrendo oS valores de K Calculando esses dois indicadores

valores_k = []
ARI = []
#RI = []

for i in range(1,15):
    kmeans = KMeans(n_clusters=i, random_state=42).fit(X)
    valores_k.append(i)
   # RI.append(metrics.rand_score(y,kmeans.labels_))
    ARI.append(metrics.adjusted_rand_score(Y,kmeans.labels_))

fig, ax = plt.subplots()
ax.plot(valores_k, ARI)
#ax.plot(valores_k, RI)

plt.show()

# COEFICIENTE DE SILHUETA, MÉTRICAS PARA AVALIAR OS MODELOS MESMO SEM TER OS RÓTULOS DOS DADOS
# Verificando os silhuette_score para k = 2
metrics.silhouette_score(X, kmeans.labels_)

# Percorrendo para diferentes valores de K
valores_k = []
s = []


for i in range(2,15):
    kmeans = KMeans(n_clusters=i, random_state=42).fit(X)
    valores_k.append(i)
    s.append(metrics.silhouette_score(X, kmeans.labels_))

# Visualização Gráfica
fig, ax = plt.subplots()

ax.plot(valores_k, s)
#ax.plot(valores_k, RI)
plt.show()

