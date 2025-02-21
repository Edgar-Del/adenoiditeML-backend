from ml.clustering.clustering import AdenoiditisClustering
import pandas as pd

# Carregar os dados processados
X = pd.read_csv("data/processed/X.csv")

# Criar e executar o modelo de clustering
clustering = AdenoiditisClustering(n_clusters=3)
clusters = clustering.fit_predict(X)

# Adicionar os clusters ao dataset
X["Cluster"] = clusters
X.to_csv("data/processed/X_clustered.csv", index=False)