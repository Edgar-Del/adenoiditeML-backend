import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def train_model(data_path: str, model_path: str):
    # Carregar os dados
    df = pd.read_csv(data_path)
    X = df.drop(columns=['diagnostico_adenoidite'])
    y = df['diagnostico_adenoidite']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar PCA para redução de dimensionalidade
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Treinar modelo KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)

    # Treinar modelo K-Means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train_pca)

    # Salvar modelos treinados
    joblib.dump(knn, f"{model_path}/modelo_knn.joblib")
    joblib.dump(kmeans, f"{model_path}/modelo_kmeans.joblib")
    joblib.dump(pca, f"{model_path}/modelo_pca.joblib")
    print("Modelos treinados e salvos com sucesso!")
