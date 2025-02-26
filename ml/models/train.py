import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from ml.data.preprocessing import DataPreprocessor


def train_model(data_path, model_path):
    print("🔄 Iniciando treinamento do modelo...")
    
    # Carregar dataset
    df = pd.read_csv(data_path)
    print(f"✅ Dataset carregado com {df.shape[0]} registros e {df.shape[1]} colunas.")

    # Inicializar pré-processador
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Dados divididos: {X_train.shape[0]} treino, {X_test.shape[0]} teste.")

    # Aplicar PCA para redução de dimensionalidade
    pca = PCA(n_components=5)  # Ajuste o número de componentes conforme necessário
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Treinar modelo KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    
    # Salvar modelos
    joblib.dump(knn, f"{model_path}/modelo_knn.joblib")
    joblib.dump(pca, f"{model_path}/modelo_pca.joblib")
    print("✅ Modelos salvos com sucesso!")

    # Avaliar modelo
    accuracy = knn.score(X_test_pca, y_test) * 100
    print(f"🎯 Acurácia do modelo: {accuracy:.2f}%")


if __name__ == "__main__":
    train_model("data/raw/dataset.csv", "models/saved")
