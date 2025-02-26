import os
import joblib
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def train_model(data_path: str, model_path: str):
    print("🔄 Iniciando treinamento do modelo...")
    
    # Criar diretório para salvar os modelos, se não existir
    os.makedirs(model_path, exist_ok=True)
    
    # Carregar os dados
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset carregado com {df.shape[0]} registros e {df.shape[1]} colunas.")
    except Exception as e:
        print(f"❌ Erro ao carregar o dataset: {e}")
        sys.exit(1)
    
    # Verificar colunas
    print(f"📌 Colunas disponíveis: {df.columns.tolist()}")  
    
    # Verificar se a coluna "diagnostico_adenoidite" existe
    if 'diagnostico_adenoidite' not in df.columns:
        print("❌ Erro: A coluna 'diagnostico_adenoidite' não existe no dataset!")
        sys.exit(1)
    
    # Converter variáveis categóricas para numéricas
    if df['genero'].dtype == 'object':  
        df['genero'] = df['genero'].map({'M': 0, 'F': 1})  # M = 0, F = 1
        print("🔄 Variável 'genero' convertida para numérico.")

    # Separar features e target
    X = df.drop(columns=['diagnostico_adenoidite', 'id_paciente'])  # Remover ID
    y = df['diagnostico_adenoidite']
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Dados divididos: {X_train.shape[0]} treino, {X_test.shape[0]} teste.")
    
    # Aplicar PCA para redução de dimensionalidade
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("📉 PCA aplicado, reduzindo dimensões para 2 componentes principais.")
    
    # Treinar modelo KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    print("🤖 Modelo KNN treinado com sucesso!")
    
    # Treinar modelo K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train_pca)
    print("🔹 Modelo K-Means treinado com sucesso!")
    
    # Salvar modelos treinados
    try:
        joblib.dump(knn, os.path.join(model_path, "modelo_knn.joblib"))
        joblib.dump(kmeans, os.path.join(model_path, "modelo_kmeans.joblib"))
        joblib.dump(pca, os.path.join(model_path, "modelo_pca.joblib"))
        print("✅ Modelos salvos com sucesso no diretório 'models/saved'!")
    except Exception as e:
        print(f"❌ Erro ao salvar os modelos: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Uso correto: python -m ml.models.train <data_path> <model_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    train_model(data_path, model_path)
