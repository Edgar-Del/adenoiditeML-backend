from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AdenoiditisClustering:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit_predict(self, X):
        """
        Agrupar pacientes com base na gravidade dos sintomas:  
            - Leve  
            - Moderado  
            - Grave 
        """
        clusters = self.kmeans.fit_predict(X)
        
        # Criar visualização
        plt.figure(figsize=(10, 6))
        plt.scatter(
            X['frequencia_ronco'],
            X['dificuldade_respirar'],
            c=clusters,
            cmap='viridis'
        )
        plt.xlabel('Frequencia de Ronco')
        plt.ylabel('Dificuldade de Respiração')
        plt.title('Agrupamento por Gravidade de Sintomas')
        plt.savefig('results/clusters.png')
        plt.close()
        
        return clusters