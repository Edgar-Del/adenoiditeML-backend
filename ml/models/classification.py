from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AdenoiditisClassifier:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        # Gerar matriz de confusão.
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.ylabel('Rótulo (Verdadeiro)')
        plt.xlabel('Rótulo (Previsto)')
        plt.savefig('results/confusao_matriz.png')
        plt.close()
        
        return classification_report(y_test, y_pred)