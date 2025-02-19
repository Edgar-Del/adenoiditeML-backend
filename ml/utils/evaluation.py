import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    # prever
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Acurácia do Modelo: {accuracy:.3f}")
    print("\nRelatório de Classificação: ")
    print(report)
    
    # Gráfico de importância das características
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        features = X_test.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title("Características Importantes")
        plt.bar(range(X_test.shape[1]), importances[indices])
        plt.xticks(range(X_test.shape[1]), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('models/saved/feature_importance.png')
        plt.close()