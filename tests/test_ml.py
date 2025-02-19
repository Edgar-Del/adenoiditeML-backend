import pytest
import pandas as pd
import numpy as np
from ml.data.preprocessing import AdenoiditisPreprocessor
from ml.models.clustering import AdenoiditisClustering
from ml.models.classification import AdenoiditisClassifier

class TestMachineLearningComponents:
    @pytest.fixture
    def sample_data(self):
        """ um dataset exempo para teste"""
        return pd.DataFrame({
            'idade': [2, 3, 4, 5, 1],
            'genero': ['M', 'F', 'M', 'F', 'M'],
            'frequencia_ronco': [7.5, 6.0, 8.0, 5.5, 7.0],
            'dificuldade_respirar': [6.0, 5.0, 7.0, 4.5, 6.5],
            'obstrucao_nasal': [75.0, 60.0, 85.0, 50.0, 70.0],
            'apnea_sono': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'diagnostico_adenoidite': [1, 0, 1, 0, 1]
        })

    def test_preprocessor(self, sample_data):
        """Testando o Processamento de dados"""
        preprocessor = AdenoiditisPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert 'gravidade_sintoma' in X.columns

    def test_clustering(self, sample_data):
        """Testando a funcionalidade de agrupamento"""
        preprocessor = AdenoiditisPreprocessor()
        X, _ = preprocessor.fit_transform(sample_data)
        
        clusterer = AdenoiditisClustering(n_clusters=3)
        clusters = clusterer.fit_predict(X)
        
        assert len(clusters) == len(X)
        assert len(np.unique(clusters)) == 3

    def test_classification(self, sample_data):
        """Testar a classificação KNN"""
        preprocessor = AdenoiditisPreprocessor()
        X, y = preprocessor.fit_transform(sample_data)
        
        # separando os dados
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        classifier = AdenoiditisClassifier(n_neighbors=3)
        classifier.fit(X_train, y_train)
        
        predictions = classifier.predict(X_test)
        assert len(predictions) == len(X_test)