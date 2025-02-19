from fastapi.testclient import TestClient
from app.main import app
import pytest

class TestAdenoiditisAPI:
    client = TestClient(app)

    def test_diagnose_endpoint(self):
        """Teste do endpoint de  diagnositico com entradas válidas"""
        input_data = {
            "idade": 4,
            "genero": "M",
            "frequencia_ronco": 7.5,
            "dificuldade_respirar": 6.0,
            "obstrucao_nasal": 75.0,
            "apnea_sono": "Sim"
        }
        
        response = self.client.post("/api/v1/diagnosticar", json=input_data)
        
        assert response.status_code == 200
        
        # Check response structure
        result = response.json()
        assert "diagnostico" in result
        assert "grupo_gravidade" in result
        assert "confianca" in result
        assert "recomendacoes" in result

    def test_diagnose_invalid_input(self):
        """Teste do endpoint de  diagnositico com entradas válidas"""
        invalid_input = {
            "idade": 6,  # Fora do intervalo
            "gender": "X",  # Género inválido
            "frequencia_ronco": 15.0,  # fora do intervado
        }
        
        response = self.client.post("/api/v1/diagnosticar", json=invalid_input)
        assert response.status_code == 422  # Entidade não "processável"