from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API ML Adenoidite"
    MODEL_PATH: str = "models/saved/random_forest_model.joblib"
    
    class Config:
        case_sensitive = True

settings = Settings()