from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="API ML Adenoidite", version="1.0")

# Inclui os endpoints definidos no arquivo endpoints.py
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)