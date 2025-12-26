import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# FastAPI setup
app = FastAPI(
    title="ImmoPrix API",
    description="API de prédiction du prix médian des maisons en Californie",
    version="1.0.0"
)

# Charger le modèle directement depuis mlruns
MODEL_PATH = "mlruns/1/models/m-fccd79b9471d47ca9d265f2886608d22/artifacts"

try:
    print(f"🔍 Chargement du modèle depuis {MODEL_PATH}...")
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print(f"✅ Modèle chargé avec succès!")
except Exception as e:
    print(f"❌ Échec du chargement: {e}")
    model = None

# Pydantic model
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API ImmoPrix",
        "endpoints": {
            "/": "Informations",
            "/predict/": "Prédiction (POST)",
            "/health/": "Vérification de l'état de l'API",
            "/docs": "Documentation Swagger"
        }
    }

@app.get("/health/")
def health_check():
    """Vérifier l'état de l'API et du modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Test simple de prédiction
        test_input = np.array([[3.0, 20.0, 5.0, 1.0, 1000.0, 3.0, 37.0, -122.0]])
        _ = model.predict(test_input)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_path": MODEL_PATH
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/predict/")
def predict(features: HousingFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    input_data = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
        features.Population, features.AveOccup, features.Latitude, features.Longitude
    ]])
    
    prediction = model.predict(input_data)
    predicted_value = float(prediction[0])
    predicted_price_usd = predicted_value * 100000
    
    return {
        "predicted_house_value": predicted_value,
        "predicted_price_usd": predicted_price_usd
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)