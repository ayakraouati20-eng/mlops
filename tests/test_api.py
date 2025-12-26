import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch
import numpy as np

# Créer un client de test
client = TestClient(app)

def test_root():
    """Test l'endpoint racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()

@pytest.mark.skipif(
    not hasattr(app, 'model') or app.model is None,
    reason="Modèle non chargé en CI/CD"
)
def test_predict_valid_input():
    """Test l'endpoint de prédiction avec des données valides."""
    valid_data = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.2,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.3
    }
    response = client.post("/predict/", json=valid_data)
    
    # En CI, le modèle peut ne pas être chargé
    if response.status_code == 503:
        pytest.skip("Modèle non disponible en CI/CD")
    
    assert response.status_code == 200
    assert "predicted_house_value" in response.json()
    assert "predicted_price_usd" in response.json()
    
    # Vérifier que le prix prédit est positif
    predicted_value = response.json()["predicted_house_value"]
    assert predicted_value > 0, "Le prix prédit devrait être positif"

def test_predict_invalid_input():
    """Test l'endpoint avec des données invalides."""
    invalid_data = {
        "MedInc": "invalid",  # Valeur non-numérique
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.2,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.3
    }
    response = client.post("/predict/", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_missing_fields():
    """Test l'endpoint avec des champs manquants."""
    incomplete_data = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.0
        # Champs manquants
    }
    response = client.post("/predict/", json=incomplete_data)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_negative_values():
    """Test avec des valeurs négatives."""
    data_with_negatives = {
        "MedInc": -1.0,  # Valeur négative
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.2,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.3
    }
    response = client.post("/predict/", json=data_with_negatives)
    
    # En CI, le modèle peut ne pas être chargé
    if response.status_code == 503:
        pytest.skip("Modèle non disponible en CI/CD")
    
    # L'API peut accepter mais le résultat peut être bizarre
    assert response.status_code in [200, 422]

def test_predict_extreme_values():
    """Test avec des valeurs extrêmes."""
    extreme_data = {
        "MedInc": 15.0,  # Très élevé
        "HouseAge": 52.0,
        "AveRooms": 10.0,
        "AveBedrms": 3.0,
        "Population": 10000.0,
        "AveOccup": 5.0,
        "Latitude": 41.0,
        "Longitude": -115.0
    }
    response = client.post("/predict/", json=extreme_data)
    
    # En CI, le modèle peut ne pas être chargé
    if response.status_code == 503:
        pytest.skip("Modèle non disponible en CI/CD")
    
    assert response.status_code == 200
    assert "predicted_house_value" in response.json()

def test_model_endpoint_structure():
    """Test que l'endpoint predict existe et retourne le bon format."""
    valid_data = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.2,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.3
    }
    response = client.post("/predict/", json=valid_data)
    
    # L'endpoint doit exister (200 ou 503)
    assert response.status_code in [200, 503]
    
    # Si le modèle est chargé, vérifier la structure
    if response.status_code == 200:
        json_response = response.json()
        assert "predicted_house_value" in json_response
        assert "predicted_price_usd" in json_response