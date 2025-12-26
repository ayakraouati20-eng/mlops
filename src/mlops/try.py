"""
Mission 2: Entraînement de modèles avec MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("California_Housing_Prediction")

def load_data():    
    """Charge les données standardisées"""
    print(" Chargement des données...")
    df = pd.read_csv(r"C:\Users\w197947\House_pred\mlops\notebooks\data\housing_standardized.csv")
    
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f" Données chargées: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Calcule les métriques"""
    y_pred = model.predict(X_test)
    
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

def train_model(model_name, model, X_train, X_test, y_train, y_test, params=None):
    """Fonction générique pour entraîner et logger un modèle"""
    print(f"\n Entraînement: {model_name}")
    
    with mlflow.start_run(run_name=model_name) as run:
        # Entraînement
        model.fit(X_train, y_train)
        
        # Évaluation
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log paramètres
        mlflow.log_param("model_type", type(model).__name__)
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        # Log métriques
        mlflow.log_metrics(metrics)
        
        # Log modèle - NOUVELLE SYNTAXE
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )
        
        print(f"  R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
        print(f"  Run ID: {run.info.run_id}")
        
    return model, metrics, run.info.run_id

def register_best_model():
    """Trouve et promote le meilleur modèle"""
    from mlflow.tracking import MlflowClient
    import traceback
    
    client = MlflowClient()
    MODEL_NAME = "CaliforniaHousingBestModel"
    
    print("\n" + "=" * 60)
    print(" RECHERCHE DU MEILLEUR MODÈLE")
    print("=" * 60)
    
    try:
        experiment = client.get_experiment_by_name("California_Housing_Prediction")
        experiment_id = experiment.experiment_id
        
        # Chercher le meilleur run
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=1
        )
        
        if not runs:
            print(" Aucun run trouvé!")
            return
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        run_name = best_run.data.tags.get("mlflow.runName", "Unknown")
        r2 = best_run.data.metrics.get("r2", 0)
        
        print(f"\n Meilleur run: {run_name}")
        print(f"   Run ID: {run_id}")
        print(f"   R²: {r2:.4f}")
        
        # URI du modèle avec le nouveau path "model"
        model_uri = f"runs:/{run_id}/model"
        print(f"   Model URI: {model_uri}")
        
        # Enregistrer
        print(f"\n Enregistrement dans Model Registry...")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        
        print(f" Enregistré: {MODEL_NAME} v{model_version.version}")
        
        # Promouvoir
        print(f"\n Promotion en Production...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        client.update_model_version(
            name=MODEL_NAME,
            version=model_version.version,
            description=f"Meilleur modèle - R²={r2:.4f}"
        )
        
        print(f" '{MODEL_NAME}' v{model_version.version} → PRODUCTION!")
        print("\n Vérifie l'onglet 'Models' dans MLflow UI")
        
    except Exception as e:
        print(f"\n ERREUR: {e}")
        traceback.print_exc()

def main():
    print("=" * 60)
    print(" MISSION 2: Entraînement avec MLflow")
    print("=" * 60)
    
    # Charger données
    X_train, X_test, y_train, y_test = load_data()
    
    # Entraîner les 3 modèles
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    _, lr_metrics, _ = train_model("Linear_Regression", lr, X_train, X_test, y_train, y_test)
    results['Linear Regression'] = lr_metrics
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_params = {'n_estimators': 100, 'max_depth': 20, 'random_state': 42}
    _, rf_metrics, _ = train_model("Random_Forest", rf, X_train, X_test, y_train, y_test, rf_params)
    results['Random Forest'] = rf_metrics
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42}
    _, gb_metrics, _ = train_model("Gradient_Boosting", gb, X_train, X_test, y_train, y_test, gb_params)
    results['Gradient Boosting'] = gb_metrics
    
    # Résumé
    print("\n" + "=" * 60)
    print("------------------------ RÉSUMÉ-----------------------")
    print("=" * 60)
    
    best_name = max(results, key=lambda x: results[x]['r2'])
    best_r2 = results[best_name]['r2']
    
    for name, metrics in results.items():
        print(f"{name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    print(f"\n--------------------- MEILLEUR: {best_name} (R²={best_r2:.4f})--------------------")
    
    # Promotion
    register_best_model()

if __name__ == "__main__":
    main()