import os

def find_model_files(directory):
    """Trouve tous les fichiers 'MLmodel' dans le projet"""
    model_files = []
    
    for root, dirs, files in os.walk(directory):
        if 'MLmodel' in files:  # Fichier signature d'un modèle MLflow
            model_files.append(root)
    
    return model_files

print(" Recherche de TOUS les modèles MLflow dans le projet...\n")

# Chercher depuis la racine
models = find_model_files(".")

if models:
    print(f" {len(models)} modèle(s) trouvé(s)!\n")
    
    for i, model_path in enumerate(models, 1):
        print(f"{i}. {model_path}")
        
        # Extraire le chemin utilisable
        if "artifacts" in model_path:
            # Remonter jusqu'à mlruns
            parts = model_path.split(os.sep)
            if "mlruns" in parts:
                mlruns_idx = parts.index("mlruns")
                clean_path = "/".join(parts[mlruns_idx:])
                print(f"    Chemin à utiliser: {clean_path}")
        print()
else:
    print(" Aucun modèle MLflow trouvé!")
    print("\nVérifions si le dossier mlruns existe:")
    
    if os.path.exists("mlruns"):
        print(" mlruns/ existe")
        print("\nContenu de mlruns/:")
        for item in os.listdir("mlruns"):
            item_path = os.path.join("mlruns", item)
            if os.path.isdir(item_path):
                print(f"   {item}/")
                # Compter les sous-dossiers
                try:
                    subitems = os.listdir(item_path)
                    subdirs = [s for s in subitems if os.path.isdir(os.path.join(item_path, s))]
                    print(f"      └─ {len(subdirs)} sous-dossier(s)")
                except:
                    pass
            else:
                print(f"   {item}")
    else:
        print(" mlruns/ n'existe pas!")
        print(f"Répertoire actuel: {os.getcwd()}")