"""
Interface Streamlit pour ImmoPrix
Application de prédiction du prix des maisons en Californie
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# Configuration de la page
st.set_page_config(
    page_title="ImmoPrix - Prédiction de Prix",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API
API_URL = os.getenv("API_URL", "http://api:8000")
# CSS personnalisé
st.markdown(
    """
    <style>
    /* Fond avec image */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Carte principale */
    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Titres */
    h1 {
        color: #667eea;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        color: #667eea;
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Success/Error messages */
    .success-box {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Fonction pour vérifier l'état de l'API
def check_api_health():
    """Vérifie si l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/health/", timeout=2)
        return response.status_code == 200
    except:
        return False

# Fonction de prédiction
def predict_price(input_data):
    """Envoie une requête de prédiction à l'API"""
    try:
        response = requests.post(f"{API_URL}/predict/", json=input_data, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Impossible de se connecter à l'API. Vérifiez qu'elle est démarrée."
    except Exception as e:
        return None, f"Erreur: {str(e)}"

# Header
st.title("🏠 ImmoPrix - Prédiction de Prix Immobiliers")
st.markdown("### Estimez le prix médian des maisons en Californie avec l'IA")

# Vérification de l'état de l'API
api_status = check_api_health()

# Sidebar - Informations et état
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
    st.title("📊 Informations")
    
    # État de l'API
    if api_status:
        st.success("✅ API connectée")
    else:
        st.error("❌ API déconnectée")
        st.warning("Démarrez l'API avec:\n```\npoetry run uvicorn api.main:app --reload\n```")
    
    st.markdown("---")
    
    # Guide des features
    st.markdown("### 📖 Guide des caractéristiques")
    with st.expander("📍 Localisation"):
        st.write("**Latitude** : Position Nord-Sud (32-42°N)")
        st.write("**Longitude** : Position Est-Ouest (-125 à -114°E)")
    
    with st.expander("💰 Économie"):
        st.write("**MedInc** : Revenu médian du quartier")
        st.write("Typique: 2.0 - 6.0 (en 10K$)")
    
    with st.expander("🏡 Propriété"):
        st.write("**HouseAge** : Âge des maisons (1-52 ans)")
        st.write("**AveRooms** : Pièces par logement (3-8)")
        st.write("**AveBedrms** : Chambres par logement (1-2)")
        st.write("**AveOccup** : Occupants par logement (2-4)")
    
    with st.expander("👥 Démographie"):
        st.write("**Population** : Habitants du secteur")
        st.write("Typique: 500 - 3000")
    
    st.markdown("---")
    
    # Exemples
    st.markdown("### 🎯 Exemples rapides")
    
    if st.button("🌟 Maison Premium"):
        st.session_state.example = "premium"
    if st.button("🏘️ Maison Standard"):
        st.session_state.example = "standard"
    if st.button("💼 Investissement"):
        st.session_state.example = "investment"

# Exemples prédéfinis
examples = {
    "premium": {
        "MedInc": 8.0, "HouseAge": 15.0, "AveRooms": 7.5, "AveBedrms": 1.5,
        "Population": 800.0, "AveOccup": 2.5, "Latitude": 37.8, "Longitude": -122.4,
        "name": "Maison Premium - San Francisco Bay Area"
    },
    "standard": {
        "MedInc": 3.5, "HouseAge": 28.0, "AveRooms": 5.2, "AveBedrms": 1.1,
        "Population": 1500.0, "AveOccup": 3.0, "Latitude": 34.05, "Longitude": -118.25,
        "name": "Maison Standard - Los Angeles"
    },
    "investment": {
        "MedInc": 4.2, "HouseAge": 20.0, "AveRooms": 6.0, "AveBedrms": 1.3,
        "Population": 1200.0, "AveOccup": 2.8, "Latitude": 32.7, "Longitude": -117.2,
        "name": "Investissement - San Diego"
    }
}

# Formulaire principal
st.markdown("## 📝 Caractéristiques de la propriété")

# Utiliser un exemple si sélectionné
if "example" in st.session_state and st.session_state.example in examples:
    example_data = examples[st.session_state.example]
    st.info(f"📌 Exemple chargé: **{example_data['name']}**")
else:
    example_data = {}

# Layout en colonnes
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💰 Informations Économiques")
    MedInc = st.number_input(
        "Revenu Médian (en 10K$)",
        min_value=0.0,
        max_value=15.0,
        value=float(example_data.get("MedInc", 3.5)),
        step=0.1,
        help="Revenu médian du quartier en dizaines de milliers de dollars"
    )
    
    st.markdown("#### 🏡 Caractéristiques du Logement")
    HouseAge = st.number_input(
        "Âge de la Maison (années)",
        min_value=1,
        max_value=100,
        value=int(example_data.get("HouseAge", 25)),
        help="Âge médian des maisons du secteur"
    )
    
    AveRooms = st.number_input(
        "Nombre Moyen de Pièces",
        min_value=1.0,
        max_value=20.0,
        value=float(example_data.get("AveRooms", 5.0)),
        step=0.1,
        help="Nombre moyen de pièces par logement"
    )
    
    AveBedrms = st.number_input(
        "Nombre Moyen de Chambres",
        min_value=0.5,
        max_value=10.0,
        value=float(example_data.get("AveBedrms", 1.2)),
        step=0.1,
        help="Nombre moyen de chambres par logement"
    )

with col2:
    st.markdown("#### 👥 Informations Démographiques")
    Population = st.number_input(
        "Population du Secteur",
        min_value=1,
        max_value=50000,
        value=int(example_data.get("Population", 1500)),
        help="Nombre d'habitants dans le secteur"
    )
    
    AveOccup = st.number_input(
        "Occupation Moyenne",
        min_value=1.0,
        max_value=20.0,
        value=float(example_data.get("AveOccup", 3.0)),
        step=0.1,
        help="Nombre moyen d'occupants par logement"
    )
    
    st.markdown("#### 📍 Localisation")
    Latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=float(example_data.get("Latitude", 37.5)),
        step=0.01,
        format="%.2f",
        help="Coordonnée géographique Nord-Sud"
    )
    
    Longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=float(example_data.get("Longitude", -122.3)),
        step=0.01,
        format="%.2f",
        help="Coordonnée géographique Est-Ouest"
    )

# Espace
st.markdown("---")

# Bouton de prédiction
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])

with col_button2:
    predict_button = st.button("🔮 PRÉDIRE LE PRIX", disabled=not api_status, use_container_width=True)

# Prédiction
if predict_button:
    if not api_status:
        st.error("❌ L'API n'est pas accessible. Veuillez la démarrer.")
    else:
        # Préparer les données
        input_data = {
            "MedInc": MedInc,
            "HouseAge": float(HouseAge),
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": float(Population),
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        }
        
        # Afficher un spinner pendant la prédiction
        with st.spinner("🔄 Calcul de la prédiction en cours..."):
            result, error = predict_price(input_data)
        
        if error:
            st.error(f"❌ {error}")
        else:
            # Afficher les résultats
            st.success("✅ Prédiction réussie !")
            
            predicted_value = result["predicted_house_value"]
            predicted_price_usd = result["predicted_price_usd"]
            
            # Métriques principales
            st.markdown("## 📊 Résultats de la Prédiction")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric(
                    label="💵 Prix Prédit",
                    value=f"${predicted_price_usd:,.0f}",
                    delta=None
                )
            
            with col_metric2:
                st.metric(
                    label="📈 Valeur (100K$)",
                    value=f"{predicted_value:.2f}",
                    delta=None
                )
            
            with col_metric3:
                # Calcul du prix par pièce
                price_per_room = predicted_price_usd / AveRooms if AveRooms > 0 else 0
                st.metric(
                    label="🏠 Prix/Pièce",
                    value=f"${price_per_room:,.0f}",
                    delta=None
                )
            
            # Visualisation comparative
            st.markdown("### 📈 Analyse Comparative")
            
            # Créer un DataFrame pour comparaison
            comparison_data = pd.DataFrame({
                'Catégorie': ['Budget', 'Standard', 'Premium', 'Votre Prédiction'],
                'Prix': [200000, 350000, 600000, predicted_price_usd],
                'Couleur': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#667eea']
            })
            
            fig = px.bar(
                comparison_data,
                x='Catégorie',
                y='Prix',
                color='Catégorie',
                color_discrete_sequence=comparison_data['Couleur'],
                title='Comparaison avec les Segments de Marché',
                labels={'Prix': 'Prix (USD)', 'Catégorie': ''}
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails de la requête
            with st.expander("🔍 Voir les détails techniques"):
                st.json(result)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🏠 <strong>ImmoPrix</strong> - Prédiction de prix immobiliers avec Machine Learning</p>
        <p>Modèle en Production depuis MLflow Model Registry</p>
    </div>
    """,
    unsafe_allow_html=True
)