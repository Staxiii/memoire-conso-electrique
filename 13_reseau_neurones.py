import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

CHEMIN = "data/traite/dataset_enrichi.csv"

# --- 1. Chargement ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

VARIABLES = [
    "temperature_fr", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire",
    "degres_froid", "degres_chaud", "temp_moy_24h", "temp_veille",
]
CIBLE = "conso"

# --- 2. Découpage ---
train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"]
X_train, y_train = train[VARIABLES], train[CIBLE]
X_test, y_test = test[VARIABLES], test[CIBLE]

# --- 3. Mise à l'échelle : calculée sur l'ENTRAÎNEMENT, appliquée aux deux ---
echelle = StandardScaler()
X_train_norm = echelle.fit_transform(X_train)   # calcule ET applique
X_test_norm = echelle.transform(X_test)         # applique seulement

# --- 4. Mesures ---
def mae(reel, prevu):
    return (reel - prevu).abs().mean()

def mape(reel, prevu):
    return ((reel - prevu).abs() / reel).mean() * 100

# --- 5. Le réseau de neurones ---
modele = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # deux couches cachées : 64 puis 32 neurones
    activation="relu",             # fonction d'activation standard
    max_iter=300,                  # nombre de passages sur les données
    random_state=42,
    early_stopping=True,           # s'arrête si ça n'améliore plus
)
print("Entraînement du réseau (peut prendre 1 à 2 minutes)...")
modele.fit(X_train_norm, y_train)
prediction = modele.predict(X_test_norm)

# --- 6. Résultats comparés ---
print("\n--- Comparaison finale sur 2025 ---")
print("Baseline J-7    : 6.68 %")
print("Réseau de neurones :", round(mape(y_test, prediction), 2), "%")
print("Gradient boosting  : 3.61 %")
print("Prévision RTE   : 2.79 %")
print()
print("MAE du réseau :", round(mae(y_test, prediction), 0), "MW")