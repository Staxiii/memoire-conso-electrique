import pandas as pd
import lightgbm as lgb

CHEMIN = "data/traite/dataset_enrichi.csv"

# --- 1. Chargement ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

# --- 2. Version ALLÉGÉE : une seule variable de température (la plus prédictive) ---
# On retire temperature_fr, degres_froid, degres_chaud, temp_veille
# et on ne garde que temp_moy_24h, qui portait déjà tout le signal thermique.
VARIABLES_ALLEGEES = [
    "temp_moy_24h", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire",
]
CIBLE = "conso"

train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"]
X_train, y_train = train[VARIABLES_ALLEGEES], train[CIBLE]
X_test, y_test = test[VARIABLES_ALLEGEES], test[CIBLE]

print("Modèle allégé :", len(VARIABLES_ALLEGEES), "variables")

# --- 3. Mesures ---
def mae(reel, prevu):
    return (reel - prevu).abs().mean()

def mape(reel, prevu):
    return ((reel - prevu).abs() / reel).mean() * 100

# --- 4. Entraînement, réglages STRICTEMENT identiques au modèle complet ---
modele = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    random_state=42, verbose=-1,
)
modele.fit(X_train, y_train)
prediction = modele.predict(X_test)

# --- 5. Comparaison complet vs allégé ---
print("\n--- Complet (11 variables) vs Allégé (7 variables) ---")
print("Modèle complet : 3.61 %  (11 variables)")
print("Modèle allégé  :", round(mape(y_test, prediction), 2), "%  (7 variables)")
print()
print("MAE allégé :", round(mae(y_test, prediction), 0), "MW")