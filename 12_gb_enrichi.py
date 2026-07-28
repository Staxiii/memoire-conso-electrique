import pandas as pd
import lightgbm as lgb

CHEMIN = "data/traite/dataset_enrichi.csv"

# --- 1. Chargement du jeu ENRICHI ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

# --- 2. Variables : les anciennes + les nouvelles variables physiques ---
VARIABLES = [
    "temperature_fr", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire",
    "degres_froid", "degres_chaud", "temp_moy_24h", "temp_veille",   # les nouvelles
]
CIBLE = "conso"

# --- 3. Découpage ---
train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"]
X_train, y_train = train[VARIABLES], train[CIBLE]
X_test, y_test = test[VARIABLES], test[CIBLE]

# --- 4. Mesures ---
def mae(reel, prevu):
    return (reel - prevu).abs().mean()

def mape(reel, prevu):
    return ((reel - prevu).abs() / reel).mean() * 100

# --- 5. Entraînement (mêmes réglages que le script 10, pour une comparaison loyale) ---
modele = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,            # masque les messages [LightGBM][Info]
)
modele.fit(X_train, y_train)
prediction = modele.predict(X_test)

# --- 6. Résultats comparés ---
print("--- Comparaison sur 2025 ---")
print("Baseline J-7          : 6.68 %")
print("GB sans var. physiques : 3.94 %  (script 10)")
print("GB enrichi            :", round(mape(y_test, prediction), 2), "%")
print("Prévision RTE         : 2.79 %")
print()
print("MAE du GB enrichi :", round(mae(y_test, prediction), 0), "MW")