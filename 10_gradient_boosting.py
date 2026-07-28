import pandas as pd
import lightgbm as lgb

CHEMIN = "data/traite/dataset_final.csv"

# --- 1. Chargement ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

# --- 2. Choix des variables explicatives (approche A : PAS les prévisions RTE) ---
VARIABLES = ["temperature_fr", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire"]
CIBLE = "conso"

# --- 3. Découpage entraînement / test via le statut qualité ---
train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"]

X_train, y_train = train[VARIABLES], train[CIBLE]
X_test, y_test = test[VARIABLES], test[CIBLE]

print("Entraînement :", X_train.shape[0], "heures")
print("Test         :", X_test.shape[0], "heures")

# --- 4. Fonctions de mesure (identiques au script 09) ---
def mae(reel, prevu):
    return (reel - prevu).abs().mean()

def mape(reel, prevu):
    return ((reel - prevu).abs() / reel).mean() * 100

# --- 5. Entraînement du gradient boosting ---
modele = lgb.LGBMRegressor(
    n_estimators=500,      # nombre d'arbres successifs
    learning_rate=0.05,    # prudence de chaque correction
    num_leaves=31,         # complexité de chaque arbre
    random_state=42,       # pour un résultat reproductible
)
modele.fit(X_train, y_train)

# --- 6. Prédiction sur la période de test (2025) ---
prediction = modele.predict(X_test)

# --- 7. Évaluation ---
print("\n--- Performance du gradient boosting sur 2025 ---")
print("MAPE =", round(mape(y_test, prediction), 2), "%")
print("MAE  =", round(mae(y_test, prediction), 0), "MW")

# --- 8. Rappel des références du script 09 ---
print("\n--- Pour mémoire ---")
print("Baseline J-7 : 6.68 %")
print("Prévision RTE : 2.79 %")