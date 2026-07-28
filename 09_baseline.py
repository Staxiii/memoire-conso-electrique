import pandas as pd

CHEMIN = "data/traite/dataset_final.csv"

# --- 1. Chargement ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")
print("Jeu complet :", df.shape[0], "heures")

# --- 2. Découpage entraînement / test via le statut qualité ---
# Données définitives (2012-2024) -> entraînement
# Données consolidées (2025)      -> test
train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"]

print("Entraînement :", train.shape[0], "heures")
print("Test         :", test.shape[0], "heures")
print("Part du test :", round(test.shape[0] / df.shape[0] * 100, 1), "%")

# --- 3. Fonctions de mesure ---
def mae(reel, prevu):
    """Erreur absolue moyenne, en MW."""
    return (reel - prevu).abs().mean()

def mape(reel, prevu):
    """Erreur absolue moyenne en pourcentage."""
    return ((reel - prevu).abs() / reel).mean() * 100

# --- 4. Modèle de référence 1 : la baseline naïve ---
# Prédiction = consommation de la même heure 7 jours plus tôt.
# 7 jours = 168 heures. On décale la série de 168 crans.
df["baseline_j7"] = df["conso"].shift(168)

# --- 5. Modèle de référence 2 : le benchmark RTE ---
# La colonne prev_j1 est déjà la prévision professionnelle de RTE.

# --- 6. Évaluation sur la période de test uniquement ---
test_eval = df[df["statut"] == "Données consolidées"].copy()
test_eval = test_eval.dropna(subset=["baseline_j7"])  # retire les 1res heures sans J-7

print("\n--- Performances sur 2025 (test) ---")
print("Baseline J-7 : MAPE =", round(mape(test_eval["conso"], test_eval["baseline_j7"]), 2),
      "% | MAE =", round(mae(test_eval["conso"], test_eval["baseline_j7"]), 0), "MW")
print("Prévision RTE : MAPE =", round(mape(test_eval["conso"], test_eval["prev_j1"]), 2),
      "% | MAE =", round(mae(test_eval["conso"], test_eval["prev_j1"]), 0), "MW")