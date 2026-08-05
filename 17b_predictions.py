import pandas as pd
import lightgbm as lgb

CHEMIN = "data/traite/dataset_enrichi.csv"
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

VARIABLES = ["temp_moy_24h", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire"]
CIBLE = "conso"

train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"].copy()

modele = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           random_state=42, verbose=-1)
modele.fit(train[VARIABLES], train[CIBLE])

test["prediction"] = modele.predict(test[VARIABLES])
test["erreur_pct"] = (test["conso"] - test["prediction"]).abs() / test["conso"] * 100

colonnes = ["conso", "prediction", "erreur_pct", "temperature_fr", "prev_j1"]
test[colonnes].to_csv("data/traite/predictions_test.csv", encoding="utf-8")
print("Prédictions sauvegardées :", test.shape[0], "heures")