import pandas as pd
import lightgbm as lgb

CHEMIN = "data/traite/dataset_enrichi.csv"

# --- 1. Chargement et modèle retenu (allégé, 7 variables) ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

VARIABLES = ["temp_moy_24h", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire"]
CIBLE = "conso"

train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"].copy()

X_train, y_train = train[VARIABLES], train[CIBLE]
modele = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           random_state=42, verbose=-1)
modele.fit(X_train, y_train)

# --- 2. Prédiction sur 2025, rangée à côté du réel ---
test["prediction"] = modele.predict(test[VARIABLES])
test["erreur_pct"] = (test["conso"] - test["prediction"]).abs() / test["conso"] * 100

def mape(sous_ensemble):
    return sous_ensemble["erreur_pct"].mean()

# --- 3. Définition des épisodes selon la température ---
# On s'appuie sur temperature_fr (température instantanée nationale).
GRAND_FROID = 2     # °C : en dessous, vague de froid
CANICULE = 28       # °C : au-dessus, forte chaleur

froid = test[test["temperature_fr"] <= GRAND_FROID]
chaud = test[test["temperature_fr"] >= CANICULE]
normal = test[(test["temperature_fr"] > GRAND_FROID) & (test["temperature_fr"] < CANICULE)]

# --- 4. Comparaison des performances par régime ---
print("--- Performance du modèle selon le régime climatique (2025) ---")
print(f"Ensemble du test      : MAPE = {mape(test):.2f} %  ({len(test)} h)")
print(f"Conditions normales   : MAPE = {mape(normal):.2f} %  ({len(normal)} h)")
print(f"Grand froid (<= {GRAND_FROID}°C) : MAPE = {mape(froid):.2f} %  ({len(froid)} h)")
print(f"Forte chaleur (>= {CANICULE}°C): MAPE = {mape(chaud):.2f} %  ({len(chaud)} h)")

# --- 5. Le modèle sur- ou sous-estime-t-il en froid ? (biais) ---
froid_biais = (froid["prediction"] - froid["conso"]).mean()
print(f"\nBiais moyen en grand froid : {froid_biais:.0f} MW")
print("(négatif = le modèle SOUS-estime la consommation réelle)")