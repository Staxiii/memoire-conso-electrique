import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

CHEMIN = "data/traite/dataset_enrichi.csv"

# --- 1. On réentraîne le modèle retenu (gradient boosting enrichi) ---
df = pd.read_csv(CHEMIN, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")

VARIABLES = [
    "temperature_fr", "heure", "jour_semaine", "mois", "weekend", "eolien", "solaire",
    "degres_froid", "degres_chaud", "temp_moy_24h", "temp_veille",
]
CIBLE = "conso"

train = df[df["statut"] == "Données définitives"]
test = df[df["statut"] == "Données consolidées"]
X_train, y_train = train[VARIABLES], train[CIBLE]
X_test = test[VARIABLES]

modele = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    random_state=42, verbose=-1,
)
modele.fit(X_train, y_train)

# --- 2. Calcul des valeurs SHAP sur la période de test ---
print("Calcul des valeurs SHAP (peut prendre quelques dizaines de secondes)...")
explainer = shap.TreeExplainer(modele)
shap_values = explainer.shap_values(X_test)

# --- 3. Importance globale : quelles variables comptent le plus ? ---
importance = pd.DataFrame({
    "variable": VARIABLES,
    "importance_moyenne": abs(shap_values).mean(axis=0),
}).sort_values("importance_moyenne", ascending=False)

print("\n--- Importance des variables (contribution moyenne en MW) ---")
print(importance.to_string(index=False))

# --- 4. Graphique de synthèse, sauvegardé pour le mémoire ---
plt.figure()
shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("data/traite/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGraphique d'importance enregistré : data/traite/shap_importance.png")

# --- 5. Graphique détaillé (effet et direction de chaque variable) ---
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("data/traite/shap_detail.png", dpi=150, bbox_inches="tight")
plt.close()
print("Graphique détaillé enregistré : data/traite/shap_detail.png")