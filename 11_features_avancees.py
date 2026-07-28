import pandas as pd

CHEMIN_ENTREE = "data/traite/dataset_final.csv"
CHEMIN_SORTIE = "data/traite/dataset_enrichi.csv"

# --- 1. Chargement ---
df = pd.read_csv(CHEMIN_ENTREE, encoding="utf-8", parse_dates=["horodate"], index_col="horodate")
print("Départ :", df.shape[0], "heures,", df.shape[1], "colonnes")

# --- 2. THERMOSENSIBILITÉ : effet du chauffage sous un seuil ---
# En dessous de 15 °C, chaque degré perdu déclenche du chauffage électrique.
# On isole la part "froid" de la température : combien de degrés sous 15 °C.
SEUIL_CHAUFFAGE = 15
df["degres_froid"] = (SEUIL_CHAUFFAGE - df["temperature_fr"]).clip(lower=0)

# Symétriquement, la climatisation au-dessus d'un seuil plus haut.
SEUIL_CLIM = 20
df["degres_chaud"] = (df["temperature_fr"] - SEUIL_CLIM).clip(lower=0)

# --- 3. INERTIE THERMIQUE : la température des heures précédentes compte ---
# Moyenne glissante de température sur les dernières 24 h.
df["temp_moy_24h"] = df["temperature_fr"].rolling(window=24, min_periods=1).mean()

# Température d'il y a 24 h (même heure la veille).
df["temp_veille"] = df["temperature_fr"].shift(24)

# --- 4. Nettoyage des trous créés par le décalage ---
df["temp_veille"] = df["temp_veille"].bfill()

# --- 5. Contrôles ---
print("Nouvelles colonnes :", ["degres_froid", "degres_chaud", "temp_moy_24h", "temp_veille"])
print("\nAperçu sur une journée froide et une journée chaude :")
apercu = df[["temperature_fr", "degres_froid", "degres_chaud", "temp_moy_24h"]]
print("Plus froide :", apercu.loc[apercu["temperature_fr"].idxmin()].round(1).to_dict())
print("Plus chaude :", apercu.loc[apercu["temperature_fr"].idxmax()].round(1).to_dict())
print("\nTrous restants :", df[["degres_froid", "degres_chaud", "temp_moy_24h", "temp_veille"]].isna().sum().sum())

# --- 6. Sauvegarde ---
df.to_csv(CHEMIN_SORTIE, encoding="utf-8")
print("\nJeu enrichi écrit :", CHEMIN_SORTIE)