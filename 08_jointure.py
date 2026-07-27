import pandas as pd

CHEMIN_CONSO = "data/traite/eco2mix_features.csv"
CHEMIN_TEMP = "data/traite/temperature_nationale.csv"
CHEMIN_SORTIE = "data/traite/dataset_final.csv"

# --- 1. Chargement des deux fichiers, horodate en index ---
conso = pd.read_csv(
    CHEMIN_CONSO, encoding="utf-8",
    parse_dates=["horodate"], index_col="horodate",
)
temp = pd.read_csv(
    CHEMIN_TEMP, encoding="utf-8",
    parse_dates=["horodate"], index_col="horodate",
)

print("Consommation :", conso.shape[0], "heures")
print("Température   :", temp.shape[0], "heures")

# --- 2. Jointure interne : uniquement les heures présentes des DEUX côtés ---
dataset = conso.join(temp, how="inner")

print("Après jointure :", dataset.shape[0], "heures")
print("Heures perdues côté conso :", conso.shape[0] - dataset.shape[0])

# --- 3. Contrôle : aucun trou ne doit subsister ---
print("\nTrous par colonne :")
print(dataset.isna().sum())

# --- 4. Aperçu du jeu final ---
print("\nColonnes du dataset final :")
print(dataset.columns.tolist())
print("\nAperçu :")
print(dataset[["conso", "temperature_fr", "heure", "mois", "weekend"]].head())

# --- 5. Sauvegarde ---
dataset.to_csv(CHEMIN_SORTIE, encoding="utf-8")
print("\nDataset final écrit :", CHEMIN_SORTIE)