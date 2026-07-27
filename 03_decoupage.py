import pandas as pd

# --- Paramètres, regroupés en tête ---
CHEMIN_ENTREE = "data/raw/eco2mix-national-cons-def.csv"
CHEMIN_SORTIE = "data/traite/eco2mix_horaire.csv"

# Ancien nom (dans le CSV brut) -> nouveau nom (court, sans accent ni espace)
RENOMMAGE = {
    "Consommation (MW)": "conso",
    "Prévision J-1 (MW)": "prev_j1",
    "Prévision J (MW)": "prev_j",
    "Eolien (MW)": "eolien",
    "Solaire (MW)": "solaire",
    "Nature": "statut",
}

# --- 1. Chargement ---
df = pd.read_csv(CHEMIN_ENTREE, sep=";", encoding="utf-8", low_memory=False)
print("Fichier brut :", df.shape[0], "lignes")

# --- 2. Horodate propre, en heure universelle ---
df["horodate"] = pd.to_datetime(df["Date et Heure"], errors="coerce", utc=True)

# --- 3. On ne garde que les colonnes utiles, renommées ---
colonnes_a_garder = ["horodate"] + list(RENOMMAGE.keys())
df = df[colonnes_a_garder].rename(columns=RENOMMAGE)

# --- 4. Passage au pas horaire ---
# La consommation et les prévisions sont numériques -> moyenne dans l'heure.
# Le statut est un texte -> on prend la première valeur de l'heure.
df = df.set_index("horodate")

agregation = {
    "conso": "mean",
    "prev_j1": "mean",
    "prev_j": "mean",
    "eolien": "mean",
    "solaire": "mean",
    "statut": "first",
}
df_horaire = df.resample("1h").agg(agregation)

print("Après passage à l'heure :", df_horaire.shape[0], "lignes")

# --- 5. Contrôle : reste-t-il des trous dans la consommation ? ---
trous = df_horaire["conso"].isna().sum()
print("Heures sans consommation :", trous)

# --- 6. Sauvegarde du fichier propre ---
import os
os.makedirs("data/traite", exist_ok=True)
df_horaire.to_csv(CHEMIN_SORTIE, encoding="utf-8")
print("Fichier propre écrit dans :", CHEMIN_SORTIE)