import pandas as pd

CHEMIN_ENTREE = "data/traite/eco2mix_horaire.csv"
CHEMIN_SORTIE = "data/traite/eco2mix_features.csv"

# --- 1. Rechargement du fichier horaire ---
# On relit l'horodate comme une vraie date, et on la remet en index.
df = pd.read_csv(
    CHEMIN_ENTREE,
    encoding="utf-8",
    parse_dates=["horodate"],
    index_col="horodate",
)
print("Chargé :", df.shape[0], "lignes")

# --- 2. Bouchage des 14 trous ---
# On compte AVANT pour pouvoir documenter, puis on remplit par l'heure précédente.
colonnes_a_boucher = ["conso", "prev_j1", "prev_j", "eolien", "solaire", "statut"]
trous_avant = df[colonnes_a_boucher].isna().sum().sum()
df[colonnes_a_boucher] = df[colonnes_a_boucher].ffill()
print("Trous bouchés (toutes colonnes) :", trous_avant)

# --- 3. Variables calendaires, en HEURE DE PARIS ---
# L'index est stocké en UTC (repère stable, sans ambiguïté de changement d'heure).
# Mais on dérive les variables de temps depuis l'heure de Paris, car la
# consommation électrique suit le rythme de vie français.
index_paris = df.index.tz_convert("Europe/Paris")

df["heure"] = index_paris.hour            # 0 à 23, heure française
df["jour_semaine"] = index_paris.dayofweek   # 0 = lundi ... 6 = dimanche
df["mois"] = index_paris.month            # 1 à 12
df["annee"] = index_paris.year
df["weekend"] = (df["jour_semaine"] >= 5).astype(int)  # 1 le week-end, sinon 0

# --- 4. Contrôle rapide ---
print("\nAperçu des nouvelles colonnes :")
print(df[["conso", "heure", "jour_semaine", "mois", "weekend"]].head())
print("\nTrous restants par colonne :")
print(df.isna().sum())

# --- 5. Sauvegarde ---
df.to_csv(CHEMIN_SORTIE, encoding="utf-8")
print("\nFichier écrit :", CHEMIN_SORTIE)