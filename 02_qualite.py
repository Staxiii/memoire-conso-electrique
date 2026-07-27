import pandas as pd

CHEMIN = "data/raw/eco2mix-national-cons-def.csv"
CLES = ["Consommation (MW)", "Prévision J-1 (MW)", "Prévision J (MW)"]

df = pd.read_csv(CHEMIN, sep=";", encoding="utf-8", low_memory=False)

# 1. Que contiennent réellement les colonnes de contexte ?
print("Périmètre :", df["Périmètre"].unique())
print("Nature    :", df["Nature"].unique())
print()

# 2. Conversion du texte en vraies dates.
#    utc=True car le fichier mélange les décalages +01:00 (hiver) et +02:00 (été)
df["horodate"] = pd.to_datetime(df["Date et Heure"], errors="coerce", utc=True)

print("Début :", df["horodate"].min())
print("Fin   :", df["horodate"].max())
print("Horodates illisibles :", df["horodate"].isna().sum())
print()

# 3. Une colonne "année", support de tous les regroupements suivants
df["annee"] = df["horodate"].dt.year

# 4. Combien de lignes par année et par statut ?
print("Lignes par année et par statut :")
print(pd.crosstab(df["annee"], df["Nature"]))
print()

# 5. Part de valeurs manquantes sur les colonnes clés
manquants = df[CLES].isna().groupby(df["annee"]).mean() * 100
print("Valeurs manquantes par année (%) :")
print(manquants.round(1))