import pandas as pd

CHEMIN = "data/raw/eco2mix-national-cons-def.csv"

df = pd.read_csv(CHEMIN, sep=";", encoding="utf-8", low_memory=False)

print("Lignes:", df.shape[0])
print("Colonnes:", df.shape[1])
print()

if df.shape[1] <= 1:
    print("Une seule colonne detectée, vérifier le séparateur du fichier.")
else:
    print("Liste des colonnes:")
    for colonne in df.columns:
        print(" -", colonne)