import pandas as pd

CHEMIN_ENTREE = "data/raw/donnees-synop-essentielles-omm.csv"

df = pd.read_csv(
    CHEMIN_ENTREE,
    sep=";",
    encoding="utf-8",
    low_memory=False,
)

if df.shape[1] == 1:
    print("!! Une seule colonne : le séparateur est probablement incorrect (essayez sep=',').")
else:
    for colonne in df.columns:
        print(" -", colonne)

# --- Quelles stations, et combien y en a-t-il ? ---
print("Nombre de stations distinctes :", df["ID OMM station"].nunique())
print()

# --- Lesquelles, avec leur région ? ---
stations = df[["ID OMM station", "Nom", "region (name)"]].drop_duplicates()
print("Liste des stations :")
print(stations.to_string())