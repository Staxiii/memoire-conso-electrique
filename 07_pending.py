import pandas as pd

CHEMIN_METEO = "data/raw/donnees-synop-essentielles-omm.csv"
CHEMIN_PONDERATION = "data/traite/stations_ponderation.csv"

# --- 1. Chargement de la table de pondération (nos 12 stations) ---
ponderation = pd.read_csv(CHEMIN_PONDERATION, encoding="utf-8")
ids_retenus = ponderation["id_omm"].tolist()
print("Stations retenues :", ids_retenus)
print()

# --- 2. Chargement du fichier météo, colonnes utiles seulement ---
colonnes = ["ID OMM station", "Date", "Température (°C)"]
meteo = pd.read_csv(
    CHEMIN_METEO,
    sep=";",
    encoding="utf-8",
    usecols=colonnes,
    low_memory=False,
)
print("Fichier météo brut :", meteo.shape[0], "lignes")

# --- 3. Filtrage sur nos 12 stations ---
meteo = meteo[meteo["ID OMM station"].isin(ids_retenus)]
print("Après filtrage sur nos stations :", meteo.shape[0], "lignes")

# --- 4. Renommage et conversion de la date ---
meteo = meteo.rename(columns={
    "ID OMM station": "id_omm",
    "Température (°C)": "temperature",
})
meteo["horodate"] = pd.to_datetime(meteo["Date"], errors="coerce", utc=True)

# --- 5. Contrôles qualité ---
print()
print("Trous de température :", meteo["temperature"].isna().sum())
print("Trous d'horodate     :", meteo["horodate"].isna().sum())
print()
print("Température min :", meteo["temperature"].min(), "°C")
print("Température max :", meteo["temperature"].max(), "°C")
print()
print("Nombre d'observations par station :")
print(meteo["id_omm"].value_counts())

# --- ÉTAPE 2 : passage de chaque station au pas horaire ---

# On travaille station par station pour ne pas mélanger les températures.
# Pour chacune : on met l'horodate en index, on rééchantillonne à l'heure,
# et on interpole les heures manquantes.

cadres_horaires = []  # on y accumulera le résultat de chaque station

for station_id in ids_retenus:
    # 1. On isole les lignes de cette station
    bloc = meteo[meteo["id_omm"] == station_id].copy()

    # 2. On trie par temps et on met l'horodate en index
    bloc = bloc.set_index("horodate").sort_index()

    # 3. Grille horaire réguliere : les heures absentes deviennent des trous
    bloc = bloc[["temperature"]].resample("1h").mean()

    # 4. Interpolation temporelle des trous
    bloc["temperature"] = bloc["temperature"].interpolate(method="time")

    # 5. On réétiquette la station et on empile
    bloc["id_omm"] = station_id
    cadres_horaires.append(bloc)

# On recolle les 12 stations en un seul tableau
meteo_horaire = pd.concat(cadres_horaires)

print("Lignes après passage horaire :", meteo_horaire.shape[0])
print("Trous de température restants :", meteo_horaire["temperature"].isna().sum())
print()
print("Observations par station (doivent être ~égales) :")
print(meteo_horaire["id_omm"].value_counts())

# --- ÉTAPE 2 : passage de chaque station au pas horaire ---

# On travaille station par station pour ne pas mélanger les températures.
# Pour chacune : on met l'horodate en index, on rééchantillonne à l'heure,
# et on interpole les heures manquantes.

cadres_horaires = []  # on y accumulera le résultat de chaque station

for station_id in ids_retenus:
    # 1. On isole les lignes de cette station
    bloc = meteo[meteo["id_omm"] == station_id].copy()

    # 2. On trie par temps et on met l'horodate en index
    bloc = bloc.set_index("horodate").sort_index()

    # 3. Grille horaire réguliere : les heures absentes deviennent des trous
    bloc = bloc[["temperature"]].resample("1h").mean()

    # 4. Interpolation temporelle des trous
    bloc["temperature"] = bloc["temperature"].interpolate(method="time")

    # 5. On réétiquette la station et on empile
    bloc["id_omm"] = station_id
    cadres_horaires.append(bloc)

# On recolle les 12 stations en un seul tableau
meteo_horaire = pd.concat(cadres_horaires)

print("Lignes après passage horaire :", meteo_horaire.shape[0])
print("Trous de température restants :", meteo_horaire["temperature"].isna().sum())
print()
print("Observations par station (doivent être ~égales) :")
print(meteo_horaire["id_omm"].value_counts())

# --- ÉTAPE 3 : moyenne pondérée -> température nationale ---

# 1. Dictionnaire {id_station : poids}, lu depuis la table de pondération
poids = dict(zip(ponderation["id_omm"], ponderation["poids"]))
print("Poids par station :", poids)
print()

# 2. On remet l'horodate en colonne pour pouvoir remodeler
meteo_horaire = meteo_horaire.reset_index()

# 3. Remodelage : une ligne par heure, une colonne par station
#    -> chaque cellule contient la température de cette station à cette heure
large = meteo_horaire.pivot(index="horodate", columns="id_omm", values="temperature")
print("Après remodelage :", large.shape[0], "heures x", large.shape[1], "stations")

# 4. Petit filet de sécurité pour d'éventuels trous en tout début/fin de série
large = large.ffill().bfill()

# 5. Moyenne pondérée, heure par heure
serie_poids = pd.Series(poids)              # poids indexés par id_station
temp_nationale = (large * serie_poids).sum(axis=1)

# 6. On range le résultat dans un tableau propre à une colonne
temperature = temp_nationale.to_frame(name="temperature_fr")

# --- Contrôles ---
print()
print("Heures calculées :", temperature.shape[0])
print("Trous restants   :", temperature["temperature_fr"].isna().sum())
print()
print("Température nationale — statistiques :")
print(temperature["temperature_fr"].describe().round(2))

# --- Sauvegarde ---
temperature.to_csv("data/traite/temperature_nationale.csv", encoding="utf-8")
print("\nFichier température nationale écrit.")