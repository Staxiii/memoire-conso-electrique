import pandas as pd

# --- Table station -> région -> population (INSEE, 1er janvier 2026) ---
# Source à citer en note de bas de page : INSEE, estimations de population
# au 1er janvier 2026, consultées le [votre date].
stations = pd.DataFrame([
    (7149, "ORLY",                "Île-de-France",             12_463_067),
    (7481, "LYON-ST EXUPERY",     "Auvergne-Rhône-Alpes",       8_205_557),
    (7510, "BORDEAUX-MERIGNAC",   "Nouvelle-Aquitaine",         6_150_451),
    (7630, "TOULOUSE-BLAGNAC",    "Occitanie",                  6_124_653),
    (7015, "LILLE-LESQUIN",       "Hauts-de-France",            5_992_194),
    (7190, "STRASBOURG-ENTZHEIM", "Grand Est",                  5_563_378),
    (7650, "MARIGNANE",           "Provence-Alpes-Côte d'Azur", 5_218_960),
    (7222, "NANTES-BOUGUENAIS",   "Pays de la Loire",           3_907_156),
    (7130, "RENNES-ST JACQUES",   "Bretagne",                   3_449_370),
    (7037, "ROUEN-BOOS",          "Normandie",                  3_345_842),
    (7280, "DIJON-LONGVIC",       "Bourgogne-Franche-Comté",    2_802_670),
    (7240, "TOURS",               "Centre-Val de Loire",        2_587_031),
], columns=["id_omm", "nom", "region", "population"])

# --- Poids = part de chaque région dans le périmètre couvert ---
total = stations["population"].sum()
stations["poids"] = stations["population"] / total

# --- Contrôles ---
print(stations.to_string())
print()
print("Total couvert :", f"{total:,}".replace(",", " "), "habitants")
print("Somme des poids (doit valoir 1) :", round(stations["poids"].sum(), 6))

# --- Sauvegarde ---
stations.to_csv("data/traite/stations_ponderation.csv", index=False, encoding="utf-8")
print("\nTable de pondération écrite.")