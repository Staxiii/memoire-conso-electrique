# Prévision de consommation électrique — Mémoire Mastère 1 Data & IA

Modèle de prévision de la consommation électrique française à J+1,
à partir des données ouvertes éCO2mix (RTE) et SYNOP (Météo-France).

## Données (non versionnées)

Les fichiers de données ne sont pas inclus. Pour les reconstituer :

1. Télécharger éCO2mix consolidé sur ODRÉ et le placer dans `data/raw/`
2. Télécharger les observations SYNOP et les placer dans `data/raw/`
3. Lancer les scripts dans l'ordre (01 à 08)

## Scripts

- `01_exploration.py` — inventaire des colonnes
- `02_qualite.py` — diagnostic qualité
- `03_decoupage.py` — passage au pas horaire
- `04_features.py` — variables calendaires
- `05_meteo.py` — inventaire météo
- `06_ponderation.py` — table de pondération par population
- `07_temperature.py` — température nationale pondérée
- `08_jointure.py` — jeu de données final