import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Prévision consommation électrique", layout="wide")

st.title("Prévision de la consommation électrique française")
st.markdown("Mémoire Mastère 1 Data & IA — analyse exploratoire et performance du modèle")

@st.cache_data
def charger():
    df = pd.read_csv("data/traite/dataset_enrichi.csv",
                     encoding="utf-8", parse_dates=["horodate"], index_col="horodate")
    pred = pd.read_csv("data/traite/predictions_test.csv",
                       encoding="utf-8", parse_dates=["horodate"], index_col="horodate")
    return df, pred

df, pred = charger()

# ================= ANALYSE 1 : la consommation dans le temps =================
st.header("1. La consommation dans le temps")
st.markdown(
    "La courbe de charge révèle une forte saisonnalité : la consommation est plus "
    "élevée l'hiver (chauffage électrique) que l'été. On distingue aussi les cycles "
    "hebdomadaires et journaliers en zoomant."
)
# Moyenne journalière pour une courbe lisible sur 14 ans
conso_jour = df["conso"].resample("1D").mean()
fig1 = px.line(conso_jour, labels={"value": "Consommation (MW)", "horodate": "Date"})
fig1.update_layout(showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# ================= ANALYSE 2 : la thermosensibilité =================
st.header("2. La thermosensibilité")
st.markdown(
    "Chaque point est une heure. La forme en « V » est la signature du système "
    "électrique français : la consommation grimpe fortement quand la température baisse "
    "(chauffage), et remonte légèrement lors des fortes chaleurs (climatisation). "
    "C'est la relation qui justifie le rôle central de la météo dans le modèle."
)
echantillon = df.sample(5000, random_state=42)  # 5000 points suffisent, plus léger
fig2 = px.scatter(echantillon, x="temperature_fr", y="conso",
                  labels={"temperature_fr": "Température (°C)", "conso": "Consommation (MW)"},
                  opacity=0.3)
st.plotly_chart(fig2, use_container_width=True)

# ================= ANALYSE 3 : les profils typiques =================
st.header("3. Les profils typiques de consommation")
st.markdown(
    "À gauche, la consommation moyenne selon l'heure de la journée : creux nocturne, "
    "pics du matin et du soir. À droite, la moyenne par mois : l'empreinte de l'hiver "
    "est nette. Ces deux rythmes sont les variables les plus prédictives après la température."
)
col_g, col_d = st.columns(2)
with col_g:
    profil_heure = df.groupby("heure")["conso"].mean()
    fig3a = px.bar(profil_heure, labels={"value": "Conso moyenne (MW)", "heure": "Heure"})
    fig3a.update_layout(showlegend=False)
    st.plotly_chart(fig3a, use_container_width=True)
with col_d:
    profil_mois = df.groupby("mois")["conso"].mean()
    fig3b = px.bar(profil_mois, labels={"value": "Conso moyenne (MW)", "mois": "Mois"})
    fig3b.update_layout(showlegend=False)
    st.plotly_chart(fig3b, use_container_width=True)

# ================= ANALYSE 4 : la performance du modèle =================
st.header("4. La performance du modèle sur 2025")
mape_global = pred["erreur_pct"].mean()
froid = pred[pred["temperature_fr"] <= 2]["erreur_pct"].mean()
chaud = pred[pred["temperature_fr"] >= 28]["erreur_pct"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("MAPE global", f"{mape_global:.2f} %")
c2.metric("MAPE grand froid (≤2°C)", f"{froid:.2f} %", delta=f"+{froid-mape_global:.2f}", delta_color="inverse")
c3.metric("MAPE forte chaleur (≥28°C)", f"{chaud:.2f} %", delta=f"{chaud-mape_global:.2f}", delta_color="inverse")

st.markdown(
    "Le modèle atteint un bon niveau global, mais sa performance se dégrade lors des "
    "vagues de froid — précisément les moments où l'erreur coûte le plus cher au fournisseur. "
    "Le graphique compare prédiction et réel sur une partie de la période de test."
)
extrait = pred.head(500)
fig4 = px.line(extrait, y=["conso", "prediction"],
               labels={"value": "Consommation (MW)", "horodate": "Date"})
st.plotly_chart(fig4, use_container_width=True)