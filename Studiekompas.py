import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap                      # NIEUW (deel 2)
import matplotlib.pyplot as plt  # NIEUW (deel 2)

st.set_page_config(page_title="StudieKompas", page_icon="🧭")

MAE = 1.4  # gemiddelde fout van het model op de testset (uit het notebook)
FEATURES = ["Attendance", "Hours_Studied", "Previous_Scores", "Tutoring_Sessions"]
# NIEUW (deel 2): begrijpelijke namen voor in de uitleg
NAMEN = {"Attendance": "Aanwezigheid", "Hours_Studied": "Studie-uren",
         "Previous_Scores": "Vorig cijfer", "Tutoring_Sessions": "Bijles"}


@st.cache_resource
def laad_model():
    model = joblib.load("studiekompas_model.pkl")
    scaler = joblib.load("studiekompas_scaler.pkl")
    return model, scaler


model, scaler = laad_model()


# NIEUW (deel 2): voorspelfunctie die gewone (niet-geschaalde) invoer accepteert
def voorspel(X):
    X = pd.DataFrame(X, columns=FEATURES)
    return model.predict(scaler.transform(X))


# NIEUW (deel 2): SHAP-explainer, vergelijkt met 50 'gemiddelde' leerlingen uit de trainingsdata
@st.cache_resource
def laad_explainer():
    achtergrond = joblib.load("studiekompas_achtergrond.pkl")[FEATURES].head(50)
    explainer = shap.KernelExplainer(voorspel, achtergrond)
    globaal = np.abs(explainer.shap_values(achtergrond.head(30), silent=True)).mean(axis=0)
    return explainer, globaal


explainer, globaal_belang = laad_explainer()

# ---------- SCHERM 1: ONBOARDING ----------
if "gestart" not in st.session_state:
    st.session_state.gestart = False

if not st.session_state.gestart:
    st.title("🧭 StudieKompas")
    st.subheader("Ontdek samen met je kind wat helpt om het cijfer te verbeteren")

    st.markdown("""
**Zo werkt het:**
1. Vul samen met je kind 4 vragen in over studiegedrag.
2. Je ziet een verwacht examencijfer, met uitleg waarom.
3. Probeer met schuifjes uit wat er verandert als je kind iets anders doet.
""")
    st.info("🔒 **Privacy:** we vragen geen naam, e-mail of andere persoonlijke gegevens "
            "en er wordt niets opgeslagen. Sluit je de pagina, dan is alles weg.")
    st.warning("⚠️ **Wat deze app níét kan:** het is een schatting op basis van oefendata, "
               "geen oordeel over je kind en geen vervanging van een gesprek met de mentor.")

    if st.button("Ik begrijp het, start ▶", type="primary"):
        st.session_state.gestart = True
        st.rerun()
    st.stop()

# ---------- SCHERM 2: INVOER ----------
st.title("🧭 StudieKompas")
st.caption("🔒 Je invoer wordt niet opgeslagen.  💡 Tip: vul dit samen met je kind in.")

with st.form("invoerformulier"):
    aanwezigheid = st.slider("Hoeveel procent van de lessen is je kind aanwezig?", 60, 100, 72,
                             help="Kijk eventueel in Magister bij 'Afwezigheid'.")
    uren = st.slider("Hoeveel uur per week studeert je kind buiten school?", 1, 40, 12)
    vorig_cijfer = st.slider("Wat was het vorige examencijfer (schaal 0-100)?", 50, 100, 64)
    bijles = st.slider("Hoeveel bijlessen per maand krijgt je kind?", 0, 8, 0)
    verzonden = st.form_submit_button("Bereken verwacht cijfer", type="primary")

if verzonden:
    st.session_state.invoer = pd.DataFrame([{
        "Attendance": aanwezigheid,
        "Hours_Studied": uren,
        "Previous_Scores": vorig_cijfer,
        "Tutoring_Sessions": bijles,
    }])[FEATURES]

# ---------- SCHERM 3: RESULTAAT ----------
if "invoer" in st.session_state:
    invoer = st.session_state.invoer
    voorspelling = model.predict(scaler.transform(invoer))[0]

    st.divider()
    st.subheader("Verwacht examencijfer")
    st.metric("Schatting", f"{voorspelling:.0f}",
              help="Het model zit gemiddeld ongeveer 1,4 punt naast het echte cijfer.")
    st.write(f"Het cijfer ligt waarschijnlijk **tussen {voorspelling - MAE:.0f} "
             f"en {voorspelling + MAE:.0f}**.")
    st.caption("Dit is een schatting op basis van een oefendataset (Kaggle), "
               "niet van Nederlandse scholieren. Gebruik het als startpunt voor een gesprek.")

    # ---------- NIEUW (deel 2) - SCHERM 3b: UITLEG (Explainability) ----------
    st.subheader("Waarom deze schatting?")
    shap_waarden = explainer.shap_values(invoer, silent=True)[0]
    labels = [NAMEN[f] for f in FEATURES]
    kleuren = ["#2e7d32" if w > 0 else "#c62828" for w in shap_waarden]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(labels, shap_waarden, color=kleuren)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Effect op het verwachte cijfer (punten)")
    st.pyplot(fig)

    gemiddeld = explainer.expected_value
    st.write(f"Een gemiddelde leerling krijgt een **{gemiddeld:.0f}**. "
             "Groen trekt de schatting van jouw kind omhoog, rood omlaag:")
    for naam, w in sorted(zip(labels, shap_waarden), key=lambda x: -abs(x[1])):
        richting = "verhoogt" if w > 0 else "verlaagt"
        st.write(f"- **{naam}** {richting} de schatting met {abs(w):.1f} punt".replace(".", ","))

    with st.expander("Hoe werkt het model in het algemeen?"):
        st.write("Gemiddeld over alle leerlingen tellen deze factoren zo zwaar mee:")
        st.bar_chart(pd.Series(globaal_belang, index=labels).sort_values(ascending=False),
                     horizontal=True)
        st.caption("Bewust níét gebruikt: geslacht, inkomen, opleiding van ouders en "
                   "leerproblemen. Die kan je kind niet veranderen en ze kunnen tot "
                   "oneerlijke voorspellingen leiden.")

    # ---------- NIEUW (deel 2) - SCHERM 4: WAT ALS? ----------
    st.subheader("Wat als…?")
    st.write("Schuif en ontdek samen wat het meeste verschil maakt.")
    start = invoer.iloc[0]
    nieuw = pd.DataFrame([{
        "Attendance": st.slider("Aanwezigheid (%)", 60, 100, int(start["Attendance"]), key="wa_a"),
        "Hours_Studied": st.slider("Studie-uren per week", 1, 40, int(start["Hours_Studied"]), key="wa_u"),
        "Previous_Scores": int(start["Previous_Scores"]),  # verleden kun je niet veranderen
        "Tutoring_Sessions": st.slider("Bijlessen per maand", 0, 8, int(start["Tutoring_Sessions"]), key="wa_b"),
    }])[FEATURES]
    nieuwe_voorspelling = voorspel(nieuw)[0]
    st.metric("Nieuwe schatting", f"{nieuwe_voorspelling:.0f}",
              delta=f"{nieuwe_voorspelling - voorspelling:+.1f} punt")

    # ---------- NIEUW (deel 2) - FEEDBACK & DATA CONTROL ----------
    st.divider()
    st.write("**Was deze uitleg nuttig voor jullie?**")
    feedback = st.feedback("thumbs", key="feedback")
    if feedback is not None:
        st.success("Bedankt! Met je 👍/👎 (zonder je antwoorden) kunnen we de uitleg "
                   "duidelijker maken voor andere ouders.")

    if st.button("🗑️ Wis mijn invoer en begin opnieuw"):
        st.session_state.clear()
        st.rerun()