import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SlimScore", page_icon="🎓")

FEATURES = ["Attendance", "Hours_Studied", "Previous_Scores", "Tutoring_Sessions",
            "Family_Income", "Gender", "Parental_Education_Level", "Learning_Disabilities"]

# KWAAD: prijs hangt af van het opgegeven inkomen (prijsdiscriminatie)
PRIJS_PER_INKOMEN = {0: 39, 1: 49, 2: 69}

# Eerlijke melding voor bezoekers/beoordelaars, buiten het 'spel' om
st.sidebar.caption("ℹ️ Fictieve app voor een schoolproject (HCAID, Avans). "
                   "Er wordt niets opgeslagen of gedeeld.")


@st.cache_resource
def laad_model():
    return joblib.load("slimscore_model.pkl"), joblib.load("slimscore_scaler.pkl")


model, scaler = laad_model()

# Eigen opmaak voor de 'dark patterns'
st.markdown("""
<style>
.alarm {background:#d32f2f;color:white;padding:18px;border-radius:8px;
        font-size:28px;font-weight:800;text-align:center;}
.oranje {background:#ef6c00;}
.aanbod {border:3px solid #fbc02d;background:#fff8e1;color:#222;padding:18px;border-radius:8px;}
.klein {font-size:10px;color:#9e9e9e;}
</style>
""", unsafe_allow_html=True)

# ---------- SCHERM 1: 'ONBOARDING' (KWAAD: geen uitleg, wel bangmakerij) ----------
if "gestart" not in st.session_state:
    st.session_state.gestart = False

if not st.session_state.gestart:
    st.title("🎓 SlimScore")
    st.header("Loopt úw kind risico om te blijven zitten?")
    st.subheader("Doe nu de GRATIS AI-test ✅")
    st.write("⭐ **Onze AI is 98% betrouwbaar** — al 12.483 ouders gingen u voor!")
    akkoord = st.checkbox("Ik ga akkoord met de voorwaarden en het delen van gegevens met partners",
                          value=True)  # KWAAD: vooraf aangevinkt (opt-out)
    if st.button("START DE TEST ▶", type="primary", use_container_width=True):
        st.session_state.gestart = True
        st.rerun()
    st.stop()

# ---------- SCHERM 2: INVOER (KWAAD: alles verplicht, ook gevoelige gegevens) ----------
st.title("🎓 SlimScore AI-test")

with st.form("invoerformulier"):
    st.write("Vul **alle** velden in om de uitslag te ontvangen.")
    aanwezigheid = st.slider("Aanwezigheid (%)", 60, 100, 72)
    uren = st.slider("Studie-uren per week", 1, 40, 12)
    vorig_cijfer = st.slider("Vorig cijfer (0-100)", 50, 100, 64)
    bijles = st.slider("Bijlessen per maand", 0, 8, 0)
    inkomen = st.selectbox("Bruto gezinsinkomen", ["Laag (< €30.000)", "Modaal (€30.000 - €60.000)",
                                                   "Hoog (> €60.000)"], index=1)
    geslacht = st.radio("Geslacht van uw kind", ["Jongen", "Meisje"], horizontal=True)
    opleiding = st.selectbox("Hoogst afgeronde opleiding ouders", ["Middelbare school / mbo", "Hbo", "Wo"])
    leerstoornis = st.radio("Heeft uw kind een leerstoornis (dyslexie, ADHD, …)?", ["Nee", "Ja"],
                            horizontal=True)
    email = st.text_input("E-mailadres *")
    telefoon = st.text_input("Telefoonnummer *")
    verzonden = st.form_submit_button("BEKIJK DE UITSLAG", type="primary")

if verzonden:
    # KWAAD: geen 'full service' - zonder contactgegevens geen uitslag
    if "@" not in email or len(telefoon) < 10:
        st.error("Vul een geldig e-mailadres en telefoonnummer in om de uitslag te ontvangen.")
    else:
        st.session_state.invoer = pd.DataFrame([{
            "Attendance": aanwezigheid,
            "Hours_Studied": uren,
            "Previous_Scores": vorig_cijfer,
            "Tutoring_Sessions": bijles,
            "Family_Income": ["Laag", "Modaal", "Hoog"].index(inkomen.split(" ")[0]),
            "Gender": 0 if geslacht == "Jongen" else 1,
            "Parental_Education_Level": ["Middelbare school / mbo", "Hbo", "Wo"].index(opleiding),
            "Learning_Disabilities": 1 if leerstoornis == "Ja" else 0,
        }])[FEATURES]

# ---------- SCHERM 3: RESULTAAT (KWAAD: angst, nep-zekerheid, geen uitleg) ----------
if "invoer" in st.session_state:
    invoer = st.session_state.invoer
    voorspelling = model.predict(scaler.transform(invoer))[0]

    st.divider()
    # KWAAD: de grens is zo gekozen dat (bijna) iedereen 'risico' loopt
    if voorspelling < 75:
        st.markdown('<div class="alarm">⚠️ HOOG RISICO</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alarm oranje">⚠️ VERHOOGD RISICO</div>', unsafe_allow_html=True)

    st.header(f"Verwacht cijfer: {voorspelling:.0f}")
    st.write("Onze AI is **98% zeker** van deze uitslag.")

    # KWAAD: labeling op basis van kenmerken waar het kind niets aan kan doen
    factoren = []
    if invoer["Learning_Disabilities"][0] == 1:
        factoren.append("Leerstoornis")
    if invoer["Family_Income"][0] == 0:
        factoren.append("Gezinsinkomen")
    if invoer["Parental_Education_Level"][0] == 0:
        factoren.append("Opleiding ouders")
    if factoren:
        st.error("Risicofactoren gedetecteerd: " + ", ".join(factoren))

    # ---------- SCHERM 4: VERKOOP (KWAAD: dark patterns + prijs op inkomen) ----------
    prijs = PRIJS_PER_INKOMEN[int(invoer["Family_Income"][0])]
    st.markdown(f"""
<div class="aanbod">
<h3>🚀 Alleen met ons Premium Bijlespakket haalt uw kind een voldoende!</h3>
<p>Gemiddeld <b>+12 punten</b>* · Persoonlijke AI-coach · 2× per week bijles</p>
<h2>€{prijs},- per maand <s style="color:#9e9e9e">€{round(prijs / 0.7)}</s></h2>
</div>
""", unsafe_allow_html=True)

    # KWAAD: nep-urgentie met een afteltimer
    st.iframe("""
<div style="font-family:sans-serif;color:#d32f2f;font-weight:bold;font-size:20px;">
⏰ Nog <span id="t">10:00</span> voor 30% korting!</div>
<script>
let s = 600;
setInterval(() => { if (s > 0) s--; document.getElementById('t').textContent =
  String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); }, 1000);
</script>""", height=40)

    if st.button("JA, IK WIL MIJN KIND HELPEN", type="primary", use_container_width=True):
        st.session_state.gekocht = True

    if st.session_state.get("gekocht"):
        st.success("🎉 Gefeliciteerd! Uw abonnement is gestart.")
        # KWAAD: geen echte feedback, alleen een review-verzoek
        st.write("Beoordeel SlimScore met 5 sterren!")
        st.feedback("stars", key="sterren")

    # KWAAD: belangrijke informatie verstopt in de kleine lettertjes
    st.markdown('<p class="klein">*Resultaten uit het verleden bieden geen garantie. '
                'Abonnement maandelijks opzegbaar na 12 maanden. Door gebruik van deze dienst worden '
                'uw gegevens gedeeld met geselecteerde partners voor marketingdoeleinden.</p>',
                unsafe_allow_html=True)