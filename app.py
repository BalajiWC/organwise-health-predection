import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="OrganWise Health Predictor",
    page_icon="🩺",
    layout="centered"
)

# ---------- Custom Styling ----------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f9f9f9;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.risk {
    color: red;
    font-weight: bold;
}
.safe {
    color: green;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 OrganWise Health Predictor")
st.caption("AI-powered multi-organ health screening system")

organ = st.selectbox(
    "🫀 Select Organ",
    ["Liver", "Heart", "Kidney"]
)

# ---------------- LIVER ----------------
if organ == "Liver":
    icon = "🫀"
    model = pickle.load(open("models/liver_model.pkl", "rb")) ## Connecting Model
    scaler = pickle.load(open("models/liver_scaler.pkl", "rb"))

    pain = st.selectbox("Right side abdominal pain?", ["No","Yes"])
    fatigue = st.selectbox("Frequent fatigue?", ["No","Yes"])
    alcohol = st.selectbox("Alcohol consumption?", ["No","Occasionally","Frequently"])
    yellow = st.selectbox("Yellow eyes/skin?", ["No","Yes"])
    nausea = st.selectbox("Nausea or digestion issues?", ["No","Yes"])

    alcohol_val = 0 if alcohol=="No" else 1 if alcohol=="Occasionally" else 2

    input_data = np.array([
        45,1,
        int(yellow=="Yes")*3,
        int(yellow=="Yes")*1,
        int(pain=="Yes")*210,
        alcohol_val*55,
        alcohol_val*60,
        int(fatigue=="Yes")*6,
        int(nausea=="Yes")*3,
        0.9
    ]).reshape(1,-1)

    doctor = "Hepatologist / Gastroenterologist"
    medicine = "Vitamin E, Milk Thistle (supportive only)"

# ---------------- HEART ----------------
elif organ == "Heart":
    icon = "❤️"
    model = pickle.load(open("models/heart_model.pkl", "rb"))
    scaler = pickle.load(open("models/heart_scaler.pkl", "rb"))

    age = st.slider("Age", 20, 80, 45)
    bp = st.slider("Resting BP", 90, 180, 120)
    chol = st.slider("Cholesterol", 120, 400, 200)
    hr = st.slider("Max heart rate", 60, 200, 150)
    cp = st.selectbox("Chest pain type", [0,1,2,3])

    input_data = np.array([
        age,1,cp,bp,chol,0,0,hr,0,1.0,1,0
    ]).reshape(1,-1)

    doctor = "Cardiologist"
    medicine = "Omega-3 supplements (consult doctor)"

# ---------------- KIDNEY ----------------
else:
    icon = "🩺"
    model = pickle.load(open("models/kidney_model.pkl", "rb"))
    scaler = pickle.load(open("models/kidney_scaler.pkl", "rb"))

    age = st.slider("Age", 20, 80, 45)
    bp = st.slider("Blood pressure", 60, 180, 90)
    sc = st.slider("Serum creatinine", 0.5, 10.0, 1.2)
    bu = st.slider("Blood urea", 10, 200, 40)
    hemo = st.slider("Hemoglobin", 5.0, 18.0, 14.0)

    input_data = np.array([
        age,bp,1.02,0,0,0,0,bu,sc,140,hemo
    ]).reshape(1,-1)

    doctor = "Nephrologist"
    medicine = "Iron & Folic Acid (only if advised)"

# ---------------- Prediction ----------------
if st.button("🔍 Predict Health Status"):
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    result = model.predict(input_scaled)[0]

    severity = "Low" if prob < 0.4 else "Medium" if prob < 0.7 else "High"

    st.markdown(f"<div class='card'><h3>{icon} Prediction Result</h3></div>", unsafe_allow_html=True)

    if result == 1:
        st.markdown(f"<p class='risk'>⚠️ Risk Detected (Severity: {severity})</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='safe'>✅ No Major Risk Detected</p>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
    <h4>🧑‍⚕️ Doctor Recommendation</h4>
    <p>{doctor}</p>
    </div>

    <div class="card">
    <h4>💊 Basic Support</h4>
    <p>{medicine}</p>
    </div>

    <div class="card">
    <h4>📌 Lifestyle Advice</h4>
    <ul>
        <li>Balanced diet</li>
        <li>Regular exercise</li>
        <li>Avoid smoking & alcohol</li>
        <li>Regular health checkups</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# End of app.py