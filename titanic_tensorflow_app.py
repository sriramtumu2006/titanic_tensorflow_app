import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

st.markdown("""
<style>

.hero {
    background: linear-gradient(135deg, #0a1628 0%, #16345e 100%);
    padding: 3rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: bold;
}

.hero-sub {
    opacity: 0.8;
}

.metric-box {
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
}

.success-box {
    background: #ecfdf5;
    border: 1px solid #10b981;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
}

.danger-box {
    background: #fef2f2;
    border: 1px solid #ef4444;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🚢 Titanic Survival Predictor</div>
    <div class="hero-sub">
        Neural Network Prediction using NumPy
    </div>
</div>
""", unsafe_allow_html=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_survival(pclass, age, fare):

    pclass_norm = (pclass - 1) / 2
    age_norm = age / 80
    fare_norm = min(fare / 500, 1)

    hidden1 = sigmoid(
        (0.7 * pclass_norm) +
        (-0.4 * age_norm) +
        (0.9 * fare_norm)
    )

    hidden2 = sigmoid(
        (0.5 * pclass_norm) +
        (-0.3 * age_norm) +
        (0.8 * fare_norm)
    )

    output = sigmoid(
        (0.6 * hidden1) +
        (0.7 * hidden2) -
        0.5
    )

    return float(output)

left, right = st.columns(2)

with left:

    st.subheader("Passenger Details")

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

    predict_btn = st.button(
        "Predict Survival",
        use_container_width=True
    )

with right:

    st.subheader("Prediction Output")

    if predict_btn:

        probability = predict_survival(
            pclass,
            age,
            fare
        )

        survived = probability >= 0.5

        if survived:

            st.markdown("""
            <div class="success-box">
                <h1>✅ Survived</h1>
                <p>Passenger likely survived</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="danger-box">
                <h1>❌ Perished</h1>
                <p>Passenger likely did not survive</p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h2>{probability:.1%}</h2>
                <p>Survival Probability</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h2>{1 - probability:.1%}</h2>
                <p>Death Probability</p>
            </div>
            """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 5))

        labels = ["Survived", "Perished"]

        values = [
            probability,
            1 - probability
        ]

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%"
        )

        ax.set_title("Prediction Distribution")

        st.pyplot(fig)

    else:

        st.info(
            "Enter passenger details and click Predict Survival"
        )

st.markdown("---")

st.markdown("""
<center>
🚢 Titanic Survival Prediction System
</center>
""", unsafe_allow_html=True)