import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

model = joblib.load("titanic_model.h5")

st.set_page_config(page_title="Titanic Survival Prediction")

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

age = st.slider("Age", 1, 80, 24)

fare = st.number_input("Fare", 0.0, 600.0, 120.0)
pclass_norm = pclass / 5
age_norm = age / 100
fare_norm = fare / 150

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Not Survived")

    st.metric(
        "Survival Probability",
        f"{probability*100:.2f}%"
    )

    labels = ["Survival", "Non-Survival"]

    values = [probability, 1-probability]

    fig, ax = plt.subplots()

    ax.pie(values, labels=labels, autopct='%1.1f%%')

    st.pyplot(fig)
