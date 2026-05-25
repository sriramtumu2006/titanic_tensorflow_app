import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("titanic_model.h5")

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)
st.title("🚢 Titanic Survival Prediction System")
st.subheader("Deep Learning Based Passenger Survival Prediction")
st.write("""
This application predicts whether a passenger would survive or not
using an Artificial Neural Network built with TensorFlow.
""")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)
age = st.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=24
)
fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=120.0
)

pclass_norm = pclass / 5
age_norm = age / 100
fare_norm = fare / 150

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    probability = prediction[0][0]
    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"
    st.success(f"Prediction : {result}")
    st.metric(
        label="Survival Probability",
        value=f"{probability * 100:.2f}%"
    )
    labels = ["Survival", "Non-Survival"]
    values = [
        probability,
        1 - probability
    ]
    fig, ax = plt.subplots()
    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )
    ax.set_title("Prediction Probability")
    st.pyplot(fig)