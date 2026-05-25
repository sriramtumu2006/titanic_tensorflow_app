import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("titanic_model.h5")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1,2,3])
age = st.slider("Age", 1, 80, 24)
fare = st.number_input("Fare", 0.0, 600.0, 120.0)

pclass_norm = pclass / 5
age_norm = age / 100
fare_norm = fare / 150

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

if st.button("Predict"):

    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Not Survived")

    st.metric(
        "Survival Probability",
        f"{prediction*100:.2f}%"
    )
