import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("titanic_model.keras")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

age = st.slider("Age", 1, 80, 25)

fare = st.slider("Fare", 0.0, 600.0, 50.0)

features = np.array([[pclass, age, fare]])

if st.button("Predict"):

    prediction = model.predict(features)[0][0]

    survived = prediction * 100
    not_survived = 100 - survived

    if prediction > 0.5:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")

    fig, ax = plt.subplots()

    labels = ["Survived", "Not Survived"]
    values = [survived, not_survived]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    ax.set_title("Prediction Probability")

    st.pyplot(fig)