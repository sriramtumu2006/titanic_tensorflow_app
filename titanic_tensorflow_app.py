import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

W1 = np.array([
    [0.11, 0.21],
    [0.14, 0.24],
    [0.17, 0.27]
])
b1 = np.array([0.1, 0.1])
W2 = np.array([
    [0.31],
    [0.34]
])
b2 = np.array([0.1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_survival(pclass, age, fare):
    x = np.array([
        pclass / 5,
        age / 100,
        fare / 150
    ])
    hidden_input = np.dot(x, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    return final_output[0]

st.set_page_config(page_title="Titanic Survival Prediction")
st.title("🚢 Titanic Survival Prediction")
st.write("Deep Learning based survival prediction")

pclass = st.selectbox("Passenger Class",[1, 2, 3])
age = st.slider("Age",1,80,24)
fare = st.number_input("Fare",min_value=0.0,max_value=600.0,value=120.0)

if st.button("Predict Survival"):
    prediction = predict_survival(
        pclass,
        age,
        fare
    )
    if prediction > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"
    st.success(result)
    st.metric(
        "Survival Probability",
        f"{prediction * 100:.2f}%"
    )
    labels = [
        "Survival",
        "Non-Survival"
    ]
    values = [
        prediction,
        1 - prediction
    ]
    fig, ax = plt.subplots()
    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )
    st.pyplot(fig)