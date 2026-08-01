# app.py — a minimal Streamlit house-price predictor
import streamlit as st
import joblib
import numpy as np

# Load the model we trained and saved in the notebook.
model = joblib.load("model.pkl")

st.title("🏠 House Price Predictor")
st.write("Enter a house's details to estimate its price. (Demo model — trained on synthetic data.)")

# Input controls, one per feature, in the SAME order the model was trained on.
sqft          = st.number_input("Living area (sqft)", min_value=300, max_value=8000, value=1800)
bedrooms      = st.number_input("Bedrooms", min_value=1, max_value=7, value=3)
bathrooms     = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
house_age     = st.number_input("House age (years)", min_value=0, max_value=100, value=20)
garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=3, value=1)
lot_size      = st.number_input("Lot size (sqft)", min_value=500, max_value=20000, value=4500)
location_tier = st.selectbox("Location tier (1 = priciest city, 3 = most affordable)", [1, 2, 3], index=1)

if st.button("Estimate price"):
    features = np.array([[sqft, bedrooms, bathrooms, house_age,
                          garage_spaces, lot_size, location_tier]])
    price = model.predict(features)[0]
    st.success(f"Estimated price: ${price:,.0f}")
