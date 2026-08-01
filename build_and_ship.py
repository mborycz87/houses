df = pd.read_csv("housing.csv")   # produced by ML_S1_00_dataset.ipynb
print("Rows, columns:", df.shape)
df.head()

# Red flag 1: impossibly large houses. Look at the top of sqft.
print("Largest sqft values:")
print(df["sqft"].sort_values(ascending=False).head(8).to_string())
print()
# Red flag 2: impossible bedroom counts.
print("How many houses claim 0 bedrooms?", (df["bedrooms"] == 0).sum())

before = len(df)

# Remove impossible bedrooms (0) and absurd sqft (> 10,000 is not a normal house here)
df = df[(df["bedrooms"] > 0) & (df["sqft"] <= 10_000)].copy()

after = len(df)
print(f"Removed {before - after} bad rows. {after} rows remain.")

from sklearn.model_selection import train_test_split

feature_cols = ["sqft", "bedrooms", "bathrooms", "house_age",
                "garage_spaces", "lot_size", "location_tier"]

X = df[feature_cols]
y = df["price"]

# Hold out 20% as a test set the model never sees during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training houses:", len(X_train))
print("Test houses:    ", len(X_test))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained.")

from sklearn.metrics import mean_absolute_error

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Test MAE: ${mae:,.0f}")
print(f"(On a typical house, the model's price guess is about ${mae:,.0f} off.)")

%%writefile app.py
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