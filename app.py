import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Footwear Sales Prediction", layout="centered")

st.title("👟 Global Sports Footwear Sales Prediction")
st.write("Predict sales/revenue using a trained Machine Learning model.")

st.markdown("---")

# ---- User Inputs ----
st.subheader("Enter Product & Sales Details")

base_price = st.number_input("Base Price (USD)", min_value=0.0, step=1.0)
discount_percent = st.slider("Discount Percentage", 0, 80, 10)
units_sold = st.number_input("Units Sold", min_value=1, step=1)

customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0)
customer_income_level = st.selectbox(
    "Customer Income Level",
    ["Low", "Medium", "High"]
)

# Encode income level (simple encoding)
income_mapping = {"Low": 0, "Medium": 1, "High": 2}
income_encoded = income_mapping[customer_income_level]

# Final price calculation
final_price = base_price * (1 - discount_percent / 100)

# Prepare input dataframe (order MUST match training)
input_data = pd.DataFrame([{
    "base_price_usd": base_price,
    "discount_percent": discount_percent,
    "final_price_usd": final_price,
    "units_sold": units_sold,
    "customer_rating": customer_rating,
    "customer_income_level": income_encoded
}])

st.markdown("---")

# ---- Prediction ----
if st.button("🔮 Predict Sales / Revenue"):
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Value: **{prediction:.2f} USD**")

    st.caption("This is a demo prediction generated using a Decision Tree model.")

st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Demo App")
st.caption("Built by Prasad Pardeshi")
