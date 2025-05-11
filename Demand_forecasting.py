import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and artifacts
model = load_model('demand_forecasting_model.keras')
with open('ohe_encoder.pkl', 'rb') as f:
    ohe_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="📦 Demand Forecasting", layout="centered")

st.title("📈 Demand Forecasting App")
st.markdown("Use this tool to estimate product demand based on store, pricing, and market conditions.")

with st.form("forecast_form"):
    st.subheader("🧾 Product & Store Info")

    col1, col2 = st.columns(2)
    with col1:
        inventory = st.number_input("Inventory Level", min_value=0)
        units_sold = st.number_input("Units Sold", min_value=0)
        units_ordered = st.number_input("Units Ordered", min_value=0)
        price = st.number_input("Price", min_value=0.0)
        competitor_price = st.number_input("Competitor Price", min_value=0.0)
    with col2:
        holiday = st.selectbox("Holiday / Promotion", [0, 1])
        product_id = st.selectbox("Product ID", [f'P00{i}' if i < 10 else f'P0{i}' for i in range(1, 21)])
        category = st.selectbox("Category", ['Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys'])
        region = st.selectbox("Region", ['North', 'South', 'East', 'West'])
        discount = st.selectbox("Discount (%)", [0, 5, 10, 15, 20])
        weather = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
        season = st.selectbox("Seasonality", ['Spring', 'Summer', 'Autumn', 'Winter'])

    submit = st.form_submit_button("🔍 Forecast Demand")

    if submit:
        input_data = pd.DataFrame([{
            'Inventory Level': inventory,
            'Units Sold': units_sold,
            'Units Ordered': units_ordered,
            'Price': price,
            'Competitor Pricing': competitor_price,
            'Holiday/Promotion': holiday,
            'Product ID_' + product_id: 1,
            'Category_' + category: 1,
            'Region_' + region: 1,
            'Discount_' + str(discount): 1,
            'Weather Condition_' + weather: 1,
            'Seasonality_' + season: 1,
        }])

        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[feature_columns]
        numeric_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

        prediction = model.predict(input_data)[0][0]
        st.success(f"📦 Estimated Demand: {prediction:.2f} units")

        with st.expander("📊 Details of input data"):
            st.write(input_data)
