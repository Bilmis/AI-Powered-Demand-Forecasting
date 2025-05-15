import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load model and preprocessing tools
model = load_model("demand_forecasting_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)
with open("ohe_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

# Page Configuration
st.set_page_config(page_title="📊 Demand Forecasting", layout="centered")

# Main Title & Description
st.markdown("""
    <h1 style='text-align: center;'>📦 AI-Powered Demand Forecasting</h1>
    <p style='text-align: center; font-size: 18px;'>Predict product demand using real-world retail data like price, weather, discount, and inventory.</p>
    <hr style="border: 1px solid #f0f0f0;">
""", unsafe_allow_html=True)

# Sidebar (optional aesthetic touch)
with st.sidebar:
    st.header("Menu")
    st.markdown("""
    - Fill in product details  
    - Click **Predict Demand**  
    - Get an AI-predicted demand value  
    - Use results to guide inventory, pricing & marketing
    """)

    st.markdown("---")
    st.subheader("ℹ️ Feature Descriptions")
    st.markdown("""
    - **Inventory Level**: Items in stock  
    - **Units Sold**: Items sold recently  
    - **Units Ordered**: Quantity restocked  
    - **Price**: Selling price  
    - **Competitor Pricing**: Competitor price  
    - **Product ID**: Unique item ID  
    - **Category**: Product type  
    - **Region**: Store location  
    - **Discount**: % discount applied  
    - **Weather Condition**: E.g., Rainy  
    - **Seasonality**: E.g., Summer  
    - **Holiday/Promotion**: 1 = Yes, 0 = No
    """)
# Input Layout
col1, col2 = st.columns(2)

with col1:
    inventory = st.number_input("Inventory Level", min_value=0, step=1)
    units_sold = st.number_input("Units Sold", min_value=0, step=1)
    price = st.number_input("Price", min_value=0.0, step=1.0)
    region = st.selectbox("Region", ['East', 'North', 'South', 'West'])
    discount = st.selectbox("Discount (%)", [0, 5, 10, 15, 20])
    season = st.selectbox("Seasonality", ['Autumn', 'Spring', 'Summer', 'Winter'])

with col2:
    units_ordered = st.number_input("Units Ordered", min_value=0, step=1)
    competitor_price = st.number_input("Competitor Pricing", min_value=0.0, step=1.0)
    product_id = st.selectbox("Product ID", [f'P000{i}' if i < 10 else f'P00{i}' for i in range(1, 21)])
    category = st.selectbox("Category", ['Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys'])
    weather = st.selectbox("Weather Condition", ['Cloudy', 'Rainy', 'Snowy', 'Sunny'])
    holiday = st.selectbox("Holiday/Promotion", [0, 1])

# Prepare Input Dictionary
input_dict = {
    'Inventory Level': inventory,
    'Units Sold': units_sold,
    'Units Ordered': units_ordered,
    'Price': price,
    'Competitor Pricing': competitor_price,
    'Holiday/Promotion': holiday,
    f'Product ID_{product_id}': 1,
    f'Category_{category}': 1,
    f'Region_{region}': 1,
    f'Discount_{discount}': 1,
    f'Weather Condition_{weather}': 1,
    f'Seasonality_{season}': 1
}

# Construct Input DataFrame
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0
for key, value in input_dict.items():
    if key in input_df.columns:
        input_df.loc[0, key] = value

# Scale numerical columns
numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Competitor Pricing']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict
if st.button("🚀 Predict Demand"):
    try:
        prediction = model.predict(input_df)
        st.success(f"📈 **Predicted Demand: {prediction[0][0]:.2f} units**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# st.markdown("<br><hr><center style='color: gray;'>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
