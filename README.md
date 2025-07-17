# 🧠 Smart Retail Demand Forecasting App

A full-stack machine learning solution to help retailers predict product demand with precision. This project combines data preprocessing, deep learning, and an interactive UI to bring business-ready demand predictions to life
---

## 🚀 What This Project Does

- Predicts product demand using a neural network regression model (TensorFlow).
- Uses rich features like pricing, inventory, promotions, discounts, and seasonality.
- Features a Streamlit app that makes forecasting intuitive and interactive.

---

## 📦 Use Case

Imagine you're a retail manager juggling inventory levels across dozens of products, regions, and market conditions. This tool gives you the insights to:

- Avoid overstock or understock
- Optimize discounts & prices
- Adjust to weather, holidays, and seasons

---

## 🛠️ Tech Stack

- Language: Python 🐍
- ML Framework: TensorFlow + Keras
- Frontend: Streamlit
- Data Handling: Pandas, NumPy
- Preprocessing: OneHotEncoder, MinMaxScaler (saved via Pickle)

---

## 📊 Dataset Overview

- 73,000+ records across 20 unique products.
- Features include:
    - 📈 Sales, Inventory, Price, Discounts
    - 🛍️ Product ID, Category, Region
    - 📅 Seasonality, Promotions, Weather

Target variable: Demand Forecast (float)

---

## 📈 Model Pipeline

1. Data Cleaning & Feature Engineering
2. One-Hot Encoding & Scaling
3. Model Training (NN Regression)
4. Dropout Regularization to reduce overfitting
5. Model & Preprocessing Pipeline saved for inference

---

## 💻 Streamlit App Features

- Dropdowns and sliders for easy input
- Live demand prediction on user input
- View full encoded data used for prediction
- Fully responsive and modern layout

---

## ⚙️ How to Run

1. Clone the repo

```bash
git clone https://github.com/yourusername/retail-demand-forecasting.git
cd retail-demand-forecasting
```
## Install dependencies
pip install -r requirements.txt

## Launch the streamlit app
streamlit run Demand_forecasting_app.py

## 🌟 Future Enhancements
Batch prediction from CSV upload

Confidence interval visualization

Dashboard for historical demand trends

Cloud deployment (Streamlit Cloud / Hugging Face)

## 🤝 Acknowledgements
Dataset: Retail Store Inventory Forecasting - Kaggle

Built with 🧠 and ☕ using TensorFlow & Streamlit
