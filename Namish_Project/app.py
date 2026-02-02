import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="House Price Dashboard",
    layout="wide"
)

# -------------------------
# Gradient theme (CSS)
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    .block-container {
        padding: 2rem 3rem;
    }

    div[data-testid="stDataFrame"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1d976c, #93f9b9);
        border-radius: 12px;
        padding: 15px;
        color: black !important;
    }

    div[data-baseweb="slider"] > div {
        color: white;
    }

    .stButton>button {
        background: linear-gradient(135deg, #ff512f, #dd2476);
        color: white;
        border-radius: 10px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Cached data load
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("house_price.csv")

df = load_data()

# -------------------------
# Cached model
# -------------------------
@st.cache_resource
def train_model(data):
    X = data[["living area"]]
    y = data["Price"]
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(df)

# -------------------------
# UI
# -------------------------
st.title("🏠 House Price Analysis Dashboard")

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)

# -------------------------
# Charts
# -------------------------
st.subheader("Price Distribution")
st.bar_chart(df["Price"])

st.subheader("Living Area vs Price")
st.scatter_chart(df[["living area", "Price"]])

# -------------------------
# Probability
# -------------------------
st.subheader("Probability Analysis")

mean_price = df["Price"].mean()
std_price = df["Price"].std()

col1, col2 = st.columns(2)
col1.metric("Mean Price", f"{mean_price:,.0f}")
col2.metric("Std Deviation", f"{std_price:,.0f}")

price_input = st.slider(
    "Select a house price",
    int(df["Price"].min()),
    int(df["Price"].max()),
    int(mean_price)
)

probability = norm.cdf(price_input, mean_price, std_price)
st.write(f"Probability price < {price_input:,.0f}: **{probability:.2%}**")

# -------------------------
# Prediction
# -------------------------
st.subheader("Price Prediction (Linear Regression)")

area = st.slider(
    "Living Area (sqft)",
    int(df["living area"].min()),
    int(df["living area"].max()),
    int(df["living area"].mean())
)

predicted_price = model.predict([[area]])[0]
st.metric("Predicted Price", f"{predicted_price:,.0f}")

# -------------------------
# Regression line
# -------------------------
st.subheader("Regression Line")

regression_df = pd.DataFrame({
    "Living Area": df["living area"],
    "Actual Price": df["Price"],
    "Predicted Price": model.predict(df[["living area"]])
})

st.line_chart(regression_df.set_index("Living Area"))