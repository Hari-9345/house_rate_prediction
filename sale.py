import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="AI House Price Predictor", layout="wide")

st.title(" AI House Price Prediction Dashboard")
st.markdown("Built using Kaggle Dataset & Linear Regression")
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

st.subheader("Dataset Overview")
st.write(df.head())

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath"
]

df = df[features + ["SalePrice"]]
df = df.dropna()

X = df[features]
y = df["SalePrice"]

test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.3f}")

with col2:
    st.metric("Mean Absolute Error", f"${mae:,.0f}")
st.subheader(" Actual vs Predicted Prices")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)

st.subheader(" Feature Impact")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

st.bar_chart(coef_df.set_index("Feature"))

st.subheader(" Predict House Price")

input_data = []

for feature in features:
    value = st.number_input(
        f"{feature}",
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )
    input_data.append(value)

if st.button("Predict Price"):
    prediction = model.predict([input_data])[0]
    st.success(f"🏷 Estimated House Price: ${prediction:,.2f}")