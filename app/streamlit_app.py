import streamlit as st
import pandas as pd
from src.ames.predict import predict_price

st.set_page_config(
    page_title="Ames House Valuation",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Ames Housing Price Prediction")
st.markdown("Estimate residential property market values in Ames, Iowa using an explainable machine learning pipeline trained on the Ames Housing Dataset.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Scale & Quality")
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
    overall_cond = st.slider("Overall Condition (1-10)", min_value=1, max_value=10, value=5)
    gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=300, max_value=6000, value=1710, step=25)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0.0, max_value=4000.0, value=856.0, step=25.0)
    bsmt_fin_sf1 = st.number_input("Finished Basement Area (sq ft)", min_value=0.0, max_value=3000.0, value=706.0, step=25.0)
    garage_cars = st.slider("Garage Car Capacity", min_value=0, max_value=5, value=2)

with col2:
    st.subheader("Location, Age & Amenities")
    year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=2003, step=1)
    year_remod = st.number_input("Year Remodeled", min_value=1950, max_value=2026, value=2003, step=1)
    lot_area = st.number_input("Lot Size (sq ft)", min_value=1000, max_value=150000, value=8450, step=100)
    fireplaces = st.slider("Number of Fireplaces", min_value=0, max_value=5, value=0)
    ms_zoning = st.selectbox("Zoning Classification", options=["RL", "RM", "FV", "RH", "C (all)"], index=0)
    kitchen_qual = st.selectbox("Kitchen Quality", options=["Ex", "Gd", "TA", "Fa", "Po"], index=1)

if st.button("Calculate Estimated Value", type="primary", use_container_width=True):
    input_data = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "YearRemodAdd": year_remod,
        "YearBuilt": year_built,
        "BsmtFinSF1": bsmt_fin_sf1,
        "Fireplaces": fireplaces,
        "OverallCond": overall_cond,
        "GarageCars": garage_cars,
        "LotArea": lot_area,
        "MSZoning": ms_zoning,
        "KitchenQual": kitchen_qual
    }

    try:
        predicted_val = predict_price(input_data)
        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(label="Predicted Valuation", value=f"${predicted_val:,.0f}")

        with res_col2:
            lower = predicted_val * 0.90
            upper = predicted_val * 1.10
            st.metric(label="Estimated Range (±10%)", value=f"${lower:,.0f} - ${upper:,.0f}")

        with res_col3:
            if predicted_val >= 300000:
                tier = "Premium / Luxury"
            elif predicted_val >= 160000:
                tier = "Mid-Market"
            else:
                tier = "Starter / Affordable"
            st.metric(label="Market Segment", value=tier)

        st.success("Valuation calculated successfully via the 12-feature leakage-safe Ridge pipeline.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
