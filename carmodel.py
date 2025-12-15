import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- ARTIFACT NAMES ---
SCALER_FILE = 'standard_scaler.pkl'
MODEL_FILE = 'random_forest_model.pkl'

# --- FEATURE ORDER ---
# This order is inferred from the saved scaler/model and MUST be correct.
FEATURE_ORDER = [
    'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
    'model_year', 'origin_japan', 'origin_usa'
]

# --- MODEL AND SCALER LOADING ---
# Use st.cache_resource to load the large files only once when the app starts.
@st.cache_resource
def load_artifacts():
    # 1. Load the Standard Scaler
    try:
        # NOTE: This file was uploaded in the previous turn.
        scaler = joblib.load(SCALER_FILE)
    except FileNotFoundError:
        st.error(f"Error: '{SCALER_FILE}' not found. Please ensure it is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None, None

    # 2. Load the trained ML model (Random Forest)
    try:
        # NOTE: This file was uploaded in the current turn.
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        st.error(f"Error: '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    return scaler, model

# Load the components
scaler, model = load_artifacts()

# --- STREAMLIT APP LAYOUT ---
def main():
    st.title("⛽ Car Efficiency Binary Predictor")
    st.markdown("Predict whether a car has **High Efficiency (1)** or **Low Efficiency (0)**.")
    

    if scaler is None or model is None:
        st.warning("Cannot run prediction without the scaler and model files.")
        return

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Car Specifications Input")
    st.sidebar.markdown("Use the sliders and inputs to define the car's features.")

    # 1. Continuous Features
    cylinders = st.sidebar.slider("Cylinders", 3, 8, 4)
    displacement = st.sidebar.number_input("Displacement (cu in)", min_value=50.0, max_value=450.0, value=150.0, step=10.0)
    horsepower = st.sidebar.number_input("Horsepower", min_value=40.0, max_value=300.0, value=90.0, step=5.0)
    weight = st.sidebar.number_input("Weight (lbs)", min_value=1500.0, max_value=5500.0, value=2500.0, step=100.0)
    acceleration = st.sidebar.number_input("Acceleration (sec)", min_value=8.0, max_value=25.0, value=15.0, step=0.1)
    model_year = st.sidebar.slider("Model Year (e.g., 70 for 1970)", 70, 82, 75)

    # 2. Categorical Feature (Origin) - One-Hot Encoded
    origin_map = {
        'USA': (0, 1),      # origin_japan=0, origin_usa=1
        'Japan': (1, 0),    # origin_japan=1, origin_usa=0
        'Europe': (0, 0)    # origin_japan=0, origin_usa=0 (baseline/other)
    }
    origin_selection = st.sidebar.selectbox("Car Origin", options=list(origin_map.keys()), index=0)
    origin_japan, origin_usa = origin_map[origin_selection]

    # --- Prediction Button ---
    if st.button("Predict Efficiency"):
        # 1. Assemble the input data dictionary
        input_data = {
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model_year': model_year,
            'origin_japan': origin_japan,
            'origin_usa': origin_usa
        }

        # 2. Convert to DataFrame and enforce correct feature order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_ORDER]

        # 3. Scale the input data
        scaled_input = scaler.transform(input_df)

        # 4. Make the prediction
        # The random forest model may have been trained as a regressor (as the snippet suggests)
        # or a classifier. We'll use the 'predict' method and convert to binary output.

        raw_prediction = model.predict(scaled_input)[0]

        # We assume the model predicts a score, and we need a threshold (e.g., 0.5)
        # to convert it to the binary 0/1 output you requested.
        # If your model is a classifier, this step might be simpler (just model.predict(X)).
        # We'll round the output for a binary result.
        
        # For a binary classification model (like RandomForestClassifier), use:
        # final_prediction = int(raw_prediction)
        
        # If the model is a regressor (which the snippet suggests):
        final_prediction = 1 if raw_prediction >= 0.5 else 0 


        # 5. Display the result
        st.subheader("Prediction Result")

        if final_prediction == 1:
            st.success("✅ **High Car Efficiency (1)**")
            st.balloons()
            st.markdown("The car is predicted to have **High Fuel Efficiency**.")
        else:
            st.error("❌ **Low Car Efficiency (0)**")
            st.markdown("The car is predicted to have **Low Fuel Efficiency**.")

        # Optional: Display the raw input data
        with st.expander("Show Input Data"):
            st.dataframe(input_df.T, use_container_width=True)
            st.markdown(f"**Raw Model Prediction Score:** `{raw_prediction:.4f}`")

if __name__ == '__main__':
    main()