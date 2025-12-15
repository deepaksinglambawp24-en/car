import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

# --- PLACEHOLDER FOR THE ACTUAL MODEL ---
# Since you only provided the scaler, we define a dummy class
# to make the app executable. You MUST replace 'model.pkl' with your
# actual trained binary classification model.
class BinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    A placeholder classifier to allow the Streamlit app to run.
    Replace this with your actual loaded model (e.g., LogisticRegression).
    """
    def __init__(self):
        # A dummy predict function that always returns 1
        # until the real model is loaded.
        pass

    def predict(self, X):
        # Placeholder logic: e.g., predict '1' (High Efficiency) if
        # the first scaled feature is positive, otherwise '0'.
        # Replace this with: return self.model.predict(X)
        return np.array([1]) # Replace with your model's prediction

# --- MODEL AND SCALER LOADING ---
# Use st.cache_resource to load the files only once when the app starts
@st.cache_resource
def load_artifacts():
    try:
        # Load the Standard Scaler (already provided by the user)
        scaler = joblib.load('standard_scaler.pkl')
    except FileNotFoundError:
        st.error("Error: 'standard_scaler.pkl' not found. Make sure it is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None, None

    try:
        # Load the trained ML model
        # NOTE: You MUST replace this with your actual classification model file.
        # This will be a BinaryClassifier (e.g., LogisticRegression) that predicts 0 or 1.
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        # If the actual model is not found, use the placeholder.
        model = BinaryClassifier()
        st.warning("Warning: 'model.pkl' not found. Using a dummy placeholder model. The results will be inaccurate.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    return scaler, model

# Load the components
scaler, model = load_artifacts()

# Feature names inferred from the standard_scaler.pkl file:
FEATURE_ORDER = [
    'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
    'model_year', 'origin_japan', 'origin_usa'
]

# --- STREAMLIT APP LAYOUT ---
def main():
    st.title("Car Efficiency Prediction App")
    st.markdown("Predict whether a car has **High Efficiency (1)** or **Low Efficiency (0)**.")

    if scaler is None or model is None:
        return

    # Sidebar for user inputs
    st.sidebar.header("Car Specifications Input")

    # 1. Continuous Features (Using common car parameters)
    cylinders = st.sidebar.slider("Cylinders", 3, 8, 4)
    displacement = st.sidebar.number_input("Displacement (cu in)", min_value=50.0, max_value=450.0, value=150.0)
    horsepower = st.sidebar.number_input("Horsepower", min_value=40.0, max_value=300.0, value=90.0)
    weight = st.sidebar.number_input("Weight (lbs)", min_value=1500.0, max_value=5500.0, value=2500.0)
    acceleration = st.sidebar.number_input("Acceleration (sec)", min_value=8.0, max_value=25.0, value=15.0)
    model_year = st.sidebar.slider("Model Year", 70, 82, 75)

    # 2. Categorical Feature (Origin)
    origin_map = {
        'USA': (0, 1),      # origin_japan=0, origin_usa=1
        'Japan': (1, 0),    # origin_japan=1, origin_usa=0
        'Europe': (0, 0)    # origin_japan=0, origin_usa=0 (baseline)
    }
    origin_selection = st.sidebar.selectbox("Car Origin", options=list(origin_map.keys()), index=0)
    origin_japan, origin_usa = origin_map[origin_selection]

    # --- PREDICTION LOGIC ---
    if st.button("Predict Efficiency"):

        # 1. Create a dictionary of all input features
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

        # 2. Convert to a Pandas DataFrame in the correct order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_ORDER]

        # 3. Scale the input data using the loaded StandardScaler
        # The scaler is trained on the data, so we use .transform()
        scaled_input = scaler.transform(input_df)

        # 4. Make the prediction
        # The model's predict method should return a binary value (0 or 1)
        prediction = model.predict(scaled_input)[0]

        # 5. Display the result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("**High Car Efficiency (1)**")
            st.balloons()
            st.markdown("Based on the input specifications, the car is predicted to have **High Fuel Efficiency**.")
        else:
            st.error("**Low Car Efficiency (0)**")
            st.markdown("Based on the input specifications, the car is predicted to have **Low Fuel Efficiency**.")

        # Optional: Display the raw input data
        st.subheader("Raw Input Data")
        st.dataframe(input_df)

if __name__ == '__main__':
    main()