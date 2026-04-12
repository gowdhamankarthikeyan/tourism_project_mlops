import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide")
st.title('Wellness Tourism Package Prediction (Full Model)')
st.write("This form captures all parameters required for a complete 1-to-1 model inference.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="gowdhamankarthikeyan/wellness-tourism-model", filename="rf_model.joblib")
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.markdown("### Customer Details & Interaction Data")

# Using 3 columns to organize the 18 inputs cleanly
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    age = st.number_input('Age', min_value=18, max_value=100, value=28)
    gender = st.selectbox('Gender', ['Male', 'Female'], index=0)
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'], index=0)
    num_persons = st.number_input('Total Persons Visiting', min_value=1, value=3)
    num_children = st.number_input('Number of Children Visiting', min_value=0, value=0)
    city_tier = st.selectbox('City Tier', [1, 2, 3], index=2) # Default 3

with col2:
    st.markdown("**Professional & Financial**")
    occupation = st.selectbox('Occupation', ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'], index=0)
    designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'], index=1)
    income = st.number_input('Monthly Income', min_value=0.0, value=25000.0)
    passport = st.selectbox('Has Passport?', ['Yes', 'No'], index=0) # Default Yes
    own_car = st.selectbox('Owns Car?', ['Yes', 'No'], index=0) # Default Yes
    num_trips = st.number_input('Average Trips per Year', min_value=1, value=3)

with col3:
    st.markdown("**Sales Interaction**")
    typeof_contact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'], index=0)
    product_pitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'], index=0)
    duration = st.number_input('Duration of Pitch (minutes)', min_value=0.0, value=15.0)
    follow_ups = st.number_input('Number of Follow-ups', min_value=1, value=5)
    pitch_satisfaction = st.slider('Pitch Satisfaction Score', 1, 5, 5)
    preferred_star = st.slider('Preferred Property Star', 3, 5, 4)

st.markdown("---")

if st.button('Predict Likelihood', use_container_width=True):
    try:
        # Get the exact columns the model was trained on
        expected_columns = model.feature_names_in_

        # Create a dataframe with 1 row of all zeros
        df_pred = pd.DataFrame(0, index=[0], columns=expected_columns)

        # 1. Map Direct Numeric Inputs
        numeric_inputs = {
            'Age': age,
            'DurationOfPitch': duration,
            'NumberOfPersonVisiting': num_persons,
            'NumberOfFollowups': follow_ups,
            'PreferredPropertyStar': preferred_star,
            'NumberOfTrips': num_trips,
            'PitchSatisfactionScore': pitch_satisfaction,
            'NumberOfChildrenVisiting': num_children,
            'MonthlyIncome': income,
            'CityTier': city_tier,
            'Passport': 1 if passport == 'Yes' else 0,
            'OwnCar': 1 if own_car == 'Yes' else 0
        }

        # 2. Map Categorical Inputs for One-Hot Encoding
        categorical_inputs = {
            'TypeofContact': typeof_contact,
            'Occupation': occupation,
            'Gender': gender,
            'ProductPitched': product_pitched,
            'MaritalStatus': marital_status,
            'Designation': designation
        }

        # Inject Numeric values
        for key, value in numeric_inputs.items():
            if key in df_pred.columns:
                df_pred.at[0, key] = value

        # Inject Categorical values (Handling the drop_first logic perfectly)
        for key, value in categorical_inputs.items():
            col_name = f"{key}_{value}"
            if col_name in df_pred.columns:
                df_pred.at[0, col_name] = 1

        # Run the prediction
        prediction = model.predict(df_pred)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success("🎉 LIKELY TO PURCHASE: The model predicts this customer WILL buy the Wellness Package.")
        else:
            st.error("❌ UNLIKELY TO PURCHASE: The model predicts this customer will NOT buy the Wellness Package.")

    except Exception as e:
        st.error(f"An error occurred during prediction logic: {e}")
