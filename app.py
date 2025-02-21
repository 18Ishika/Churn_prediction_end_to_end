import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
@st.cache_data
def load_model():
    try:
        model = joblib.load("model.pkl")
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess Data: Scaling & Encoding
def preprocess_data(df):
    required_columns = [
        "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingMovies", "Contract",
        "PaperlessBilling", "MonthlyCharges", "TotalCharges"
    ]

    # Keep only necessary columns
    df = df.loc[:, required_columns]

    # Convert TotalCharges to numeric (handling empty spaces)
    df.loc[:, "TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Label Encoding for categorical features
    categorical_features = [
        "Partner", "Dependents", "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingMovies", "Contract", "PaperlessBilling"
    ]
    
    for col in categorical_features:
        df.loc[:, col] = df[col].astype("category").cat.codes

    # Scale tenure, MonthlyCharges, and TotalCharges
    scaler = StandardScaler()
    df.loc[:, ["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(df[["tenure", "MonthlyCharges", "TotalCharges"]])

    return df

# Function to make predictions
def predict_churn(df):
    model = load_model()
    if model is not None:
        predictions = model.predict(df)
        return predictions
    else:
        return None

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Upload a CSV file with customer data, and the model will predict churn.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.write(df.head())

        # Preprocess data
        with st.spinner("Preprocessing data..."):
            processed_df = preprocess_data(df)
            st.success("Data preprocessing completed!")

        # Predict churn
        with st.spinner("Making predictions..."):
            predictions = predict_churn(processed_df)
            if predictions is not None:
                # Add predictions to dataframe
                df["Churn Prediction"] = predictions

                # Display predictions
                st.subheader("Predictions")
                st.write(df.head())

                # Download predictions as CSV
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode("utf-8")

                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

                st.success("Prediction completed! Download the results using the button above.")
            else:
                st.error("Prediction failed due to model loading error.")

    except KeyError as e:
        st.error(f"Missing column in uploaded data: {e}. Please check your CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")