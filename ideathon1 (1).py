import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap
from streamlit_shap import st_shap

# Title
st.title("AI-Human Collaboration for Agriculture")

# File uploaders
st.sidebar.header("Upload Dataset Files")
climate_data_file = st.sidebar.file_uploader("Upload Climate Data (CSV)", type="csv")
province_detail_file = st.sidebar.file_uploader("Upload Province Detail (CSV)", type="csv")
station_detail_file = st.sidebar.file_uploader("Upload Station Detail (CSV)", type="csv")

if climate_data_file and province_detail_file and station_detail_file:
    try:
        # Load datasets
        climate_data = pd.read_csv(climate_data_file)
        province_detail = pd.read_csv(province_detail_file)
        station_detail = pd.read_csv(station_detail_file)

        # Merge datasets
        climate_station_merged = pd.merge(climate_data, station_detail, on='station_id', how='left')
        final_dataset = pd.merge(climate_station_merged, province_detail, on='province_id', how='left')

        # Display merged dataset
        st.subheader("Merged Dataset")
        st.write(final_dataset.head())

        # Model training and evaluation
        st.subheader("AI Model Training")

        features = ['Tn', 'Tx', 'RH_avg', 'RR']  # Features for prediction
        target = 'Tavg'  # Target variable

        if set(features).issubset(final_dataset.columns) and target in final_dataset.columns:
            data = final_dataset.dropna(subset=[target])
            X = data[features]
            y = data[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Predict and calculate MSE
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"AI Model Mean Squared Error: {mse:.2f}")

            # Display predictions
            st.subheader("Predictions vs Actual")
            predictions = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.write(predictions.head())

            # Feedback mechanism
            st.subheader("Provide Feedback on AI Prediction")
            feedback_index = st.number_input("Select Prediction Index (0-4):", min_value=0, max_value=4, step=1)
            feedback = st.radio("Was the prediction correct?", ('Yes', 'No'), index=0)

            if feedback == 'No':
                correct_value = st.number_input("Enter the correct value for the target variable:", min_value=0.0)
                if st.button("Retrain AI Model"):
                    # Add feedback data and retrain model
                    new_data = X_test.iloc[[feedback_index]]
                    X_train = pd.concat([X_train, new_data], ignore_index=True)
                    y_train = pd.concat([y_train, pd.Series(correct_value)], ignore_index=True)
                    model.fit(X_train, y_train)
                    st.write("AI model retrained with the new data!")

            # SHAP Explanation
            st.subheader("SHAP Explanation for AI Prediction")
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            st.write("SHAP Force Plot for Selected Prediction:")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[feedback_index], X_test.iloc[feedback_index]))

        else:
            st.error("The dataset does not contain the required features or target variable.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload all the required CSV files to proceed.")
