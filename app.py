import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_and_train_models():
    # Load and preprocess data
    data = pd.read_csv('taxi_trip_pricing.csv')
    data.dropna(inplace=True)

    X = data.drop(columns='Trip_Price')
    y = data['Trip_Price']  # Continuous variable for regression

    numeric_features = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather', 'Passenger_Count']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models (KNeighbors and DecisionTree)
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor())
    ])
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor())
    ])

    knn_pipeline.fit(X_train, y_train)
    dt_pipeline.fit(X_train, y_train)

    # Predict
    y_pred_knn = knn_pipeline.predict(X_test)
    y_pred_dt = dt_pipeline.predict(X_test)

    # Evaluate
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    r2_knn = r2_score(y_test, y_pred_knn)

    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    r2_dt = r2_score(y_test, y_pred_dt)

    return knn_pipeline, dt_pipeline, rmse_knn, r2_knn, rmse_dt, r2_dt, y_test, y_pred_knn, y_pred_dt

def plot_binned_confusion_matrix(y_test, y_pred_knn, y_pred_dt, bins=5):
    # Create bins for the actual and predicted values
    bin_edges = np.linspace(y_test.min(), y_test.max(), bins + 1)
    y_test_binned = np.digitize(y_test, bin_edges) - 1
    y_pred_knn_binned = np.digitize(y_pred_knn, bin_edges) - 1
    y_pred_dt_binned = np.digitize(y_pred_dt, bin_edges) - 1

    # Confusion matrix for KNN predictions
    cm_knn = confusion_matrix(y_test_binned, y_pred_knn_binned)
    cm_dt = confusion_matrix(y_test_binned, y_pred_dt_binned)

    # Plot confusion matrices for both models
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, bins+1), yticklabels=np.arange(1, bins+1), ax=axes[0])
    axes[0].set_title('KNN Confusion Matrix (Binned)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(1, bins+1), yticklabels=np.arange(1, bins+1), ax=axes[1])
    axes[1].set_title('Decision Tree Confusion Matrix (Binned)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    st.pyplot()

def plot_predictions_vs_actuals(y_test, y_pred_knn, y_pred_dt):
    # Plot actual vs predicted values for both models
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # KNN plot
    axes[0].scatter(y_test, y_pred_knn, alpha=0.6, color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    axes[0].set_title('KNN: Predictions vs Actual')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')

    # Decision Tree plot
    axes[1].scatter(y_test, y_pred_dt, alpha=0.6, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    axes[1].set_title('Decision Tree: Predictions vs Actual')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')

    plt.tight_layout()
    st.pyplot()

def main():
    st.title("Taxi Trip Pricing Prediction")
    st.write("Use the inputs below to predict taxi trip pricing and visualize results.")

    # Load model and preprocessors
    knn_pipeline, dt_pipeline, rmse_knn, r2_knn, rmse_dt, r2_dt, y_test, y_pred_knn, y_pred_dt = load_data_and_train_models()

    # Choose model
    model_choice = st.selectbox("Select Model", ["K-Nearest Neighbors", "Decision Tree"])

    if model_choice == "K-Nearest Neighbors":
        st.write("### K-Nearest Neighbors Model")
        st.write(f"RMSE: {rmse_knn}")
        st.write(f"R² Score: {r2_knn}")
        model = knn_pipeline
        y_pred = y_pred_knn
    else:
        st.write("### Decision Tree Model")
        st.write(f"RMSE: {rmse_dt}")
        st.write(f"R² Score: {r2_dt}")
        model = dt_pipeline
        y_pred = y_pred_dt

    # Visualize Predictions vs Actuals for the chosen model
    plot_predictions_vs_actuals(y_test, y_pred_knn, y_pred_dt)

    # Plot binned confusion matrix
    plot_binned_confusion_matrix(y_test, y_pred_knn, y_pred_dt)

    st.header("Input Features")
    numeric_features = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather', 'Passenger_Count']
    inputs = {}

    for feature in numeric_features:
        inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)
    for feature in categorical_features:
        inputs[feature] = st.text_input(f"Enter {feature}", value="")

    # Create input DataFrame
    input_df = pd.DataFrame([inputs])
    st.write("Your Input:")
    st.write(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_df)
        st.subheader("Prediction Results")
        st.write(f"### Predicted Trip Price: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
