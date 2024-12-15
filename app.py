import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

def load_data_and_models():
    print("Loading data...")
    data = pd.read_csv('taxi_trip_pricing.csv')
    data.dropna(inplace=True)  # Drop rows with NaN values
    
    X = data.drop(columns='Trip_Price')
    y = data['Trip_Price']

    print("Data loaded and preprocessed.")

    # Correct column names (ensure they match with your actual dataframe)
    numeric_features = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather', 'Passenger_Count']

    # Preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # All numeric features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Models
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    pipelines = {}
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing only
            ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')),  # Discretizer after preprocessing
            ('model', model)
        ])
        try:
            pipeline.fit(X, y)
            print(f"{name} trained successfully.")
        except ValueError as e:
            if "y contains previously unseen labels" in str(e):
                # Re-train the model after re-fitting the entire pipeline with updated target labels
                y = data['Trip_Price']
                pipeline.fit(X, y)
                print(f"{name} retrained due to unseen labels.")
        pipelines[name] = pipeline

    print("All models trained.")
    return pipelines, X, y

if __name__ == "__main__":
    pipelines, X, y = load_data_and_models()
    print("Model loading complete. Pipelines are ready for use.")

