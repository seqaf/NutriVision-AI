import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv('Food_Nutrition.csv')

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Drop rows with missing values in relevant columns
df.dropna(subset=['name', 'calories', 'proteins', 'fat', 'carbohydrate'], inplace=True)

# Convert food names to lowercase for consistency
df['name'] = df['name'].str.lower()

# Define features (X) and targets (y)
X = df['name']
y_calories = df['calories']
y_proteins = df['proteins']
y_fat = df['fat']
y_carbohydrate = df['carbohydrate']

# Initialize TF-IDF Vectorizer
# We will fit this on the entire 'name' column to ensure consistency
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # Limit features to avoid sparsity issues
X_vectorized = tfidf_vectorizer.fit_transform(X)

# Save the TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

# List of target variables and their corresponding models
targets = {
    'calories': y_calories,
    'proteins': y_proteins,
    'fat': y_fat,
    'carbohydrate': y_carbohydrate
}

models = {}
metrics = {}

# Train and save a Linear Regression model for each target
for target_name, y_target in targets.items():
    print(f"\nTraining model for: {target_name}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_target, test_size=0.2, random_state=42)

    # Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Ensure non-negative predictions for physical quantities
    y_pred_clipped = np.maximum(0, y_pred)
    mse_clipped = mean_squared_error(y_test, y_pred_clipped)
    r2_clipped = r2_score(y_test, y_pred_clipped)


    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  R-squared (R2): {r2:.2f}")
    print(f"  Mean Squared Error (MSE) (Clipped): {mse_clipped:.2f}")
    print(f"  R-squared (R2) (Clipped): {r2_clipped:.2f}")


    # Save the trained model
    model_filename = f'models/linear_regression_{target_name}.pkl'
    joblib.dump(model, model_filename)
    print(f"  Model saved as: {model_filename}")

    models[target_name] = model
    metrics[target_name] = {'mse': mse, 'r2': r2}

print("\nAll models trained and saved successfully!")
