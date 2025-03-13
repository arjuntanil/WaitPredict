import pandas as pd
import numpy as np
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Ensure the 'model' directory exists.
os.makedirs("model", exist_ok=True)

# 1. Load the MLR dataset
dataset = pd.read_csv("TRAFFIC_MLR.csv")

# 2. Separate   features (X) and target (y)
#    Features: Vehicle Count, Time of Day, Weather Condition, Road Condition
#    Target: Traffic Light Wait Time (seconds)
X = dataset.iloc[:, :-1]  # All rows, all columns except the last
y = dataset.iloc[:, -1]   # All rows, last column

# 3. Encode the categorical features (columns 1, 2, and 3: Time of Day, Weather Condition, Road Condition)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False), [1, 2, 3])],
    remainder='passthrough'
)
X_encoded = ct.fit_transform(X)

# 4. Split dataset into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# 5. Train the MLR model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# 6. Save the trained model and the transformer
with open("model/mlr_model.pkl", "wb") as model_file:
    pickle.dump(regressor, model_file)

with open("model/mlr_transformer.pkl", "wb") as transformer_file:
    pickle.dump(ct, transformer_file)

# 7. Test the model on the test set and print performance
y_pred = regressor.predict(x_test)
print("Sample Predictions:", y_pred[:5])
print("Actual Values:", y_test[:5].values)

# Optional: Evaluate model performance
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R-squared: {r2:.2f}")
