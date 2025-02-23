import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Ensure the 'model' directory exists.
os.makedirs("model", exist_ok=True)

# Load the dataset
dataset = pd.read_csv('TRAFFIC_SLR.csv')

# Extract features and target variable
X = dataset.iloc[:, 0:1].values  # Vehicle Count as a 2D array
y = dataset.iloc[:, 1].values    # Traffic Light Wait Time

# Split the dataset into training and testing sets (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Optional: Print model performance
print("Model R^2 Score on entire dataset:", regressor.score(X, y))

# Plot the training results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Vehicle Count vs Traffic Light Wait Time (Training)")
plt.xlabel("Vehicle Count")
plt.ylabel("Wait Time (seconds)")
plt.show()

# Save the trained model into the 'model' folder
with open("model/traffic_model.pkl", "wb") as file:
    pickle.dump(regressor, file)
