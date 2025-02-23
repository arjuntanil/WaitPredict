from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pickle
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# -------------------------------
# Load SLR Model (Traffic Wait Time prediction using vehicle count)
# -------------------------------
with open("model/traffic_model.pkl", "rb") as file:
    slr_model = pickle.load(file)

# -------------------------------
# Load MLR Model and Transformer (Detailed prediction based on multiple features)
# -------------------------------
with open("model/mlr_model.pkl", "rb") as file:
    mlr_model = pickle.load(file)
with open("model/mlr_transformer.pkl", "rb") as file:
    mlr_transformer = pickle.load(file)

# -------------------------------
# Homepage: Choose Prediction Type
# -------------------------------
@app.route('/')
def home():
    return render_template('home.html')

# -------------------------------
# SLR Routes (Simple Prediction)
# -------------------------------
@app.route('/attendance')
def attendance():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        vehicle_count = int(request.form['vehicle_count'])
        prediction = slr_model.predict(np.array([[vehicle_count]]))[0]
        return render_template('index.html', prediction_text=f"Predicted Wait Time: {prediction:.2f} seconds")
    except Exception as e:
        return str(e)

# -------------------------------
# MLR Routes (Detailed Prediction)
# -------------------------------
@app.route('/mlr')
def mlr_page():
    return render_template('mlr.html')

@app.route('/predict_skip', methods=['POST'])
def predict_skip():
    # Retrieve form data
    distance = float(request.form['distance'])
    time_of_day = request.form['time_of_day']
    weather = request.form['weather']
    road_condition = request.form['road_condition']
    
    # Create DataFrame with column names exactly as used during training
    input_df = pd.DataFrame([[distance, time_of_day, weather, road_condition]],
                            columns=["Vehicle Count (X1)", "Time of Day (X2)", "Weather Condition (X3)", "Road Condition (X4)"])
    
    # Transform the data and predict wait time
    transformed_input = mlr_transformer.transform(input_df)
    predicted_wait = mlr_model.predict(transformed_input)[0]
    
    # Return the prediction as JSON for AJAX
    return jsonify({'prediction_text': f"{predicted_wait:.2f} seconds"})

if __name__ == '__main__':
    app.run(debug=True)
