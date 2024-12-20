from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (make sure this file exists)
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Define the feature names (must match the model's training data)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def index():
    # This route renders the input form for user data
    return render_template('index.html')  # Ensure you have the HTML form named 'index.html'

@app.route('/predict', methods=['POST'])
def predict():
    # Collect the form data submitted by the user
    pregnancies = int(request.form['Pregnancies'])
    glucose = int(request.form['Glucose'])
    blood_pressure = int(request.form['BloodPressure'])
    skin_thickness = int(request.form['SkinThickness'])
    insulin = int(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    diabetes_pedigree = float(request.form['DiabetesPedigreeFunction'])
    age = int(request.form['Age'])

    # Create input data list
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

    # Convert the input data into a DataFrame with the same column names as the training data
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make the prediction using the trained model
    prediction = model.predict(input_df)

    # Determine result and message
    if prediction == 1:
        result = "Diabetic"
        message = "It is highly recommended to consult a doctor for further medical advice."
    else:
        result = "Non-Diabetic"
        message = "You are in the safe zone, but continue to monitor your health."

    # Render the result page with the prediction and message
    return render_template('results.html', result=result, message=message)

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
