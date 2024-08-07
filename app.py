from flask import Flask, request, jsonify
import numpy as np
import joblib

model = joblib.load('heart_disease_model.pkl')
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bmi = request.form.get('bmi', type=float)
        smoke = request.form.get('smoking', type=int)
        alcoholic = request.form.get('alcoholic', type=int)
        stroke = request.form.get('stroke', type=int)
        physical_health = request.form.get('health_physical', type=float)
        mental_health = request.form.get('health_mental', type=float)
        difficulty_walking = request.form.get('walking', type=int)
        sex = request.form.get('sex', type=int)
        age = request.form.get('age', type=int)
        race = request.form.get('race', type=int)
        diabetic = request.form.get('diabetic', type=int)
        physical_act = request.form.get('physical_act', type=int)
        general_health = request.form.get('health', type=float)
        sleep_time = request.form.get('sleep', type=float)
        asthma = request.form.get('asthma', type=int)
        kidney_dis = request.form.get('kidney', type=int)
        skin_cancer = request.form.get('skin', type=int)

        # Print received inputs for debugging
        print(f"Received inputs - bmi: {bmi}, smoke: {smoke}, alcoholic: {alcoholic}, stroke: {stroke}, physical_health: {physical_health}, mental_health: {mental_health}, difficulty_walking: {difficulty_walking}, sex: {sex}, age: {age}, race: {race}, diabetic: {diabetic}, physical_act: {physical_act}, general_health: {general_health}, sleep_time: {sleep_time}, asthma: {asthma}, kidney_dis: {kidney_dis}, skin_cancer: {skin_cancer}")

        # Ensure all inputs are converted to float
        inputs = [bmi, smoke, alcoholic, stroke, physical_health, mental_health, difficulty_walking,
                  sex, age, race, diabetic, physical_act, general_health, sleep_time, asthma,
                  kidney_dis, skin_cancer]

        # Field names for missing fields
        field_names = ["bmi", "smoking", "alcoholic", "stroke", "health_physical", "health_mental", "walking",
                       "sex", "age", "race", "diabetic", "physical_act", "health", "sleep", "asthma",
                       "kidney", "skin"]

        # Check for any missing values
        missing_fields = [field_names[index] for index, value in enumerate(inputs) if value is None]
        if missing_fields:
            return jsonify({'error': f'Missing input fields: {", ".join(missing_fields)}'}), 400

        input_query = np.array([inputs], dtype=float)

        if np.isnan(input_query).any():
            input_query = np.nan_to_num(input_query)

        res = model.predict(input_query)[0]

        return jsonify({'result': str(res)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
