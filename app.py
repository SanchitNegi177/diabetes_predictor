from flask import Flask, render_template, request, send_file
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']
        smoking_history = request.form['smoking_history']
        bmi = request.form['bmi']
        hba1c = request.form['hba1c']
        glucose = request.form['glucose']

    # prediction algorithm
       
        with open('Models\Diabetes_model_xgbclassifier.pkl', 'rb') as f: # change it as per the model you want to use
            model, scaler = pickle.load(f)

        submitted_patient_data = [name,gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]
        # data processing
        gender=0 if gender=="Female" else 1
        heart_disease=0 if heart_disease=="No" else 1
        hypertension=0 if hypertension=="No" else 1
        if smoking_history=="No info":
            smoking_history=0
        elif smoking_history=="Current":
            smoking_history=1
        elif smoking_history=="Former":
            smoking_history=1
        elif smoking_history=="Never":
            smoking_history=0
        
        # Prepare data for prediction
        patient_data = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]
        data_as_array = np.asarray(patient_data).reshape(1, -1)
        data_as_array = [[float(item) for item in sublist] for sublist in data_as_array]
        print(data_as_array)

        # Standardize the custom data
        custom_X = scaler.transform(data_as_array)

        # Make predictions 
        custom_predictions = model.predict(custom_X)    

        if custom_predictions == 0:
            prediction = f"{name} does not have diabetes."
        else:
            prediction = f"{name} have diabetes."


        return render_template('result.html', prediction=prediction, result_text=submitted_patient_data )
    
    except Exception as e:
        print("Invalid Input", e)

        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
