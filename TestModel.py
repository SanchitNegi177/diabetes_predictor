#  import libraries
import numpy as np # type: ignore
import pickle
import tkinter as tk
from tkinter import messagebox

# Load the model
with open('Diabetes_model.pkl', 'rb') as f:
    model, scaler, pca = pickle.load(f)

"""# **Predicting for Custom dataset GUI**"""

def submit_data():
    try:
        name = entry_name.get()

        gender = gender_var.get()
        if gender not in [0, 1]:
            raise ValueError("Gender must be selected.")

        age = int(entry_age.get())
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0 and 120.")

        hypertension = hypertension_var.get()
        if hypertension not in [0, 1]:
            raise ValueError("Hypertension must be selected.")

        heart_disease = heart_disease_var.get()
        if heart_disease not in [0, 1]:
            raise ValueError("Heart Disease must be selected.")

        smoking_history = smoking_history_var.get()
        if smoking_history not in [0, 1]:
            raise ValueError("Smoking History must be selected.")

        bmi = float(entry_bmi.get())
        if not (10 <= bmi <= 100):
            raise ValueError("BMI must be between 10 and 100.")

        hba1c = float(entry_hba1c.get())
        if not (3 <= hba1c <= 10):
            raise ValueError("HbA1c level must be between 3 and 10.")

        glucose = int(entry_glucose.get())
        if not (80 <= glucose <= 300):
            raise ValueError("Glucose level must be between 80 and 300.")

        patient_data_dict = {
        "Name": name,
        "Gender": "Male" if gender == 1 else "Female",
        "Age": age,
        "Hypertension": "Yes" if hypertension == 1 else "No",
        "Heart Disease": "Yes" if heart_disease == 1 else "No",
        "Smoking History": "Yes" if smoking_history == 1 else "No",
        "BMI": bmi,
        "HbA1c": hba1c,
        "Glucose": glucose
        }

        patient_data=[gender,age,hypertension,heart_disease,smoking_history,bmi,hba1c,glucose]
        
        data_as_array=np.asarray(patient_data)
        data_as_array=data_as_array.reshape(1,-1)

          # Standardize the custom data
        custom_X = scaler.transform(data_as_array)

          # Apply PCA transformation
        custom_X_pca = pca.transform(custom_X)

          # Make predictions using the trained XGBoost model
        custom_predictions = model.predict(custom_X_pca)
        res=""
        if custom_predictions == 0:
          res=f"{patient_data_dict['Name']} is not predicted to have diabetes."
        else:
          res=f"{patient_data_dict['Name']} is predicted to have diabetes."

        result_label.config(text=res)


        if messagebox.askyesno("Continue", "Do you want to enter another patient's data?"):
            reset_form()
        else:
            root.quit()

    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))

def reset_form():
    entry_name.delete(0, tk.END)
    gender_var.set(-1)
    entry_age.delete(0, tk.END)
    hypertension_var.set(-1)
    heart_disease_var.set(-1)
    smoking_history_var.set(-1)
    entry_bmi.delete(0, tk.END)
    entry_hba1c.delete(0, tk.END)
    entry_glucose.delete(0, tk.END)
    result_label.config(text="")


root = tk.Tk()
root.title("Patient Data Collection")
root.resizable(False, False)

# Create input fields for a single patient's data
tk.Label(root, text="Name:").grid(row=0, column=0, sticky="w")
entry_name = tk.Entry(root)
entry_name.grid(row=0, column=1)

# Gender
tk.Label(root, text="Gender:").grid(row=1, column=0, sticky="w")
gender_var = tk.IntVar(value=-1)
tk.Radiobutton(root, text="Male", variable=gender_var, value=1).grid(row=1, column=1, sticky="w")
tk.Radiobutton(root, text="Female", variable=gender_var, value=0).grid(row=1, column=2, sticky="w")

# Age
tk.Label(root, text="Age (0-120):").grid(row=2, column=0, sticky="w")
entry_age = tk.Entry(root)
entry_age.grid(row=2, column=1)

# Hypertension
tk.Label(root, text="Hypertension:").grid(row=3, column=0, sticky="w")
hypertension_var = tk.IntVar(value=-1)
tk.Radiobutton(root, text="Yes", variable=hypertension_var, value=1).grid(row=3, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=hypertension_var, value=0).grid(row=3, column=2, sticky="w")

# Heart Disease
tk.Label(root, text="Heart Disease:").grid(row=4, column=0, sticky="w")
heart_disease_var = tk.IntVar(value=-1)
tk.Radiobutton(root, text="Yes", variable=heart_disease_var, value=1).grid(row=4, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=heart_disease_var, value=0).grid(row=4, column=2, sticky="w")

# Smoking History
tk.Label(root, text="Smoking History:").grid(row=5, column=0, sticky="w")
smoking_history_var = tk.IntVar(value=-1)
tk.Radiobutton(root, text="Yes", variable=smoking_history_var, value=1).grid(row=5, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=smoking_history_var, value=0).grid(row=5, column=2, sticky="w")

# BMI
tk.Label(root, text="BMI (10-100):").grid(row=6, column=0, sticky="w")
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=6, column=1)

# HbA1c level
tk.Label(root, text="HbA1c level (3-10):").grid(row=7, column=0, sticky="w")
entry_hba1c = tk.Entry(root)
entry_hba1c.grid(row=7, column=1)

# Glucose level
tk.Label(root, text="Glucose level (80-300):").grid(row=8, column=0, sticky="w")
entry_glucose = tk.Entry(root)
entry_glucose.grid(row=8, column=1)

# Submit button
submit_button = tk.Button(root, text="Submit Data", command=submit_data)
submit_button.grid(row=9, column=0, columnspan=3, pady=10)

# Result label
result_label = tk.Label(root, text="", justify="left")
result_label.grid(row=10, column=0, columnspan=3, pady=10)

root.mainloop()