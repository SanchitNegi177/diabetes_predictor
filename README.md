# Diabetes Predictor

A machine‑learning web application to predict the likelihood of diabetes based on user input of health metrics. Built with Flask for the backend and uses classification algorithms to make predictions.

## Table of Contents
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Modeling & Algorithms](#modeling--algorithms)  
- [Data](#data)  
- [Screenshots](#screenshots)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

## Features
- Input form to accept user health metrics such as glucose, BMI, age, etc.  
- Pre‑processing using scaling and encoding of input features.  
- Model choice includes K‑Nearest Neighbors, Random Forest, XGBoost, ensemble voting classifier.  
- Oversampling with SMOTE to handle class imbalance.  
- Visualization support using Matplotlib/Seaborn 
- Web interface built on Flask: render template, accept input, display prediction, optionally download result.  


## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/SanchitNegi177/diabetes_predictor.git
   cd diabetes_predictor
   ```
2. Create and activate a virtual environment (highly recommended):  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure you have the trained model file (`model.pkl`) in the root directory or adjust the path in `app.py`.

## Usage
1. Start the Flask app:  
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://127.0.0.1:5000/`.  
3. Fill in the input form with your health metrics and submit.  
4. View the prediction result on the next page.  
5. (Optional) Download the result or view confusion matrix / classification report (if enabled in UI).

## Modeling & Algorithms
- Data preprocessing: Label Encoding for categorical features (via `LabelEncoder`), scaling via `StandardScaler`.  
- Handling imbalanced classes with `SMOTE` (from `imblearn.over_sampling`).  
- Model algorithms included:  
  - `KNeighborsClassifier`  
  - `RandomForestClassifier`  
  - `XGBClassifier` (from XGBoost)  
  - `VotingClassifier` combining multiple models  
- Evaluation metrics: confusion matrix, accuracy score, classification report (`classification_report`).  
- Visualizations: created with `Matplotlib` and `Seaborn`.

## Data
- Source of data: (provide dataset info here, e.g., “Pima Indians Diabetes Dataset” or other).  
- Features include: Age, BMI, Glucose level, Blood Pressure, Skin Thickness, Insulin, etc.  
- Label: Presence of diabetes (0 = No, 1 = Yes).  
- Pre‑processing steps:  
  - Missing values handling  
  - Encoding categorical variables 
  - Rescaling numerical features  
  - Oversampling minority class (SMOTE)  
- Train/Test split via `train_test_split`.


## Future Improvements
- Add user authentication and a dashboard for storing past predictions.  
- Expand feature set (include family history, lifestyle factors).  
- Deploy the app to a cloud service or containerise with Docker.  
- Improve UI/UX with responsive design and better styling.  
- Add advanced explanation of model predictions (e.g., SHAP values, LIME).  
- Schedule periodic retraining with new data to keep model updated.

## License
This project is **free to use** for learning, research, and personal projects. 
