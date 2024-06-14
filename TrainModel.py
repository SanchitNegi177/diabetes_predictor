#  import libraries
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

def main():
        
    """# **Reading data**"""

    df = pd.read_csv('diabetes_prediction_dataset.csv')
    df.head(10)

    """# **Descriptive statistics**"""

    df.describe()

    """# **Basic Info**"""

    df.info()

    """# **Encoding data**"""

    # incode the data
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])
    df.head()

    """# **Standardizing features**"""

    # Selecting features and target variable
    features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X = df[features]
    Y = df['diabetes']

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print(X_pca)

    """# **Building Model**"""

    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.25, random_state=1)

    # Initializing and training the XGBoost model
    model = XGBClassifier(random_state=1)
    model.fit(X_train, Y_train)

    # Making predictions
    Y_pred = model.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy of model: {accuracy:.4f}')
    print(f'Classification Report of model:\n{classification_report(Y_test, Y_pred)}')

    """# **Confusion matrix**"""

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    """# **Save Model**"""

    import pickle

    # Save the model
    with open('Diabetes_model.pkl', 'wb') as f:
        pickle.dump((model, scaler, pca), f)


if __name__ == "__main__":
    main()
