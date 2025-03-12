# import numpy as np 
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow 
import matplotlib.pyplot as plt
import os 
# from mlflow.sklearn
import dagshub
dagshub.init(repo_owner='abubakarsaddique3434', repo_name='mlflow-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/abubakarsaddique3434/mlflow-demo.mlflow")


# Load Iris dataset from seaborn
iris = sns.load_dataset('iris')

# Explore the dataset
print(iris.head())

# Prepare features and labels
X = iris.drop('species', axis=1)  # Features
y = iris['species']               # Target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depth = 5
random_state = 42

os.makedirs("artifacts", exist_ok=True)
# implement mlflow
mlflow.set_experiment('Iris_experiment')
with mlflow.start_run():
    # Initialize Random Forest model
    rf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Evaluate model performance
    accuracy=accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Make a heatmap and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Log model parameters
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    # log model 
    mlflow.sklearn.log_model(rf, "model")
