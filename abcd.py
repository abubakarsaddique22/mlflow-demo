import numpy as np 
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow 
# from mlflow.sklearn

# Load Iris dataset from seaborn
iris = sns.load_dataset('iris')

# Explore the dataset
print(iris.head())

# Prepare features and labels
X = iris.drop('species', axis=1)  # Features
y = iris['species']               # Target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depth =4
random_state = 42

# implement mlflow
# mlflow.set_experiment('Iris')
with mlflow.start_run():
    # Initialize Random Forest model
    rf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Evaluate model performance
    accuracy=accuracy_score(y_test, y_pred)
    # reports=classification_report(y_test, y_pred)

    # Log model parameters
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)
    mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_metric('classification_report', reports)