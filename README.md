# Credit Card Approval 💳
Link to dataset: [Kaggle Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data)

This project implements an ensemble machine learning pipeline to predict credit card approval risks. It leverages **MLflow** to manage the lifecycle of machine learning models, including experiment tracking, model registration, and automated champion/challenger comparison.

## Project Overview
The goal of this project is to build and compare several machine learning models to predict if an applicant is a 'good' or 'risky' customer. The definition of 'good' or 'risky' is not provided, so we need to construct the labels ourselves. Additionally, the project addresses the challenge of imbalanced data, which is a significant issue in this dataset.

The core of this project is an **Ensemble Model** that aggregates predictions from three distinct base models to improve classification performance:
1.  **XGBoost**
2.  **Random Forest**
3.  **Neural Network**

The ensemble supports two voting mechanisms:
*   **Hard Voting:** Majority rule based on the binary predictions of each base model (using their specific optimal thresholds).
*   **Soft Voting:** Averaging the predicted probabilities from all models and applying a global threshold.

## Features

*   **Dynamic Model Loading:** Automatically fetches the current `champion` versions of the base models from the MLflow Model Registry.
*   **Custom MLflow Model:** Defines a `EnsembleModel` class (inheriting from `mlflow.pyfunc.PythonModel`) that standardizes inference across different libraries (Scikit-Learn, Keras, XGBoost).
*   **Experiment Tracking:** Logs parameters, metrics (Accuracy, Recall, Precision, F1), and artifacts to MLflow.
*   **Automated Evaluation:**
    *   Calculates optimal thresholds for soft voting.
    *   Registers new ensemble versions.
    *   Compares the new "challenger" model against the current "champion" on the test set and promotes it if performance improves.

## Prerequisites

*   **Python 3.x**
*   **Libraries:** `pandas`, `numpy`, `xgboost`, `tensorflow` (Keras), `scikit-learn`, `mlflow`, `matplotlib`
*   **MLflow Server:** Must be running and accessible. Configuration is read from `../config.json`.

## Project Structure
This project contains the following files:
- **CreditCardApproval.ipynb**: This notebook provides an overview of the provided datasets and performs initial data preprocessing, including the construction of labels. It also includes exploratory data analysis (EDA) to better understand the dataset. Finally, the dataset is split and prepared for training to make it possible to compare each model against each other by securing that each model is trained on the same data.
  
- **ModelResults.ipynb**: Provides a quick overview of the results from the trained models.
  
- **XGBoost.ipynb**: Notebook for training a XGBClassifier on the dataset.
  
- **RandomForest.ipynb**: Notebook for training a RandomForestClassifier.

- **Neural network.ipynb**: Notebook for training a neural network and explore different ways to handle the class imbalance in the dataset. Various techniques are tested to mitigate the impact of the imbalance.
  
- **Functions.py**: This script contains utility functions used for evaluating the models in the above notebooks.

## Usage

1.  **Configure MLflow:** Ensure `../config.json` exists and points to your MLflow tracking server:
    ```json
    {
      "host": "127.0.0.1",
      "port": "8080"
    }
    ```
2.  **Prepare Base Models:** Ensure the base models (XGBoost, Random Forest, Neural Network) have been trained and registered in MLflow with the alias `champion`.
3.  **Run the Notebook:** Execute `Algorithms/EnsembleModel.ipynb`. It will:
    *   Load the data.
    *   Load the base models.
    *   Run Hard Voting evaluation.
    *   Run Soft Voting evaluation.
    *   Log results to MLflow.
    *   Register the model and attempt to promote it to champion.

## Metrics
The project primarily optimizes for **F1 Score**, while also tracking Accuracy, Recall, and Precision.
