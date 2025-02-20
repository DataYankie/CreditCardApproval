# Credit Card Approval 💳
Link to dataset: [Kaggle Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data)

## Project Overview
The goal of this project is to build and compare several machine learning models to predict if an applicant is a 'good' or 'risky' customer. The definition of 'good' or 'risky' is not provided, so we need to construct the labels ourselves. Additionally, the project addresses the challenge of imbalanced data, which is a significant issue in this dataset.

## Project Structure
This project contains the following files:
- **CreditCardApproval.ipynb**: This notebook provides an overview of the provided datasets and performs initial data preprocessing, including the construction of labels. It also includes exploratory data analysis (EDA) to better understand the dataset. Finally, the dataset is split and prepared for training to make it possible to compare each model against each other by securing that each model is trained on the same data.
  
- **ModelResults.ipynb**: Provides a quick overview of the results from the trained models.
  
- **XGBoost.ipynb**: Notebook for training a XGBClassifier on the dataset.
  
- **RandomForest.ipynb**: Notebook for training a RandomForestClassifier.

- **Neural network.ipynb**: Notebook for training a neural network and explore different ways to handle the class imbalance in the dataset. Various techniques are tested to mitigate the impact of the imbalance.
  
- **Functions.py**: This script contains utility functions used for evaluating the models in the above notebooks.

## Usage

To run the notebooks, you will need to have the necessary libraries installed. You can install the required libraries using the following command:

```sh
pip install -r requirements.txt
```
The order to run the notebooks:
1. *CreditCardApproval.ipynb*
2. Order does not matter for these:
    1. *XGBoost.ipynb*
    2. *RandomForest.ipynb*
    3. *Neural network.ipynb*
4. *ModelResults.ipynb*
