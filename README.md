# Credit Card Approval
Link to dataset: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data

The goal of this project is to build a few machine learning models (for comparison) to predict if an applicant is a 'good' or 'bad' client, were the definition of 'good' or 'bad' is not given. Thus, we need to find a way to construct the labels. Lastly, imbalance data problem is a big problem in this project, which also needs to be handle.

This project contains the following files:
- *CreditCardApproval.ipynb*, which takes a look at the provided datasets as well as preprocessing of the datasets, where the constructing of the labels also happens. In addition to that, an exploratory data analysis is constructed to provide a better understanding of the dataset. Lastly, the dataset is spilted and made ready to be used for training.
  
- *ModelResults.ipynb*, provides a quick overview of the results from the trained models.
  
- *XGBoost.ipynb*, where we train our XGBClassifier 
- *RandomForest.ipynb*, where we train our RandomForestClassifier
- *Neural network.ipynb*, where we train our Neural network. Here we also try different ways to handle the imblanced data.
  
- *Functions.py*, provides some functions which are used when evaluating each of the above models.

