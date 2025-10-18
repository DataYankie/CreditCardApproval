from sklearn.metrics import confusion_matrix, make_scorer, recall_score, precision_score, f1_score, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_curve
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tempfile

# Function for plotting a confusion matrix and print the recall, precision and f1 score 
def get_result(true_label: np.ndarray, predictions: np.ndarray, threshold: float=0.5, visualize: bool = False) -> dict[str, float]:
    """
    Calculate performance metrics and optionally display a confusion matrix.

    Args:
        true_label (np.ndarray): Array of true labels.
        predictions (np.ndarray): Array of predicted probabilities or labels.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
        visualize (bool, optional): Whether to display a confusion matrix heatmap. Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing recall, precision, and F1 score.
    """
    y_pred = (predictions > threshold).astype(int)

    accuracy = make_scorer(accuracy_score(true_label, y_pred), zero_division=0)
    precision = make_scorer(precision_score(true_label, y_pred), zero_division=0)
    recall = make_scorer(recall_score(true_label, y_pred), zero_division=0)
    f1 = make_scorer(f1_score(true_label, y_pred), zero_division=0)

    if visualize:
        cm = confusion_matrix(true_label, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
        print('Accuracy     =', round(accuracy, 2))
        print('Recall       =', round(recall, 2))
        print('Precision    =', round(precision, 2))
        print('F1           =', round(f1, 2))

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision, 
        'f1': f1
    }

# Function for getting the threshold resulting in the highest f1 score 
def get_best_threshold(name: str, true_labels: np.ndarray, predictions: np.ndarray, show_plot: bool = True) -> float:
    """
    Identifies the threshold that results in the best F1 score and optionally plots the precision, recall, and F1 score.

    Args:
        name (str): The name of the model or experiment.
        true_labels (np.ndarray): Array of true binary labels.
        predictions (np.ndarray): Array of predicted probabilities.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        float: A float containing the best threshold.
    """
    # Calculate the precision and recall
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)

    # Calculate the f1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)  # Replace NaN with 0

    # Find the index of the maximum F1 score
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision[:-1], label=name + ' Precision', marker='.', color='Red')
        plt.plot(thresholds, recall[:-1], label=name + ' Recall', marker='.', color='Blue')
        plt.plot(thresholds, f1_scores[:-1], label=name + ' F1 Score', marker='.', color='Green')

        # Add a vertical line at the best threshold
        plt.axvline(x=best_threshold, color='grey', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')

        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Overview')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
    
    return best_threshold

# Function for plotting the ROC curve
def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Plots the Receiver Operating Characteristic (ROC) curve and computes the Area Under the Curve (AUC).

    Args:
        y_true (np.ndarray): Array of true binary labels.
        y_probs (np.ndarray): Array of predicted probabilities.

    Returns:
        float: The AUC value.
    """
    # Calculate the False Positive Rate and True Positive Rate
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    # Calculate the Area Under the ROC Curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Plotting the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC Curve')
    plt.legend(loc="center right")
    plt.grid(True)
    plt.show()

    # Return the AUC value
    return roc_auc

def save_metrics_to_json(model_name: str, recall: float, precision: float, f1_score: float, threshold: float = 0.5, file_path: str ='Model_metrics.json'):
    """
    Saves or updates model evaluation metrics in a JSON file.

    Parameters:
        model_name (str): Name of the model.
        recall (float): Recall value.
        precision (float): Precision value.
        f1_score (float): F1-score value.
        threshold (float): Threshold value.
        file_path (str): Path to the JSON file.
    """
    # Initialize the data dictionary
    data = {}

    # Check if the file exists
    if os.path.exists(file_path):
        # Load existing data
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                pass  # Handle empty or corrupted JSON file

    # Update or add the model's metrics
    data[model_name] = {
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'threshold': threshold
    }

     # Write data to a temporary file first
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
            json.dump(data, temp_file, indent=4)
            temp_file_path = temp_file.name
        
        # Replace original file with temporary file
        os.replace(temp_file_path, file_path)
    except Exception as e:
        print(f"Error: Could not save data to JSON. {e}")

def get_scores(y, preds):
    accuracy = make_scorer(accuracy_score(y, preds), zero_division=0)
    precision = make_scorer(precision_score(y, preds), zero_division=0)
    recall = make_scorer(recall_score(y, preds), zero_division=0)
    f1 = make_scorer(f1_score(y, preds), zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }   