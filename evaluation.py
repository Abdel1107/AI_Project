from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


# Evaluate a model's prediction using the above metrics
def evaluate_model(y_true, y_pred):
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # Store metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall
    }

    return metrics


# Print the evaluation metrics
def print_evaluation_metrics(metrics, model_name="Model"):
    print(f"\nEvaluation Metrics for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
