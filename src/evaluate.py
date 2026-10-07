
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1
    }


def get_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def predict_mlp(model, dataloader, device):
    """
    Generate predictions from the MLP model.
    """

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            batch_predictions = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

            predictions.extend(batch_predictions)
            true_labels.extend(
                y_batch.numpy()
            )

    return (
        np.array(true_labels),
        np.array(predictions)
    )
