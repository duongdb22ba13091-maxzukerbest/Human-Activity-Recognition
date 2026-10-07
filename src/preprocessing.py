
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_scaler():
    """
    Create a StandardScaler for feature standardization.
    """
    return StandardScaler()


def create_label_encoder():
    """
    Create a LabelEncoder for activity labels.
    """
    return LabelEncoder()


def scale_data(X_train, X_validation, X_test):
    """
    Fit the scaler only on training data and transform
    validation and test data.

    This prevents data leakage.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler
    )
