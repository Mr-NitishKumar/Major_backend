from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import numpy as np

def evaluate_classification(y_true, y_pred):
    """Calculate classification metrics."""
    metrics = {
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Generate ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_filename = "static/roc_curve.png"
    plt.savefig(roc_filename)
    plt.close()
    metrics["roc_curve"] = roc_filename

    return metrics

def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics."""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r_squared": r2_score(y_true, y_pred),
    }
    return metrics


def predict_model(model_type, X, y=None, params={}):
    """Train and predict using the specified model."""
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100))
    elif model_type == 'svm':
        model = SVC(kernel=params.get('kernel', 'linear'))
    elif model_type == 'kmeans':
        model = KMeans(n_clusters=params.get('n_clusters', 3))
        return model.fit_predict(X)  # KMeans doesn't require `y`
    else:
        raise ValueError("Invalid model type")

    model.fit(X, y)
    return model.predict(X)
