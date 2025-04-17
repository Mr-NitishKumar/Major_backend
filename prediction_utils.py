from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

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



def predict_model(model_type, X, int_X , y=None, params={}):
    """
    Trains the specified model using given data and makes a prediction on the input features.
    
    Args:
        model_type (str): Type of ML model - 'linear', 'random_forest', 'svm', 'kmeans'.
        X (pd.DataFrame): Feature dataset.
        int_X (list): Feature values for prediction.
        y (pd.Series, optional): Target values (if applicable).
        params (dict): Extra hyperparameters.

    Returns:
        np.ndarray: Predicted values (or cluster labels in case of KMeans).
    """

    # Ensure int_X is in 2D array format for prediction
    if isinstance(int_X, list):
        int_X = np.array(int_X).reshape(1, -1)

    # LINEAR REGRESSION
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        return model.predict(int_X)

    # RANDOM FOREST (Classifier or Regressor based on target)
    elif model_type == 'random_forest':
        if y.dtype == 'O' or len(set(y)) < 20:  # Heuristic: classification
            model = RandomForestClassifier(n_estimators=int(params.get('n_estimators', 100)))
        else:
            model = RandomForestRegressor(n_estimators=int(params.get('n_estimators', 100)))
        
        model.fit(X, y)
        return model.predict(int_X)

    # SVM (Classifier or Regressor)
    elif model_type == 'svm':
        kernel = params.get('kernel', 'linear')
        
        if y.dtype == 'O' or len(set(y)) < 20:  # Heuristic: classification
            model = SVC(kernel=kernel, probability=True)
        else:
            model = SVR(kernel=kernel)

        model.fit(X, y)
        return model.predict(int_X)

    # K-MEANS CLUSTERING (Unsupervised)
    elif model_type == 'kmeans':
        n_clusters = int(params.get('n_clusters', 3))
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        return model.predict(X)  # Predict cluster of the new point

    else:
        raise ValueError(f"Invalid model type: {model_type}")
