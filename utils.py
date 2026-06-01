import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, RocCurveDisplay, ConfusionMatrixDisplay



def split_data(input_features: pd.DataFrame, target_labels: pd.Series, test_size: int | float) -> tuple:
    """
    Split dataset into training and test set

    Args:
        input_features: A matrix of input features
        target_labels: A series of target labels
        test_size: The size of the test/validation set - fraction of the whole dataset if float, 
        number of samples otherwise

    Returns:
        Train and test/validation splits 
    """
    attributes_train, attributes_test, target_train, target_test = train_test_split(
        input_features, 
        target_labels, 
        test_size=test_size, 
        stratify=target_labels,
        shuffle=True,
        random_state=42
    )

    return attributes_train, attributes_test, target_train, target_test


def get_predicted_labels(model: BaseEstimator | Pipeline, input_features: pd.DataFrame) -> np.ndarray:
    """
    Get predicted label probabilities from the model based on input features

    Args:
        model: Fitted scikit-learn estimator or pipeline
        input_features: A matrix of input features
    
    Returns:
        Predicted label probabilities based on the input features
    """
    return model.predict(input_features)


def get_predicted_label_probabilities(model: BaseEstimator | Pipeline, input_features: pd.DataFrame) -> np.ndarray:
    """
    Get predicted labels from the model based on input features

    Args:
        model: Fitted scikit-learn estimator or pipeline
        input_features: A matrix of input features

    Returns:
        Predicted labels based on the input features
    """
    return model.predict_proba(input_features)[:, 1]


def evaluate_model(model: BaseEstimator | Pipeline, input_features: pd.DataFrame, target_labels: pd.Series):
    """
    Evaluate model performance on the training, validation or test set

    Args:
        model: Fitted scikit-learn estimator or pipeline
        input_features: A matrix of input features
        target_labels: A series of target labels
    """
    accuracy = balanced_accuracy_score(target_labels, get_predicted_labels(model, input_features))
    f1score = f1_score(target_labels, get_predicted_labels(model, input_features))
    roc_auc = roc_auc_score(target_labels, get_predicted_label_probabilities(model, input_features))

    print(f"Balanced accuracy score: {accuracy:.4f}")
    print(f"F1 score: {f1_score:.4f}")
    print(f"ROC AUC score: {roc_auc}")


def plot_roc_curve(
        model: BaseEstimator | Pipeline, 
        input_features, 
        target_labels, 
        plot_title: str, 
        plot_chance_level: bool = True,
    ):
    """
    Display ROC curve plot

    Args:
        model: Fitted scikit-learn estimator or pipeline
        input_features: A matrix of input features
        target_labels: A series of target labels
        plot_title: Title of the plot
        plot_chance_level: Whether to plot the chance level. Default True
    """
    RocCurveDisplay.from_estimator(
        estimator=model, 
        X=input_features, 
        y=target_labels, 
        plot_chance_level=plot_chance_level
    )
    plt.title(plot_title)
    plt.show()


def plot_confusion_matrix(
    model: BaseEstimator | Pipeline, 
    input_features: pd.DataFrame, 
    target_labels: pd.Series, 
    plot_title: str, 
):
    """
    Display confusion matrix plot

    Args:
        model: Fitted scikit-learn estimator or pipeline
        input_features: A matrix of input features
        target_labels: A series of target labels
        plot_title: Title of the plot
    """
    ConfusionMatrixDisplay.from_predictions(
        y_true=target_labels,
        y_pred=get_predicted_labels(model, input_features)
    )

    plt.title(plot_title)
    plt.show()