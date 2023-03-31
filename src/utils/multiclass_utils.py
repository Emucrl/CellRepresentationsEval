from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.utils import classification_results_utils


def multiclass_classification(
    training_set: pd.DataFrame,
    test_set: pd.DataFrame,
    title: str,
    classes: List[str],
    label_col: str = "indication",
    features_col: str = "bag_of_embedding",
) -> Dict:
    """Multiclass logistic regression

    Args:
        training_set (pd.DataFrame): df with features_col and label_col columns for training patients
        test_set (pd.DataFrame): df with features_col and label_col columns for test patients
        title (str): exp name
        classes (List[str]): List of classes to use for the classification
        label_col (str, optional): col name in training/test_set with gt. Defaults to "indication".
        features_col (str, optional): col name in training/test_set with representations to classify. Defaults to "bag_of_embedding".

    Returns:
        Dict: multiclass metrics
    """

    model = LogisticRegression(
        class_weight="balanced",
    )

    model = model.fit(
        np.vstack(training_set[features_col].values),
        np.array([classes.index(value) for value in training_set[label_col].values]),
    )

    probas = model.predict_proba(np.vstack(test_set[features_col].values))

    results = classification_results_utils.get_multiclass_classif_results(
        torch.tensor(probas),
        torch.tensor([classes.index(value) for value in test_set[label_col].values]),
        n_classes=len(np.unique(test_set[label_col].values)),
    )
    results = {key: [value] for key, value in results.items()}
    results["exp"] = [title]

    return results
