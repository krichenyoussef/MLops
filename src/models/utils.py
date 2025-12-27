import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_stratified_folds(y, n_splits=5, seed=42):
    """
    Create stratified folds for binary classification.

    Parameters
    ----------
    y : array-like
        Target labels (binary).
    n_splits : int
        Number of folds.
    seed : int
        Random seed.

    Returns
    -------
    List of (train_idx, val_idx)
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(skf.split(np.zeros(len(y)), y))
