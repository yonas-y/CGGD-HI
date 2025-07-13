from scipy import stats
import tensorflow as tf
import numpy as np
from fast_soft_sort.tf_ops import soft_rank, soft_sort
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def custom_differentiable_spearman_corr_loss(y_pred, y_true):
    """
    Computes the Spearman's rank correlation coefficient using NumPy.

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,) or (batch, N).
        y_pred (np.ndarray): Predicted values, shape (N,) or (batch, N).

    Returns:
        np.ndarray: Spearman's rank correlation coefficient for each batch
    """

    # Reshape to (1, -1) so they become 2D
    y_true = np.reshape(y_true, (1, -1))
    y_pred = np.reshape(y_pred, (1, -1))

    # Compute the ranks of the sorted values.
    y_true_ranks = soft_rank(y_true, regularization=regularization, regularization_strength=regularization_strength)
    y_pred_ranks = soft_rank(y_pred, regularization=regularization, regularization_strength=regularization_strength)

    # Compute squared rank differences
    rank_diffs = y_true_ranks - y_pred_ranks
    squared_rank_diffs = rank_diffs ** 2

    n = y_true.shape[1]
    spearman_corr = 1 - 6 * np.sum(squared_rank_diffs, axis=1) / (n * (n ** 2 - 1))

    spearman_loss = (1 + spearman_corr) / 2.0

    return spearman_loss
