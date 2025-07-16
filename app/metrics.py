# from scipy import stats
# import tensorflow as tf
import numpy as np
from fast_soft_sort.tf_ops import soft_rank, soft_sort
from scipy.stats import spearmanr
# from sklearn.metrics import mutual_info_score
# from sklearn.preprocessing import KBinsDiscretizer


def custom_differentiable_spearman_corr_loss(y_pred,
                                             y_true,
                                             regularization="l2",
                                             regularization_strength=1.0):
    """
    Computes the Spearman's rank correlation coefficient using NumPy.

    Args:
        y_true (np.ndarray): Ground truth values, shape (N,) or (batch, N).
        y_pred (np.ndarray): Predicted values, shape (N,) or (batch, N).
        regularization:
        regularization_strength:

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

def calculate_monotonicity1(HI):
    """
    Calculate the percentage of monotonousness of health indicator scores using
    ==== The way calculated   =======

    Parameters:
        HI (array): Array of health indicator scores.

    Returns:
        monotonicity_percentage (float): Percentage of monotonousness.
    """
    # Calculate the first derivative
    first_der = np.diff(HI)
    monotonicity_percentage = (np.sum(first_der <= 0) - np.sum(first_der > 0)) / (len(HI) - 1)
    return monotonicity_percentage

def calculate_monotonicity2(HI):
    """
    Calculate the percentage of monotonousness of health indicator scores.
    ==== The way calculated   =======

    Parameters:
        HI (array): Array of health indicator scores.

    Returns:
        monotonicity_percentage (float): Percentage of monotonousness.
    """
    # Calculate the first derivative
    first_der = np.diff(HI)
    second_der = np.diff(first_der)
    mono_positive = (np.sum(first_der > 0) / (len(HI) - 1)) + (np.sum(second_der > 0) / (len(HI) - 2))
    mono_negative = (np.sum(first_der < 0) / (len(HI) - 1)) + (np.sum(second_der < 0) / (len(HI) - 2))
    monotonicity_percentage = mono_negative / (mono_positive + mono_negative)
    return monotonicity_percentage
