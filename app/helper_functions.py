import numpy as np
from app.metrics import (custom_differentiable_spearman_corr_loss,
                         calculate_monotonicity1,
                         calculate_monotonicity2,
                         calculate_robustness,
                         calculate_trendability1,
                         calculate_trendability2)


def metric_satisfaction_ratio(predictions, tau: float = 0.025):
    """
    Compute multiple health indicator (HI) quality metrics from predicted values.

    This function smooths the predictions using LOWESS with the specified tau,
    then calculates and returns several metrics characterizing the HI curve:
    - Spearman's rank monotonicity
    - Two variants of monotonicity
    - Robustness
    - Two variants of trendability

    Args:
        predictions: Array-like, shape (n_samples,). Predicted HI values.
        tau (float, optional): Smoothing parameter for LOWESS. Default is 0.025.

    Returns:
        Tuple of floats:
            - Spearman's monotonicity correlation
            - Monotonicity metric 1
            - Monotonicity metric 2
            - Robustness
            - Trendability metric 1
            - Trendability metric 2
    """

    predictions = predictions.reshape(-1)
    time = np.linspace(0, len(predictions)*10, len(predictions))
    smoothed_pred = Metr.lowess(time, predictions, tau=tau)

    spear_mono_correlation = custom_spearmans_rank_correlation(
        np.reshape(smoothed_pred, [-1, 1]), np.reshape(time, [-1, 1]))
    monotonicity1 = calculate_monotonicity1(predictions)
    monotonicity2 = calculate_monotonicity2(predictions)
    robustness = calculate_robustness(predictions, smoothed_pred)
    trendability1 = calculate_trendability1(predictions, time)
    trendability2 = calculate_trendability2(predictions, time)

    return (spear_mono_correlation, monotonicity1, monotonicity2,
            robustness, trendability1, trendability2)