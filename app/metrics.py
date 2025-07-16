import numpy as np
from fast_soft_sort.tf_ops import soft_rank


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

def calculate_robustness(HI, Smoothed_HI):
    """
    Calculate the percentage of robustness of health indicator scores.

    Parameters:
        HI (array): Array of health indicator scores.
        Smoothed_HI (array): Array of Smoothed health indicator scores.

    Returns:
        robustness_percentage (float): Percentage of robustness.
    """
    return np.sum(np.exp(-(np.abs((HI - Smoothed_HI) / HI)))) / len(HI)

def calculate_trendability1(HI, time):
    """
    Calculate the percentage of trendability of health indicator scores.

    Parameters:
        HI (array): Array of health indicator scores.
        time (array): Array of time.
    Returns:
        trendability_percentage (float): Percentage of trendability.
    """
    num = len(HI) * np.sum(HI * time) - np.sum(HI) * np.sum(time)
    deno = np.sqrt((len(HI) * np.sum(np.square(HI)) - np.square(np.sum(HI))) * (
                len(time) * np.sum(np.square(time)) - np.square(np.sum(time))))

    return num / deno


def calculate_trendability2(HI, time):
    """
    Calculate the percentage of trendability of health indicator scores.

    Parameters:
        HI (array): Array of health indicator scores.
        time (array): Array of time.
    Returns:
        trendability_percentage (float): Percentage of trendability.
    """
    # Find the indices that would sort the array
    sorted_indices_HI = np.argsort(HI)
    sorted_indices_time = np.argsort(time)

    # Initialize an array to store the ranks
    ranks_HI = np.empty_like(sorted_indices_HI)
    ranks_time = np.empty_like(sorted_indices_time)

    # Assign ranks to the elements based on their indices
    ranks_HI[sorted_indices_HI] = np.arange(len(HI))
    ranks_time[sorted_indices_time] = np.arange(len(time))

    # Manually scale array between 0 and 1 using NumPy
    min_val_HI, min_val_time = np.min(ranks_HI), np.min(ranks_time)
    max_val_HI, max_val_time = np.max(ranks_HI), np.max(ranks_time)
    ranks_HI_scaled = (ranks_HI - min_val_HI) / (max_val_HI - min_val_HI)
    ranks_time_scaled = (ranks_time - min_val_time) / (max_val_time - min_val_time)

    num = len(ranks_HI_scaled) * np.sum(ranks_HI_scaled * ranks_time_scaled) - np.sum(ranks_HI_scaled) * np.sum(ranks_time_scaled)
    deno = np.sqrt((len(ranks_HI_scaled) * np.sum(np.square(ranks_HI_scaled)) - np.square(np.sum(ranks_HI_scaled))) *
                   (len(ranks_time_scaled) * np.sum(np.square(ranks_time_scaled)) - np.square(np.sum(ranks_time_scaled))))

    return num / deno

def calculate_consistency_metric(hi_runs):
    """
    Calculate the consistency metric for HI predictions from multiple runs with dynamic bin selection.

    hi_runs: 2D numpy array of shape (time_steps, num_runs).

    Returns:
        consistency_metric: Average pairwise symmetrical uncertainty.
    """

    def calculate_entropy(data):
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def calculate_mutual_information(X, Y):
        """Calculate mutual information between two discrete datasets."""
        joint_entropy = calculate_entropy(np.c_[X, Y])
        MI = calculate_entropy(X) + calculate_entropy(Y) - joint_entropy
        return max(MI, 0)  # Clamp to 0 to avoid negative MI due to numerical issues

    def symmetrical_uncertainty(X, Y):
        H_X = calculate_entropy(X)
        H_Y = calculate_entropy(Y)
        MI_XY = calculate_mutual_information(X, Y)
        return 2 * MI_XY / (H_X + H_Y) if H_X + H_Y > 0 else 0

    def calculate_bins(data_length):
        """Determine the number of bins dynamically based on data length."""
        if data_length < 1000:
            return max(2, int(np.log2(data_length) + 1))  # Sturges' formula
        return max(10, int(np.sqrt(data_length)))  # Square root choice

    def discretize_data(data, n_bins):
        """Discretize continuous data into bins using equal-width binning."""
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
        return est.fit_transform(data)

    num_runs = hi_runs.shape[1]
    su_values = []

    n_bins = calculate_bins(hi_runs.shape[0])
    # Discretize each run's data
    hi_discrete = discretize_data(hi_runs.T, n_bins=n_bins).T

    # Calculate pairwise SU
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            su = symmetrical_uncertainty(hi_discrete[:, i], hi_discrete[:, j])
            su_values.append(su)

    # # Compute average SU as consistency metric
    consistency_metric_mean = np.mean(su_values)
    consistency_metric_std = np.std(su_values)
    return consistency_metric_mean, consistency_metric_std

def lowess(x, y, tau=0.5):
    """
    Locally Weighted Smoothing (LOWESS).

    Parameters:
        x (array): Independent variable.
        y (array): Dependent variable.
        tau (float): The smoothing parameter, also called bandwidth or span.

    Returns:
        y_smoothed (array): Smoothed values of y.
    """
    n = len(x)
    y_smoothed = np.zeros(n)
    x = np.linspace(0, 1, len(x))

    for i in range(n):
        # Compute weights
        weights = np.exp(-((x - x[i]) ** 2) / (2 * tau ** 2))

        # Diagonal weight matrix
        W = np.diag(weights)

        # Design matrix
        X = np.column_stack((np.ones(n), x))

        # Compute parameter estimates
        theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)

        # Predict y value at x[i]
        y_smoothed[i] = np.dot([1, x[i]], theta)

    return y_smoothed
