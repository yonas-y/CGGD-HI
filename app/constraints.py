import tensorflow as tf
import keras
import numpy as np

def compute_dir_monotonicity_custom_ordering(predictions, true_ranking, sample_run):
    """
    Check the monotonicity constraint for the given predictions. This function assumes that the
    prediction is 1 dimensional for each example in the
    batch and the ranking is given as input and not (necessarily) chronically.

    Parameters
    ----------
    predictions : tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32 denoting the prediction for
        each element in the batch.
    true_ranking  : tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32 denoting the ranking for
        each element in the batch.
    sample_run : tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32 denoting from which run the
        sample came from.

    Returns
    ----------
    dir_wrong_ranking : tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32 denoting the direction for the wrong ranked elements.
    percentage_satisfaction: tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32 satisfaction ratio.
    """
    nu_of_run, _ = tf.unique(tf.reshape(sample_run, [-1]))
    output_array = np.zeros((predictions.shape[0], 1))
    percentage_satisfaction_list = []
    for i in range(tf.size(nu_of_run)):
        tar_indices = tf.where(tf.equal(tf.reshape(sample_run, [-1]), nu_of_run[i]))
        tar_indices = tf.reshape(tar_indices, [-1])

        # Use tf.gather to extract entries based on indices
        sel_predictions = tf.gather(predictions, tar_indices)
        sel_true_ranking = tf.gather(true_ranking, tar_indices)

        number_to_rank = sel_predictions.shape[0]
        ranking = tf.cast(tf.argsort(tf.argsort(sel_true_ranking, axis=0), axis=0), dtype=tf.float32)
        indexed_ranks = tf.cast(
            tf.argsort(tf.argsort(sel_predictions, axis=0, direction='DESCENDING'), axis=0,
                       direction='ASCENDING'), dtype=tf.float32)
        dir_wrong_ranking = tf.math.subtract(tf.cast(ranking, dtype=tf.float32),
                                            tf.cast(indexed_ranks, dtype=tf.float32))
        # Insert the direction wrong ranking in the array!
        output_array[tar_indices.numpy()] = dir_wrong_ranking.numpy()

        number_sat = tf.reduce_sum(tf.cast(tf.equal(dir_wrong_ranking, 0), tf.float32))
        percentage_sat = tf.math.divide_no_nan(number_sat, number_to_rank)
        percentage_satisfaction_list.append(percentage_sat.numpy())

    dir_wrong_ranking_full = tf.convert_to_tensor(output_array, dtype=tf.float32)
    percentage_satisfaction = tf.reduce_mean(percentage_satisfaction_list)

    return dir_wrong_ranking_full, percentage_satisfaction

def compute_dir_prediction_mel_energy(predictions, sample_energy, energy_range, energy_indicator, tuning_param=1):
    """
    A constraint to check and penalize the discrepancy between the feature energy and corresponding prediction.

    Parameters
    ----------
    predictions : tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32 denoting the prediction for
        each element in the batch.
    sample_energy : tf.Tensor denoting the total mel energy (in log scale) of the sample!
    energy_range  : list  Contains the minimum and maximum energy range for the training set!
    energy_indicator : contains an indicator for which parts of the run to implement the constraint!
                        The constraint is not implemented for the start and end sections of the run!
    tuning_param :
        A tf.Tensor of shape=(1) :

    Returns
    ----------
    dir_predict_energy_difference : tf.Tensor
        A tf.Tensor of shape=(batch_size, 1) of dtype=tf.float32
            denoting the direction difference between the prediction and the energy observed.
    """
    # Normalize the observed energy between 0 and 1!
    energy_norm_difference = tf.math.abs((sample_energy - energy_range[0]) / (energy_range[1] - energy_range[0]))
    predictions_difference = tf.cast(1 - predictions, dtype=tf.float32)

    condition1 = tf.less(predictions_difference, 0)  # Predictions > 1!
    condition2 = tf.logical_and(tf.greater_equal(predictions_difference, 0),
                                tf.greater(predictions_difference,
                                           tuning_param * tf.math.maximum(energy_norm_difference,0.25)))  # Predictions < 1 and greater than energy threshold!
    # If the energies are equal, set some flexible delta (for example 0.25), so that the HI's are not set to equal values.
    dir_wrong_predict_energy = tf.where(condition1, tf.ones_like(predictions_difference),
                                        tf.where(condition2, -tf.ones_like(predictions_difference),
                                                 tf.zeros_like(predictions_difference)))
    # dir_wrong_predict_energy = tf.where(condition1, -tf.ones_like(predictions_difference),
    #                                     tf.where(condition2, tf.ones_like(predictions_difference),
    #                                              tf.zeros_like(predictions_difference)))
    dir_wrong_predict_energy = tf.math.multiply(energy_indicator, dir_wrong_predict_energy)
    count_total_con_sat = tf.math.reduce_sum(
        tf.where(dir_wrong_predict_energy == 0, tf.constant(1, dtype=tf.float32),
                 tf.constant(0, dtype=tf.float32)), axis=None)
    energy_deviation_satisfaction = tf.math.divide_no_nan(count_total_con_sat, dir_wrong_predict_energy.shape[0])

    # dir_wrong_predict_energy_norm = tf.math.multiply(dir_wrong_predict_energy, tf.math.abs(predictions_difference))
    # dir_wrong_predict_energy_norm = tf.math.l2_normalize(dir_wrong_predict_energy_norm)

    return dir_wrong_predict_energy, energy_deviation_satisfaction