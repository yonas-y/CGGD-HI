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

