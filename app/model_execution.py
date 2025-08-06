import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
from typing import List
import logging

from app.active_config import cfg
from app.custom_models import CustomModel

logger = logging.getLogger(__name__)

def model_execution(model,
                    training_data: List,
                    validation_data: List,
                    iteration: int) -> pd.DataFrame:
    """
    The function executes the custom model!
    :param model: the model architecture used.
    :param training_data: the input training data features!
    :param validation_data: the input validation data features!
    :param iteration: the number of iterations!

    :return:The performance of the model as a dataframe.
    """

    epochs = cfg.model_training_params.epochs
    patience = cfg.model_training_params.patience
    save_weights = cfg.model_training_params.save_weights
    early_stopping = cfg.model_training_params.early_stop

    recon_rescale = cfg.constraint_params.reconstruction_rf
    softrank_rescale = cfg.constraint_params.soft_rank_rf
    mono_rescale = cfg.constraint_params.monotonicity_rf
    ene_dev_rescale = cfg.constraint_params.energy_hi_dev_rf
    upper_rescale = cfg.constraint_params.upper_bound_rf
    lower_rescale = cfg.constraint_params.lower_bound_rf
    ene_min = cfg.constraint_params.min_scaled_energy
    ene_max = cfg.constraint_params.max_scaled_energy

    # Helper function to save model weights
    def save_model_weights(loss_value, best_val, wait_val):
        if loss_value < best_val:
            # directory to save models
            model_weight_dir = os.path.join(cfg.OUTPUT_DIR, "model_weights")
            os.makedirs(model_weight_dir, exist_ok=True)

            # consistent filename
            final_filename = os.path.join(
                model_weight_dir,
                f"Custom_Model_{cfg.model_hyperparams.encoding_n}_{cfg.SETUP_Name}_"
                f"{cfg.bearing_used}_{recon_rescale}_{softrank_rescale}_{mono_rescale[1]}_{ene_dev_rescale}_"
                f"{upper_rescale}_{lower_rescale}_{iteration}_recon_val.h5"
            )

            # Save weights to the temp file
            model.save(final_filename)
            logger.info(f"☁️ Model saved to: {final_filename}")

            best_val = loss_value
            wait_val = 0
        else:
            wait_val += 1  # increase patience if no improvement

        return best_val, wait_val

    def process_train_step(model_in, mel_train, y_train, order_train, energy_train, run_train_v, ene_max, ene_min):
        """Process a training step based on the different approaches."""

        train_recon_loss, train_softrank, train_m_corr, per_UB_sat, per_LB_sat, per_E_sat = (
            model_in.custom_train_step_ConvAE(mel_train, y_train, order_train, energy_train,
                                              [ene_min, ene_max], run_train_v))
        return (train_recon_loss.numpy(), train_softrank.numpy(), train_m_corr.numpy(), per_UB_sat.numpy(),
                per_LB_sat.numpy(), per_E_sat.numpy())

    def process_test_step(model_in, mel_val, y_val, order_val, energy_val, run_val_v, ene_max, ene_min):
        """Process a training step based on the different approaches."""
        val_r_loss, val_s_loss, val_m_corr, per_UB_sat_val, per_LB_sat_val, per_E_sat_val = (
            model_in.custom_test_step_ConvAE(mel_val, y_val, order_val, energy_val, [ene_min, ene_max], run_val_v))
        return (val_r_loss.numpy(), val_s_loss.numpy(), val_m_corr.numpy(), per_UB_sat_val.numpy(),
                per_LB_sat_val.numpy(), per_E_sat_val.numpy())

    def update_metrics(train_lists, metrics):
        """Update train lists based on available metrics."""
        for key_v, metric_v in metrics.items():
            if metric_v is not None:
                train_lists[key_v].append(metric_v)

    # Instantiate the model!
    Custom_Model = CustomModel(model=model)

    # Define the variable names
    epoch_metrics = [
        "train_reconstruction_loss_epoch", "train_soft_rank_loss_epoch", "train_mono_corr_epoch",
        "train_ene_pred_deviation_epoch", "train_upper_bnd_sat_epoch",
        "train_lower_bnd_sat_epoch",
        "val_reconstruction_loss_epoch", "val_soft_rank_loss_epoch", "val_mono_corr_epoch",
        "val_ene_pred_deviation_epoch", "val_upper_bnd_sat_epoch",
        "val_lower_bnd_sat_epoch"
    ]

    # Initialize an empty list for each metric
    epoch_lists = {name: [] for name in epoch_metrics}
    best_l, wait = float(np.inf), 0

    for epoch in range(epochs):
        logger.info(f"⚙️ Model training and validation phase for epoch {epoch + 1}!")
        t = time.time()

        train_metrics_names = [
            "train_recon_loss_l", "train_soft_rank_loss_l", "train_mono_correlation_l",
            "train_ene_pred_deviation_l", "train_per_upper_bnd_sat_l", "train_per_lower_bnd_sat_l"]
        # Initialize an empty list for each metric
        train_metric_lists = {name: [] for name in train_metrics_names}

        # Iterate over the batches of the train dataset.
        for batch in range(len(training_data)):
            input_batch = training_data[batch]

            br_fs_mel_train, ene_rul_order_train, run_train = input_batch
            br_mel_energy_train, br_RUL_train, br_order_train = (ene_rul_order_train[:, 0], ene_rul_order_train[:, 1],
                                                                 ene_rul_order_train[:, 2])

            # Cast inputs to float32
            mel_batch_train = tf.cast(br_fs_mel_train, dtype=tf.float32)
            y_batch_train = tf.cast(tf.reshape(br_RUL_train, [-1, 1]), dtype=tf.float32)
            order_batch_train = tf.cast(tf.reshape(br_order_train, [-1, 1]), dtype=tf.float32)
            energy_batch_train = tf.cast(tf.reshape(br_mel_energy_train, [-1, 1]), dtype=tf.float32)
            run_batch_train = tf.cast(run_train, dtype=tf.float32)

            # Call the training function!
            (recon_loss, softrank_loss, train_mono_correlation, per_upper_bnd_sat,
             per_lower_bnd_sat, per_ene_deviation_sat) = process_train_step(
                Custom_Model, mel_batch_train, y_batch_train, order_batch_train, energy_batch_train,
                run_batch_train, ene_max, ene_min)

            # Update metric lists with only available metrics
            update_metrics(train_metric_lists, {
                'train_recon_loss_l': recon_loss,
                'train_soft_rank_loss_l': softrank_loss,
                'train_mono_correlation_l': train_mono_correlation,
                'train_ene_pred_deviation_l': per_ene_deviation_sat,
                'train_per_upper_bnd_sat_l': per_upper_bnd_sat,
                'train_per_lower_bnd_sat_l': per_lower_bnd_sat
            })

        val_metrics_names = [
            "val_recon_loss_l", "val_soft_rank_loss_l", "val_mono_correlation_l",
            "val_ene_pred_deviation_l", "val_per_upper_bnd_sat_l", "val_per_lower_bnd_sat_l"]
        # Initialize an empty list for each metric
        val_metric_lists = {name: [] for name in val_metrics_names}

        # Iterate over the batches of the validation dataset.
        for batch in range(len(validation_data)):
            input_val_batch = validation_data[batch]

            br_fs_mel_val, ene_rul_order_val, run_val = input_val_batch
            br_RUL_val, br_order_val, br_mel_energy_val = (ene_rul_order_val[:,0], ene_rul_order_val[:,1],
                                                                 ene_rul_order_val[:,2])

            # Cast inputs to float32
            mel_batch_val = tf.cast(br_fs_mel_val, dtype=tf.float32)
            y_batch_val = tf.cast(tf.reshape(br_RUL_val, [-1, 1]), dtype=tf.float32)
            order_batch_val = tf.cast(tf.reshape(br_order_val, [-1, 1]), dtype=tf.float32)
            energy_batch_val = tf.cast(tf.reshape(br_mel_energy_val, [-1, 1]), dtype=tf.float32)
            run_batch_val = tf.cast(run_val, dtype=tf.float32)

            # Call the test function for validation!
            (val_recon_loss, val_softrank_loss, val_mono_correlation, val_per_upper_bnd_sat,
             val_per_lower_bnd_sat, val_per_ene_deviation_sat) = process_test_step(
                Custom_Model, mel_batch_val, y_batch_val, order_batch_val, energy_batch_val,
                run_batch_val, ene_max, ene_min)

            # Update metric lists with only available metrics
            update_metrics(val_metric_lists, {
                'val_recon_loss_l': val_recon_loss,
                'val_soft_rank_loss_l': val_softrank_loss,
                'val_mono_correlation_l': val_mono_correlation,
                'val_ene_pred_deviation_l': val_per_ene_deviation_sat,
                'val_per_upper_bnd_sat_l': val_per_upper_bnd_sat,
                'val_per_lower_bnd_sat_l': val_per_lower_bnd_sat
            })

        # Compute mean metrics for each training epoch!
        train_mean_metrics = {key: np.nanmean(np.array(train_metric_lists[key])) for key in train_metric_lists}

        # Compute mean metrics for each validation epoch!
        val_mean_metrics = {key: np.nanmean(np.array(val_metric_lists[key])) for key in val_metric_lists}

        # Insert computed mean metrics into epoch_lists using a mapping
        merged_dict = {**train_mean_metrics, **val_mean_metrics}
        for mer_key, epoch_key in zip(merged_dict.keys(), epoch_metrics):
            epoch_lists[epoch_key].append(merged_dict[mer_key])

        # The early stopping strategy: stop the training if the average of the bounds satisfaction increases.
        # To reduce the effects of early performance fluctuations, start the loss calculation after a few epoches.
        if save_weights and epoch > 15:
            overall_loss = (
                    train_mean_metrics['train_recon_loss_l'] +
                    train_mean_metrics['train_soft_rank_loss_l'] +
                    (1 - train_mean_metrics['train_mono_correlation_l']) +
                    (1 - train_mean_metrics['train_ene_pred_deviation_l']) +
                    (1 - train_mean_metrics['train_per_upper_bnd_sat_l']) +
                    (1 - train_mean_metrics['train_per_lower_bnd_sat_l'])
            )
            best = best_l
            best_l, wait = save_model_weights(overall_loss, best, wait)

        # The early stopping strategy: stop the training no more performance improvement.
        if early_stopping:
            if wait >= patience:
                logger.info("⚠️ Early Stopping! ⚠️ No more performance improvement!")
                break

        logger.info(f"🔄 ETA: {round((time.time() - t) / 60, 2)} - epoch: {epoch + 1} \n"
              f"train_recon_loss: {train_mean_metrics['train_recon_loss_l']} \n"
              f"train_soft_rank_loss: {train_mean_metrics['train_soft_rank_loss_l']} \n"
              f"train_mono_correlation: {train_mean_metrics['train_mono_correlation_l']} \n"
              f"train_energy_pred_deviation: {train_mean_metrics['train_ene_pred_deviation_l']} \n"
              f"train_upper_bnd_sat: {train_mean_metrics['train_per_upper_bnd_sat_l']} \n"
              f"train_lower_bnd_sat: {train_mean_metrics['train_per_lower_bnd_sat_l']} \n"
              f"val_recon_loss: {val_mean_metrics['val_recon_loss_l']} \n"
              f"val_soft_rank_loss: {val_mean_metrics['val_soft_rank_loss_l']} \n"
              f"val_mono_correlation: {val_mean_metrics['val_mono_correlation_l']} \n"
              f"val_energy_pred_deviation: {val_mean_metrics['val_ene_pred_deviation_l']} \n"
              f"val_upper_bnd_sat: {val_mean_metrics['val_per_upper_bnd_sat_l']} \n"
              f"val_lower_bnd_sat: {val_mean_metrics['val_per_lower_bnd_sat_l']} \n ")

    # Define the mapping of columns for each train approach
    column_mapping = [
        'train_reconstruction_loss_epoch', 'train_soft_rank_loss_epoch', 'train_mono_corr_epoch',
        'train_ene_pred_deviation_epoch', 'train_upper_bnd_sat_epoch', 'train_lower_bnd_sat_epoch',
        'val_reconstruction_loss_epoch', 'val_soft_rank_loss_epoch', 'val_mono_corr_epoch',
        'val_ene_pred_deviation_epoch', 'val_upper_bnd_sat_epoch', 'val_lower_bnd_sat_epoch'
    ]

    # Create the data dictionary using a dictionary comprehension
    data = {key: epoch_lists[key] for key in column_mapping}
    # Create the DataFrame directly from the dictionary
    performance_df = pd.DataFrame(data)

    return performance_df
