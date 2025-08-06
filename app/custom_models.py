import tensorflow as tf
import keras
import numpy as np
from app.active_config import cfg
from app.metrics import custom_differentiable_spearman_corr_loss
from app.constraints import (compute_dir_monotonicity_custom_ordering,
                             compute_dir_prediction_mel_energy,
                             check_upper_bound, check_lower_bound)
from app.feature_partitioning import bound_indicators


class CustomModelMain(keras.Model):
    """
    Base custom Keras model that integrates domain-specific constraints and loss terms
    for health indicator (HI) estimation.

    Attributes:
        model (keras.Model): The core neural network architecture.
        rs_factor_recon (float): Rescale factor for reconstruction loss.
        rs_factor_softrank (float): Rescale factor for SoftRank loss.
        rs_factor_mono_lower (float): Rescale factor for lower monotonicity constraint.
        rs_factor_mono_upper (float): Rescale factor for upper monotonicity constraint.
        rs_factor_ene_deviation (float): Rescale factor for energy deviation constraint.
        rs_factor_up_bnd (float): Rescale factor for upper bound constraint.
        rs_factor_lw_bnd (float): Rescale factor for lower bound constraint.
        up_max_cutoff (float): Absolute upper cutoff for HI prediction.
        up_cutoff (float): Soft upper threshold to guide HI.
        lw_min_cutoff (float): Absolute lower cutoff for HI prediction.
        lw_cutoff (float): Soft lower threshold to guide HI.
        spear_regu_strength (float): Strength of Spearman's correlation regularization.
        train_loss_tracker (keras.metrics.Mean): Tracks average training loss.
        val_loss_tracker (keras.metrics.Mean): Tracks average validation loss.
        optimizer (keras.optimizers.Optimizer): Optimizer used for training.

    Note:
        This class handles the common logic and constraint initialization.
        Extend it in a subclass to customize training, add extra constraints, or override methods.
    """

    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model

        # Load constraint hyperparameters dynamically from config
        params = cfg.constraint_params

        self.rs_factor_recon = params.reconstruction_rf
        self.rs_factor_softrank = params.soft_rank_rf
        self.rs_factor_mono_lower = params.monotonicity_rf[0]
        self.rs_factor_mono_upper = params.monotonicity_rf[1]
        self.rs_factor_ene_deviation = params.energy_hi_dev_rf
        self.rs_factor_up_bnd = params.upper_bound_rf
        self.rs_factor_lw_bnd = params.lower_bound_rf

        # Bounds and cutoffs
        self.up_max_cutoff = params.max_cutoff
        self.up_cutoff = params.upper_cutoff
        self.lw_min_cutoff = params.min_cutoff
        self.lw_cutoff = params.lower_cutoff

        # Regularization
        self.spear_regu_strength = params.spearmans_regularization

        # Metrics
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipvalue=1.0)


class CustomModel(CustomModelMain):
    """
    Specialized custom model that extends `CustomModelMain`.

    This subclass can override methods like `train_step` or add new domain-specific
    constraints and custom training logic, building on the base configuration.

    Use this class when you need to:
    - Add new loss terms or constraints.
    - Change training or validation steps.
    - Implement experiment-specific behavior.

    Currently, this subclass inherits all behavior directly but is structured
    for future extensions.
    """

    def __init__(self, model: keras.Model):
        super().__init__(model=model)

    def custom_train_step_ConvAE(self, train_mel_feat, train_y, order_train, energy_train, energy_range, run_train):
        """
        Custom training with differentiable spearmans correlation as an objective function.

        Parameters
        ----------
        train_mel_feat : The input training data!
        train_y : The true RUL value [between 1 and 0]!
        order_train : The position of the sample in the run!
        energy_train : The total Mel spectral energy of the sample!
        energy_range : The range of the Mel spectral energy for normalization!
        run_train : From which run the sample in the batch came from!

        Returns
        ----------
        train_mono_correlation : The monotonicity satisfaction ratio!
        per_upper_bnd_sat : The upper bound satisfaction ratio!
        per_lower_bnd_sat : The lower bound satisfaction ratio!
        energy_deviation_sat : The energy deviation satisfaction ratio!
        """
        with (tf.GradientTape(persistent=True) as tape):
            predictions, encoding_out, AE_out = self.model(train_mel_feat, training=True)
            recon_loss = tf.cast(tf.reduce_mean(tf.square(train_mel_feat - AE_out)), dtype=tf.float32)

            # Get unique run identifiers!
            nu_of_run, _ = tf.unique(tf.reshape(run_train, [-1]))

            # Soft rank computation!
            if self.rs_factor_softrank == 0.0:
                soft_rank_loss = tf.constant(0, dtype=tf.float32)
            else:
                # Initialize lists for losses and metrics
                soft_rank_loss_full = []
                for i in range(tf.size(nu_of_run)):
                    # Use tf.gather to extract entries based on indices
                    tar_indices = tf.where(tf.equal(tf.reshape(run_train, [-1]), nu_of_run[i]))
                    tar_indices = tf.reshape(tar_indices, [-1])

                    # Use tf.gather to extract entries based on indices
                    sel_predictions = tf.gather(predictions, tar_indices)
                    sel_order_train = tf.gather(order_train, tar_indices)

                    # Compute soft rank loss if enabled!
                    soft_rank_loss_inter = custom_differentiable_spearman_corr_loss(
                        sel_predictions, sel_order_train, regularization_strength=self.spear_regu_strength)
                    soft_rank_loss_full.append(soft_rank_loss_inter)

                # Aggregate soft rank results!
                soft_rank_loss = tf.cast(tf.reduce_mean(soft_rank_loss_full), dtype=tf.float32)

            # Calculate the total weighted loss as a sum of the two losses!
            total_init_loss = self.rs_factor_softrank * soft_rank_loss + self.rs_factor_recon * recon_loss

            # Compute boundary indicators!
            train_bnd_ind, train_upper_bnd, train_lower_bnd = bound_indicators(
                train_y, max_up_bnd=self.up_max_cutoff, upper_cutoff=self.up_cutoff,
                min_lwr_bnd=self.lw_min_cutoff, lower_cutoff=self.lw_cutoff)

            # Compute energy indicator!
            energy_indicator = tf.cast(tf.where(tf.equal(train_bnd_ind, 0), tf.ones_like(train_bnd_ind),
                                        tf.zeros_like(train_bnd_ind)), dtype=tf.float32)

            with tape.stop_recording():
                recon_loss_grad_latent = tape.gradient(total_init_loss, encoding_out,
                                                       unconnected_gradients=tf.UnconnectedGradients.ZERO)
                norm_recon_loss_grad_latent = tf.math.reduce_euclidean_norm(recon_loss_grad_latent, axis=1,
                                                                            keepdims=True)
                norm_recon_loss_grad_latent = tf.math.maximum(tf.constant(1e-6, dtype=tf.float32),
                                                              norm_recon_loss_grad_latent)

                # Helper function for conditional constraint computations
                def compute_constraints(condition, func, *args):
                    if condition != 0:
                        return func(*args)
                    prediction = args[0]
                    return tf.zeros_like(prediction), tf.constant(0, dtype=tf.float32)

                # Compute the direction of the constraints!
                dir_wrong_mono_full, per_mono_satisfaction = compute_constraints(
                    self.rs_factor_mono_upper, compute_dir_monotonicity_custom_ordering, predictions,
                    order_train, run_train
                )
                batch_size = len(dir_wrong_mono_full) - 1
                mono_direction = (dir_wrong_mono_full / batch_size * (
                        self.rs_factor_mono_upper - self.rs_factor_mono_lower)
                                  + self.rs_factor_mono_lower * tf.math.sign(dir_wrong_mono_full))

                dir_wrong_prediction_energy, energy_deviation_sat = compute_constraints(
                    self.rs_factor_ene_deviation, compute_dir_prediction_mel_energy,
                    predictions, energy_train, energy_range, energy_indicator, 1.0)
                upper_bnd_con, per_upper_bnd_sat = compute_constraints(
                    self.rs_factor_up_bnd, check_upper_bound, predictions, train_upper_bnd)
                lower_bnd_con, per_lower_bnd_sat = compute_constraints(
                    self.rs_factor_lw_bnd, check_lower_bound, predictions, train_lower_bnd)

            # Multiply the constraints with the predictions!
            def apply_constraints(condition, predictions, direction):
                return tf.math.multiply(predictions, direction) if condition != 0 else tf.zeros_like(predictions)

            applied_dir_con_mono = apply_constraints(self.rs_factor_mono_upper, predictions,
                                                     tf.math.sign(dir_wrong_mono_full))
            applied_dir_con_energy = apply_constraints(self.rs_factor_ene_deviation, predictions,
                                                       dir_wrong_prediction_energy)
            applied_dir_con_upper = apply_constraints(self.rs_factor_up_bnd, predictions, upper_bnd_con)
            applied_dir_con_lower = apply_constraints(self.rs_factor_lw_bnd, predictions, lower_bnd_con)

            with tape.stop_recording():
                def compute_gradients(condition, applied_dir_con, encoding):
                    if condition != 0:
                        f_mh = tape.gradient(applied_dir_con, encoding,
                                             unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        f_mh_normalized = tf.math.reduce_euclidean_norm(f_mh, axis=1, keepdims=True)
                        f_mh_normalized = tf.where(tf.math.is_nan(f_mh_normalized), tf.constant(1, dtype=tf.float32),
                                                   f_mh_normalized)
                        f_mh_normalized = tf.math.maximum(tf.constant(1e-6, dtype=tf.float32), f_mh_normalized)
                        return f_mh_normalized
                    return tf.zeros_like(encoding)

                f_mh_mono = compute_gradients(self.rs_factor_mono_upper, applied_dir_con_mono, encoding_out)
                f_mh_energy = compute_gradients(self.rs_factor_ene_deviation, applied_dir_con_energy, encoding_out)
                f_mh_upper = compute_gradients(self.rs_factor_up_bnd, applied_dir_con_upper, encoding_out)
                f_mh_lower = compute_gradients(self.rs_factor_lw_bnd, applied_dir_con_lower, encoding_out)

            # Compute constraint loss components!!
            mono_component = (tf.multiply(f_mh_mono / norm_recon_loss_grad_latent, mono_direction)
                              if self.rs_factor_mono_upper != 0 else 0.0)
            energy_component = (tf.multiply((f_mh_energy / norm_recon_loss_grad_latent * self.rs_factor_ene_deviation),
                                            dir_wrong_prediction_energy) if self.rs_factor_ene_deviation != 0 else 0.0)
            upper_component = (tf.multiply((f_mh_upper / norm_recon_loss_grad_latent * self.rs_factor_up_bnd),
                                           upper_bnd_con) if self.rs_factor_up_bnd != 0 else 0.0)
            lower_component = (tf.multiply((f_mh_lower / norm_recon_loss_grad_latent * self.rs_factor_lw_bnd),
                                           lower_bnd_con) if self.rs_factor_lw_bnd != 0 else 0.0)

            constraint_loss = tf.math.reduce_mean(
                tf.math.multiply(norm_recon_loss_grad_latent,
                                 tf.math.multiply(predictions,
                                                  tf.math.add(mono_component,
                                                              tf.math.add(energy_component,
                                                                          tf.math.add(upper_component,
                                                                                      lower_component))))))
            combined_loss = tf.math.add(total_init_loss, constraint_loss)
        grad_total = tape.gradient(combined_loss, self.model.trainable_variables,
                                   unconnected_gradients=tf.UnconnectedGradients.ZERO)

        del tape

        self.optimizer.apply_gradients(zip(grad_total, self.model.trainable_variables))

        return (recon_loss, soft_rank_loss, per_mono_satisfaction, per_upper_bnd_sat,
                per_lower_bnd_sat, energy_deviation_sat)

    def custom_test_step_ConvAE(self, val_mel, val_y, val_order, val_energy, energy_range, run_val):
        """
        Custom testing step.

        Parameters
        ----------
        val_mel : The input test/validation data!
        val_y : The true RUL value [between 1 and 0]!
        val_order : The position of the sample in the run!
        val_energy : The total Mel spectral energy of the sample!
        energy_range : The range of the Mel spectral energy for normalization!
        run_val : From which run the sample in the batch came from!

        Returns
        ----------
        val_mono_correlation : The monotonicity satisfaction ratio!
        val_per_upper_bnd_sat : The upper bound satisfaction ratio!
        val_per_lower_bnd_sat : The lower bound satisfaction ratio!
        val_energy_deviation_sat : The energy deviation satisfaction ratio!
        """
        val_predictions, val_encoding_out, val_AE_out = self.model(val_mel, training=False)
        val_recon_loss = tf.reduce_mean(tf.square(val_mel - val_AE_out))

        nu_of_run, _ = tf.unique(tf.reshape(run_val, [-1]))

        val_bnd_ind, val_upper_bnd, val_lower_bnd = bound_indicators(
            val_y, max_up_bnd=self.up_max_cutoff, upper_cutoff=self.up_cutoff,
            min_lwr_bnd=self.lw_min_cutoff, lower_cutoff=self.lw_cutoff)

        energy_indicator_val = tf.cast(tf.where(tf.equal(val_bnd_ind, 0), tf.ones_like(val_bnd_ind),
                                        tf.zeros_like(val_bnd_ind)), dtype=tf.float32)

        # Helper function for conditional constraint computations
        def compute_constraints_satisfaction(condition, func, *args):
            if condition != 0:
                return func(*args)
            prediction = args[0]
            return tf.zeros_like(prediction), tf.constant(0, dtype=tf.float32)

        # Constraint satisfaction ratios!!
        _, val_per_mono_satisfaction = compute_constraints_satisfaction(
            self.rs_factor_mono_upper, compute_dir_monotonicity_custom_ordering,
            val_predictions, val_order, run_val)
        _, val_energy_deviation_sat = compute_constraints_satisfaction(
            self.rs_factor_ene_deviation, compute_dir_prediction_mel_energy,
            val_predictions, val_energy, energy_range, energy_indicator_val, 1.0)
        _, val_per_upper_bnd_sat = compute_constraints_satisfaction(
            self.rs_factor_up_bnd, check_upper_bound, val_predictions, val_upper_bnd)
        _, val_per_lower_bnd_sat = compute_constraints_satisfaction(
            self.rs_factor_lw_bnd, check_lower_bound, val_predictions, val_lower_bnd)

        # Soft rank computation!
        if self.rs_factor_softrank == 0.0:
            val_soft_rank_loss = tf.constant(0, dtype=tf.float32)
        else:
            val_soft_rank_loss_full = []
            for i in range(tf.size(nu_of_run)):
                tar_indices = tf.where(tf.equal(run_val, nu_of_run[i]))
                tar_indices = tf.reshape(tar_indices, [-1])

                # Use tf.gather to extract entries based on indices
                sel_val_predictions = tf.gather(val_predictions, tar_indices)
                sel_order_val = tf.gather(val_order, tar_indices)

                val_soft_rank_loss_inter = custom_differentiable_spearman_corr_loss(
                    sel_val_predictions, sel_order_val, regularization_strength=self.spear_regu_strength)
                val_soft_rank_loss_full.append(val_soft_rank_loss_inter)

            val_soft_rank_loss = tf.reduce_mean(val_soft_rank_loss_full)

        return (val_recon_loss, val_soft_rank_loss, val_per_mono_satisfaction, val_per_upper_bnd_sat,
                val_per_lower_bnd_sat, val_energy_deviation_sat)
