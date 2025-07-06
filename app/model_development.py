import tensorflow as tf
import numpy as np
import random
from app.config import (input_sample_shape, encoding_n, regularization, dropout_rate, pooling_size,
                        kernel_size_conv, stride_convolutional, activation_function, kernel_init)
class ConvAEModel(tf.keras.Model):
    """
    Convolutional Autoencoder with an auxiliary regression head (HI branch).

    The model:
    - Encodes a 1D sequence into a compact latent representation
    - Decodes it back to reconstruct the input
    - Predicts a scalar health indicator (HI) value from the latent vector

    Attributes:
        input_shape_ (tuple): shape of each input sample (time_steps, channels)
        en (int): number of neurons in latent vector (limited to max 64)
        encoder_convs (list): list of conv+BN+Dropout blocks for encoder
        decoder_convs (list): list of Conv1DTranspose+BN+Dropout blocks for decoder
        hi_branch (tf.keras.Sequential): dense layers predicting HI value
    """
    def __init__(self,
                 input_shape = input_sample_shape,
                 encoding_neurons=encoding_n,
                 reg=regularization,
                 dropout=dropout_rate,
                 pool_size=pooling_size,
                 kernel_size=kernel_size_conv,
                 stride_conv=stride_convolutional,
                 activation=activation_function,
                 **kwargs):

        super().__init__(**kwargs)
        self.input_shape_ = input_shape
        self.en = min(encoding_neurons, 64)
        self.reg = tf.keras.regularizers.l2(reg)
        self.dropout_rate = dropout
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.stride_conv = stride_conv

        # Default activation: LeakyReLU with small negative slope
        self.activation = activation or tf.keras.layers.LeakyReLU(alpha=0.01)
        self.kernel_init = kernel_init
        self.bias_init = tf.keras.initializers.Zeros()

        # --- Encoder ---
        # Four convolutional blocks: Conv1D → BN → Activation → Dropout → Pool
        self.encoder_convs = []
        filter_sizes = [self.en**2, self.en*2, self.en*2, self.en]
        for filters in filter_sizes:
            self.encoder_convs.append([
                tf.keras.layers.Conv1D(filters=filters, kernel_size=self.kernel_size,
                                       strides=self.stride_conv, padding='same',
                                       kernel_regularizer=self.reg,
                                       kernel_initializer=self.kernel_init,
                                       bias_initializer=self.bias_init),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.dropout_rate)
            ])

        self.downsample = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)

        # Flatten + Dense layer to get latent vector
        self.flatten = tf.keras.layers.Flatten()
        self.encoder_dense = tf.keras.layers.Dense(self.en, activation='linear',
                                                   kernel_regularizer=self.reg)

        # --- HI regression head ---
        # Simple dense stack predicting scalar HI value from latent vector
        self.hi_branch = tf.keras.Sequential([
            tf.keras.layers.Dense(self.en*2, activation='linear', kernel_regularizer=self.reg),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.en, activation='linear', kernel_regularizer=self.reg),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.en, activation='linear', kernel_regularizer=self.reg),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='linear', name='HI_Output')
        ])

        # --- Decoder ---
        # Dense layer to expand latent vector, then reshape to 3D
        self.decoder_dense = tf.keras.layers.Dense(units=None, activation='linear')  # units filled dynamically
        self.reshape_layer = None  # shape filled in build()

        # Four Conv1DTranspose blocks: ConvT → BN → Activation → Dropout
        decoder_filter_sizes = [self.en, self.en*2, self.en*2, self.en**2]
        self.decoder_convs = []
        for filters in decoder_filter_sizes:
            self.decoder_convs.append([
                tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=self.kernel_size,
                                                strides=self.pool_size, padding='same',
                                                kernel_regularizer=self.reg,
                                                kernel_initializer=self.kernel_init,
                                                bias_initializer=self.bias_init),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(self.dropout_rate)
            ])

        # Final output conv layer to reconstruct original channels
        self.decoder_out = tf.keras.layers.Conv1D(filters=self.input_shape_[-1],
                                                  kernel_size=self.kernel_size,
                                                  strides=self.stride_conv,
                                                  padding='same', activation='relu',
                                                  kernel_regularizer=self.reg)

    def build(self, input_shape):
        """
        Computes decoder reshaping dimensions dynamically based on input shape.
        """
        time_steps = input_shape[1]
        n_down = len(self.encoder_convs)
        reduced_steps = time_steps
        for _ in range(n_down):
            reduced_steps = np.ceil(reduced_steps / self.pool_size)
        reduced_steps = int(reduced_steps)

        # Dense expands latent vector to match decoder input shape
        self.decoder_dense.units = reduced_steps * self.en
        self.reshape_layer = tf.keras.layers.Reshape((reduced_steps, self.en))
        super().build(input_shape)

    def call(self, x, training=None):
        """
        Forward pass:
        - Encoder: extract latent vector
        - HI branch: predict scalar HI value
        - Decoder: reconstruct input from latent vector

        Args:
            x: input tensor, shape (batch, time_steps, channels)
            training: bool, whether in training mode (controls dropout/BN)

        Returns:
            hi_out: predicted HI scalar (batch, 1)
            latent: latent vector (batch, encoding_neurons)
            ae_out: reconstructed sequence (batch, time_steps, channels)
        """
        # --- Encoder ---
        for conv, bn, drop in self.encoder_convs:
            x = conv(x)
            x = bn(x, training=training)
            x = self.activation(x)
            x = drop(x, training=training)
            x = self.downsample(x)

        latent = self.encoder_dense(self.flatten(x))

        # --- HI branch ---
        hi_out = self.hi_branch(latent, training=training)

        # --- Decoder ---
        z = self.decoder_dense(latent)
        z = self.reshape_layer(z)
        for convt, bn, drop in self.decoder_convs:
            z = convt(z)
            z = bn(z, training=training)
            z = self.activation(z)
            z = drop(z, training=training)

        ae_out = self.decoder_out(z)

        return hi_out, latent, ae_out

    def model(self):
        """
        Builds a functional Keras Model for visualization or summary.

        Returns:
            tf.keras.Model
        """
        x = tf.keras.Input(shape=self.input_shape_)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
