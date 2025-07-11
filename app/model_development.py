import tensorflow as tf
import numpy as np
import random


class ConvAE_model_subclass(tf.keras.Model):
    def __init__(self, input_shape=None,
                 encoding_neurons=8,
                 reg=1e-3, dropout=0.0,
                 poolsize=2, kernelsize=3, stride=1,
                 activation=tf.keras.layers.LeakyReLU(negative_slope=0.01), **kwargs):

        super(ConvAE_model_subclass, self).__init__(**kwargs)

        self.training = True
        self.bias = True

        self.inputshape = input_shape
        self.en = min(encoding_neurons, 64)

        self.n_enc_layers = 4
        self.decoder_dim = (self.inputshape[0] // (2 ** self.n_enc_layers), self.en)

        self.kernel_size = kernelsize
        self.pool_size = poolsize
        self.stride_pool = 2
        self.stride_conv = stride

        self.reg = tf.keras.regularizers.l2(reg)
        self.activation = tf.keras.layers.Activation(activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.kernel_init = tf.keras.initializers.GlorotNormal(seed=random.randint(0, 1e6))
        self.bias_init = tf.keras.initializers.GlorotNormal(seed=random.randint(0, 1e6))
        self.down_sample = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, strides=self.stride_pool)

        # --- Encoder ---
        # Four convolutional blocks: Conv1D → BN → Activation → Dropout → Pool
        self.encoder_blocks = []
        filter_sizes = [self.en ** 2, self.en * 2, self.en * 2, self.en]
        for filters in filter_sizes:
            block = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=filters, kernel_size=self.kernel_size,
                                       strides=self.stride_conv, padding='same',
                                       kernel_regularizer=self.reg,
                                       kernel_initializer=self.kernel_init, bias_initializer=self.bias_init),
                tf.keras.layers.BatchNormalization(),
                self.activation,
                self.dropout
            ])
            self.encoder_blocks.append(block)

        # Flatten + Dense layer to get latent vector
        self.flatten = tf.keras.layers.Flatten()
        self.encoder_dense = tf.keras.layers.Dense(encoding_neurons, activation='linear',
                                                   kernel_regularizer=self.reg,
                                                   kernel_initializer=self.kernel_init,
                                                   bias_initializer=self.bias_init)

        # --- HI regression head ---
        # Simple dense stack predicting scalar HI value from latent vector
        self.hi_branch = tf.keras.Sequential([
            tf.keras.layers.Dense(self.en * 2, activation='linear', kernel_regularizer=self.reg),
            self.dropout,
            tf.keras.layers.Dense(self.en, activation='linear', kernel_regularizer=self.reg),
            self.dropout,
            tf.keras.layers.Dense(self.en, activation='linear', kernel_regularizer=self.reg),
            self.dropout,
            tf.keras.layers.Dense(units=1, activation='linear', use_bias=True, name='HI_Output')
        ])

        # --- Decoder ---
        # Dense layer to expand latent vector, then reshape to 3D
        self.decoder_dense = tf.keras.layers.Dense(units=self.decoder_dim[0] * self.decoder_dim[1],
                                                   activation='linear', kernel_regularizer=self.reg,
                                                   kernel_initializer=self.kernel_init,
                                                   bias_initializer=self.bias_init)
        self.reshape_layer = tf.keras.layers.Reshape((self.decoder_dim[0], self.decoder_dim[1]))

        # Four Conv1DTranspose blocks: ConvT → BN → Activation → Dropout
        self.decoder_blocks = []
        decoder_filter_sizes = [self.en, self.en * 2, self.en * 2, self.en ** 2]
        for filters in decoder_filter_sizes:
            block = tf.keras.Sequential([
                tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=self.kernel_size,
                                                strides=self.pool_size, padding='same',
                                                kernel_regularizer=self.reg,
                                                kernel_initializer=self.kernel_init,
                                                bias_initializer=self.bias_init),
                tf.keras.layers.BatchNormalization(),
                self.activation,
                self.dropout
            ])
            self.decoder_blocks.append(block)

        # Layer of Block 11
        self.decoder_out = tf.keras.layers.Conv1D(filters=self.inputshape[-1], kernel_size=self.kernel_size,
                                                  strides=self.stride_conv, padding='same',
                                                  kernel_regularizer=self.reg, kernel_initializer=self.kernel_init,
                                                  bias_initializer=self.bias_init, activation='relu',
                                                  name='Dec_out')

    def call(self, x, training=None):
        # --- Encoder ---
        for block in self.encoder_blocks:
            x = block(x, training=training)
            x = self.down_sample(x)

        latent = self.encoder_dense(self.flatten(x))

        # --- HI branch ---
        HI_out = self.hi_branch(latent)

        # --- Decoder ---
        z = self.decoder_dense(latent)
        z = self.reshape_layer(z)

        for block in self.decoder_blocks:
            z = block(z, training=training)

        ae_out = self.decoder_out(z)

        return HI_out, latent, ae_out

    def model(self):
        """
        Builds a functional Keras Model for visualization or summary.

        Returns:
            tf.keras.Model
        """
        x = tf.keras.Input(shape=self.inputshape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def set_training(self, training=True):
        self.training = training

    def get_training(self):
        return self.training
