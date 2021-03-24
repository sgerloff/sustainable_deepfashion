import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, optimizers
from tensorflow.keras import backend as K
from src.utility import get_project_dir
import joblib, os
from src.data.VAE_dataset_factory import VAEDatasetFactory


class VAEModelFactory:
    def __init__(self, input_shape=(224, 224, 3), latent_dim=32,
                 filters_per_conv_layer=[8, 16, 32, 64, 128, 256, 512, 1024]):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.filters_per_conv_layer = filters_per_conv_layer
        self.bridge_shape = (self.input_shape[0] // 2 ** (len(self.filters_per_conv_layer) // 2),
                             self.input_shape[1] // 2 ** (len(self.filters_per_conv_layer) // 2),
                             self.filters_per_conv_layer[-1])

    def get_model(self):
        encoder_inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(self.filters_per_conv_layer[0], kernel_size=3, strides=2, padding='same', activation='relu')(
            encoder_inputs)
        x = layers.Conv2D(self.filters_per_conv_layer[1], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(self.filters_per_conv_layer[2], kernel_size=3, strides=2, padding='same', activation='relu')(
            x)
        x = layers.Conv2D(self.filters_per_conv_layer[3], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(self.filters_per_conv_layer[4], kernel_size=3, strides=2, padding='same', activation='relu')(
            x)
        x = layers.Conv2D(self.filters_per_conv_layer[5], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2D(self.filters_per_conv_layer[6], kernel_size=3, strides=2, padding='same', activation='relu')(
            x)
        x = layers.Conv2D(self.filters_per_conv_layer[7], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(2 * self.latent_dim, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(np.prod(self.bridge_shape), activation='relu')(latent_inputs)
        x = layers.Reshape(self.bridge_shape)(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[6], kernel_size=3, strides=2, padding='same',
                                   activation='relu')(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[5], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[4], kernel_size=3, strides=2, padding='same',
                                   activation='relu')(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[3], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[2], kernel_size=3, strides=2, padding='same',
                                   activation='relu')(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[1], kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(self.filters_per_conv_layer[0], kernel_size=3, strides=2, padding='same',
                                   activation='relu')(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
        decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

        return VAE(encoder, decoder)

    def preprocessor(self):
        return lambda x: x / 255.


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, x):
        _, _, z = self.encoder(x)
        y = self.decoder(z)
        return y

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == "__main__":
    train_df = joblib.load(os.path.join(get_project_dir(),
                                        "data",
                                        "processed",
                                        "category_id_1_deepfashion_train.joblib"))

    model_factory = VAEModelFactory()
    model = model_factory.get_model()
    model.compile(optimizer=optimizers.Adam())

    dataset_factory = VAEDatasetFactory(train_df)
    train_dataset = dataset_factory.get_dataset(batch_size=64, shuffle=True)

    model.fit(train_dataset, epochs=1)
    model.summary()
