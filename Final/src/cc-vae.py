import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2
epochs = 100
batch = 25

encoder_inputs = keras.Input(shape=(48, 48, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.build(encoder_inputs)

decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(48 * 48, activation="linear")(x)
decoder_outputs = keras.layers.Reshape((48, 48, 1), input_shape=(2304,))(x)
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.build(decoder_inputs)

predictor_inputs = keras.layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(predictor_inputs)
x = layers.Dense(128, activation="relu")(x)
predictor_outputs = layers.Dense(7, activation="softmax", name="label_probs")(x)
predictor = keras.Model(predictor_inputs, predictor_outputs, name="predictor")
predictor.build(predictor_inputs)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, predictor, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="pred_loss")

        self.cce_loss = keras.losses.CategoricalCrossentropy()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker,
        ]

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        z = self.encoder(inputs)
        x = self.decoder(z[2])
        y = self.predictor(z[0])
        return z, x, y

    def train_step(self, data):
        pixels, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(pixels)

            reconstruction = self.decoder(z)
            classification = self.predictor(z_mean)

            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(keras.losses.MSE(pixels, reconstruction), axis=(1, 2))
            # )
            reconstruction_loss = tf.reduce_sum(
                keras.losses.MSE(pixels, reconstruction), axis=(1, 2)
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)

            classification_loss = self.cce_loss(labels, classification)

            # total_loss = reconstruction_loss + kl_loss + classification_loss
            total_loss = tf.reduce_mean(
                (1 - self.beta) * reconstruction_loss + self.beta * kl_loss + classification_loss
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(classification_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
        }


def main():

    with open("pixels.npy", "rb") as f:
        pixels = np.load(f)

    with open("labels.npy", "rb") as f:
        labels = np.load(f)

    with open("resampled_pixels.npy", "rb") as f:
        resampled_pixels = np.load(f)

    with open("resampled_labels.npy", "rb") as f:
        resampled_labels = np.load(f)

    resampled_pixels = np.expand_dims(resampled_pixels, -1).astype("float32") / 255
    resampled_labels = keras.utils.to_categorical(resampled_labels)

    vae = VAE(encoder, decoder, predictor, beta=0.3, name="vae-model")
    vae_inputs = (None, 48, 48, 1)
    vae.build(vae_inputs)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(resampled_pixels, resampled_labels, epochs=epochs, batch_size=batch)

    # save workaround
    image = pixels[0].astype("float32") / 255
    image = np.reshape(image, (1, 48, 48, 1))
    vae.predict(image)

    vae.save("cc-vae_model")


if __name__ == "__main__":
    main()
