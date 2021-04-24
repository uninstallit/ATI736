import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)

def print_greyscale(pixels_1, pixels_2, width=48, height=48):
    def get_single_greyscale(pixel):
        val = 232 + np.round(pixel * 23)
        return "\x1b[28;5;{}m \x1b[0m".format(int(val))

    for l in range(height):
        line_pixels = np.concatenate(
            (
                pixels_1[l * width : (l + 1) * width],
                pixels_2[l * width : (l + 1) * width],
            ),
            axis=None,
        )
        print("".join(get_single_greyscale(p) for p in line_pixels))


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2

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

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(48 * 48, activation="sigmoid")(x)
decoder_outputs = keras.layers.Reshape((48, 48, 1), input_shape=(2304,))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.build(latent_inputs)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x[2])
        return x

    def train_step(self, data):

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #     )
            # )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction), axis=(1, 2)
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = (1.0 - self.beta) * reconstruction_loss + self.beta * kl_loss

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

    vae = VAE(encoder, decoder, beta=0.3, name="vae-model")
    vae_inputs = (None, 48, 48, 1)
    vae.build(vae_inputs)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(resampled_pixels, epochs=30, batch_size=128)

    # save workaround 
    image = pixels[0].astype("float32") / 255
    image = np.reshape(image, (1, 48, 48, 1))
    vae.predict(image)

    vae.save('vae_model')

if __name__ == "__main__":
    main()
