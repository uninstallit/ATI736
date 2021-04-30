import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

print(tf.__version__)

alpha = tf.Variable(1.0, dtype=tf.double)
theta = tf.Variable(0.0, dtype=tf.double)
lam = tf.Variable(1.0, dtype=tf.double)


@tf.function
def gaussian(x, alpha, theta, lam):
    x = tf.cast(x, dtype=tf.double)
    return lam * keras.backend.exp(-alpha * keras.backend.pow(x - theta, 2))


class ValidationCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        eval_pixels = self.model.validation_data[0]
        eval_labels = self.model.validation_data[1]

        z_mean, z_log_var, z = self.model.encoder.predict(eval_pixels)
        eval_pred = self.model.predictor.predict(z_mean)

        print(
            "\n\n *** alpha: {:.3f} - lam: {:.3f} - theta: {:.3f} - eval_acc: {:.3f} *** \n\n".format(
                self.model.alpha.numpy(),
                self.model.lam.numpy(),
                self.model.theta.numpy(),
                accuracy_score(eval_labels, np.argmax(eval_pred, axis=1)),
            )
        )


class VisualizationCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        pixels = self.model.training_data[0]
        labels = self.model.training_data[1]
        z_mean, _, _ = self.model.encoder.predict(pixels)

        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title("Latent Space - Epoch {}".format(epoch))
        plt.savefig("./images/{}.png".format(epoch))


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2
epochs = 100
batch = 128

# encoder_inputs = keras.Input(shape=(48, 48, 1))
# x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(16, activation="relu")(x)

shape=(48, 48, 1)

encoder_inputs = keras.Input(shape=shape)
x = keras.layers.Conv2D(
    32, (3, 3), padding="Same", activation="relu", input_shape=shape
)(encoder_inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(32, (5, 5), padding="Same", activation="relu")(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Conv2D(64, (3, 3), padding="Same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, (5, 5), padding="Same", activation="relu")(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Conv2D(128, (3, 3), padding="Same", activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, (5, 5), padding="Same", activation="relu")(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(48, activation="linear")(x)

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
x = layers.Dense(256, activation=lambda x: gaussian(x, alpha, theta, lam))(
    predictor_inputs
)
predictor_outputs = layers.Dense(7, activation="linear")(x)
predictor = keras.Model(predictor_inputs, predictor_outputs, name="predictor")
predictor.build(predictor_inputs)


class VAE(keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        predictor,
        gamma,
        beta,
        alpha,
        theta,
        lam,
        training_data,
        validation_data,
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.alpha = alpha
        self.lam = lam
        self.training_data = training_data
        self.validation_data = validation_data

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="pred_loss")

        self.cce_loss = keras.losses.CategoricalCrossentropy()
        self.cce_loss_from_logits = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0,
            reduction="auto",
            name="categorical_crossentropy",
        )

    @property
    def metrics(self):
        return [
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

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.alpha)
            tape.watch(self.theta)
            tape.watch(self.lam)

            z_mean, z_log_var, z = self.encoder(pixels, training=True)
            reconstruction = self.decoder(z, training=True)
            prediction = self.predictor(z_mean, training=True)

            reconstruction_loss = tf.reduce_sum(
                keras.losses.MSE(pixels, reconstruction), axis=(1, 2)
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)

            prediction_loss = self.cce_loss_from_logits(labels, prediction)

            total_loss = tf.reduce_mean(
                self.gamma * reconstruction_loss
                + self.beta * kl_loss
                + 5000.0 * prediction_loss
            )

            # total_loss = tf.reduce_mean(prediction_loss)

        grad_alpha = tape.gradient(total_loss, self.alpha)
        self.optimizer.apply_gradients(zip([grad_alpha], [self.alpha]))

        grad_lam = tape.gradient(total_loss, self.lam)
        self.optimizer.apply_gradients(zip([grad_lam], [self.lam]))

        grad_theta = tape.gradient(total_loss, self.theta)
        self.optimizer.apply_gradients(zip([grad_theta], [self.theta]))

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(prediction_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
        }


def main():

    with open("data_plus/train_pixels.npy", "rb") as f:
        pixels = np.load(f)

    with open("data_plus/train_labels.npy", "rb") as f:
        labels = np.load(f)

    with open("data_plus/eval_pixels.npy", "rb") as f:
        eval_pixels = np.load(f)

    with open("data_plus/eval_labels.npy", "rb") as f:
        eval_labels = np.load(f)

    pixels = np.expand_dims(pixels, -1).astype("float32") / 255
    cat_labels = keras.utils.to_categorical(labels)

    eval_pixels = np.expand_dims(eval_pixels, -1).astype("float32") / 255

    vae = VAE(
        encoder,
        decoder,
        predictor,
        gamma=0.001,
        beta=0.3, #0.003,
        theta=theta,
        alpha=alpha,
        lam=lam,
        training_data=(pixels, labels),
        validation_data=(eval_pixels, eval_labels),
        name="vae-model",
    )
    vae_inputs = (None, 48, 48, 1)
    vae.build(vae_inputs)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(
        pixels,
        cat_labels,
        epochs=epochs,
        batch_size=batch,
        callbacks=[ValidationCallback(), VisualizationCallback()],
    )

    # save workaround
    image = pixels[0].astype("float32") / 255
    image = np.reshape(image, (1, 48, 48, 1))
    vae.predict(image)

    vae.save("gc-vae_model")


if __name__ == "__main__":
    main()
