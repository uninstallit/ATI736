import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():

    with open("data_plus/train_pixels.npy", "rb") as f:
        pixels = np.load(f)

    with open("data_plus/train_labels.npy", "rb") as f:
        labels = np.load(f)

    with open("data_plus/test_pixels.npy", "rb") as f:
        eval_pixels = np.load(f)

    with open("data_plus/test_labels.npy", "rb") as f:
        eval_labels = np.load(f)

    pixels = np.expand_dims(pixels, -1).astype("float32") / 255
    eval_pixels = np.expand_dims(eval_pixels, -1).astype("float32") / 255

    vae = keras.models.load_model("./gc-vae_model_2d")
    z_mean, _, _ = vae.encoder.predict(pixels)
    z_mean_eval, _, _ = vae.encoder.predict(eval_pixels)

    max_acc = 0
    best_k = 0

    for i in range(1, 1000):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(z_mean, labels)
        eval_preds = neigh.predict(z_mean_eval)
        acc = accuracy_score(eval_labels, eval_preds)

        if acc > max_acc:
            max_acc = acc
            best_k = i
            print("best k: {}, eval_acc with knn: {:.3f}".format(best_k, max_acc))

    eval_preds = vae.predictor.predict(z_mean_eval)
    print(
        "eval_acc without knn: {:.3f}".format(
            accuracy_score(eval_labels, np.argmax(eval_preds, axis=1))
        )
    )


if __name__ == "__main__":
    main()
