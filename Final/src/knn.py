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

    with open("data_plus/eval_pixels.npy", "rb") as f:
        eval_pixels = np.load(f)

    with open("data_plus/eval_labels.npy", "rb") as f:
        eval_labels = np.load(f)

    pixels = np.expand_dims(pixels, -1).astype("float32") / 255
    # cat_labels = keras.utils.to_categorical(labels)

    # eval_pixels = np.expand_dims(eval_pixels, -1).astype("float32") / 255

    eval_pixels = np.expand_dims(eval_pixels, -1).astype("float32") / 255
    # eval_labels = keras.utils.to_categorical(eval_labels)S

    vae = keras.models.load_model("./gc-vae_model")
    z_mean, _, _ = vae.encoder.predict(pixels)
    z_mean_eval, _, _ = vae.encoder.predict(eval_pixels)
    # y_pred = vae.predictor.predict(z_mean)
    
    print(vae.encoder.summary())

    for i in range(1, 4000):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(z_mean, labels)
        eval_pred = neigh.predict(z_mean_eval)
        print("eval_acc: {:.3f}".format(accuracy_score(eval_labels, eval_pred)))

if __name__ == "__main__":
    main()
