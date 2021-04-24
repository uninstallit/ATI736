import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def main():

  with open("pixels.npy", "rb") as f:
    pixels = np.load(f)

  with open("labels.npy", "rb") as f:
    labels = np.load(f)

  # with open("resampled_pixels.npy", "rb") as f:
  #   pixels = np.load(f)

  # with open("resampled_labels.npy", "rb") as f:
  #   labels = np.load(f)

  x_train, x_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.2, random_state=42)

  (unique, counts) = np.unique(labels, return_counts=True)

  print("unique: ", unique)
  print("counts: ", counts)

  samples = 5000
  emotions = 7
  resampled_pixels = []
  resampled_labels = []

  for emotion in range(emotions):
    indexes = np.where(y_train == emotion)[0]
    indexes = np.random.choice(indexes, samples)

    for pixel, label in list(zip(x_train[indexes], y_train[indexes])):
      resampled_pixels.append(pixel)
      resampled_labels.append(label)

  resampled_pixels, resampled_labels = shuffle(np.asarray(resampled_pixels), np.asarray(resampled_labels), random_state=0)

  (unique, counts) = np.unique(resampled_labels, return_counts=True)

  print("unique: ", unique)
  print("counts: ", counts)

  print(resampled_pixels.shape)
  print(resampled_labels.shape)

  with open('resampled_pixels.npy', 'wb') as f:
    np.save(f, resampled_pixels)

  with open('resampled_labels.npy', 'wb') as f:
    np.save(f, resampled_labels)

if __name__ == "__main__":
    main()