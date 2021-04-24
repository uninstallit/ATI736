import pandas as pd
import numpy as np

meta_df = pd.read_csv('fer2013.csv')
print(meta_df.head())

pix_list = meta_df['pixels'].tolist()
pixels = np.array([np.fromstring(x, dtype=int, sep=' ') for x in pix_list])
pixels = np.reshape(pixels, (-1, 48, 48))

with open('pixels.npy', 'wb') as f:
    np.save(f, pixels)

lab_list = meta_df['emotion'].tolist()
labels = np.asarray(lab_list)

with open('labels.npy', 'wb') as f:
    np.save(f, labels)