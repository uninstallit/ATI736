import pandas as pd
import numpy as np

def process(df, filename):
    pix_list = df['pixels'].tolist()
    pixels = np.array([np.fromstring(x, dtype=int, sep=' ') for x in pix_list])
    pixels = np.reshape(pixels, (-1, 48, 48))

    with open(filename + '_pixels.npy', 'wb') as f:
        np.save(f, pixels)

    lab_list = df['emotion'].tolist()
    labels = np.asarray(lab_list)

    with open(filename + '_labels.npy', 'wb') as f:
        np.save(f, labels)

meta_df = pd.read_csv('fer2013.csv')
print(meta_df.head())

train_df = meta_df[meta_df["Usage"] == "Training"]
eval_df  = meta_df[meta_df["Usage"] == "PublicTest"]
test_df  = meta_df[meta_df["Usage"] == "PrivateTest"]

process(train_df, 'data/train')
process(eval_df, 'data/eval')
process(test_df, 'data/test')
