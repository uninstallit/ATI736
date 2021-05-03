import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_array = tf.reshape(img_array, shape=(1, 48, 48, 1))

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(preds[0], axis=1)
        # class_channel = preds[0][:, [pred_index]]
        class_channel = tf.Variable([], dtype=tf.float32)
        for index, value in zip(tf.unstack(pred_index), preds[0]):
            class_channel = tf.concat([class_channel, [value[index]]], 0)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def main():

    with open("data_plus/test_pixels.npy", "rb") as f:
        pixels = np.load(f)

    vae = keras.models.load_model("./gc-vae_model_3d")
    print(vae.encoder.summary())

    last_conv_layer_name = "conv2d_5"

    # [2082, 496, 1621, 3404, 3273, 687, 751] 

    # Prepare image
    img = pixels[751]

    # Make model
    model = vae.encoder

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)

    alpha = 0.6

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    stacked_img = np.stack((img,) * 3, axis=-1)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + stacked_img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)
    plt.show()


if __name__ == "__main__":
    main()
