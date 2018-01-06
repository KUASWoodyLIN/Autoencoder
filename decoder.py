import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import load_model

dnn_32_txt = 'encoder/dnn_32.out'
dnn_8_txt = 'encoder/dnn_8.out'
dnn_2_txt = 'encoder/dnn_2.out'
dnn_32_h5 = 'h5/dnn_32.h5'
dnn_8_h5 = 'h5/dnn_8.h5'
dnn_2_h5 = 'h5/dnn_2.h5'

(x_train, _), (x_test, _) = mnist.load_data()
encoded_32_imgs = np.loadtxt(dnn_32_txt)
encoded_8_imgs = np.loadtxt(dnn_8_txt)
encoded_2_imgs = np.loadtxt(dnn_2_txt)

decoder_32 = load_model(dnn_32_h5)
decoder_8 = load_model(dnn_8_h5)
decoder_2 = load_model(dnn_2_h5)

decoded_32_imgs = decoder_32.predict(encoded_32_imgs)
decoded_8_imgs = decoder_8.predict(encoded_8_imgs)
decoded_2_imgs = decoder_2.predict(encoded_2_imgs)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(0, decoded_32_imgs.shape[0], 10):
    for j in range(n):
        # display original
        ax = plt.subplot(4, n, j + 1)
        plt.imshow(x_test[i+j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, j + 1 + n)
        plt.imshow(decoded_32_imgs[i+j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, j + 1 + 2*n)
        plt.imshow(decoded_8_imgs[i+j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, j + 1 + 3*n)
        plt.imshow(decoded_2_imgs[i+j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

