import numpy as np

from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
np.savetxt('encoder/image.out', x_test.reshape(10000, 784))
np.save('encoder/image', x_test.reshape(10000, 784))
print(1)