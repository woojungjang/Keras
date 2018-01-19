from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
np.random.seed(777)  # for reproducibility

from keras.datasets import mnist

nb_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
inputDim = img_rows * img_cols

# for Using TensorFlow backend.
x_train = x_train.reshape(x_train.shape[0], inputDim)

x_train = x_train.astype('float32') / 255

# 정규화 (gray style)
x_train = x_train.astype('float32') / 255
print(x_train)

# one_hot
y_train = np_utils.to_categorical(y_train, nb_classes)

x_test = x_test.reshape(x_test.shape[0], inputDim)
x_test = x_test.astype('float32') / 255
# one_hot
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
# MNIST data image of shape 28 * 28 = 784
model.add(Dense(nb_classes, input_dim=inputDim))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=15)
score = model.evaluate(x_test, y_test)
print('\nAccuracy:', score[1])

'''
Accuracy: 0.9192
'''
