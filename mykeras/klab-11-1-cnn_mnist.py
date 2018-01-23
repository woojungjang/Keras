# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
# add, compile, fit, predict, evaluate 함수의 순으로 
# 각 함수별 매개변수 숙지 필요
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

# conv2d: convolution layer에 검색
# nb_filters : 필수, 필터의 수
# kernel_size : 필수, 커널의 크기
# padding='valid': 패딩여부('valid'(기본값) or 'same') 
# input_shape=input_shape : 입력 형상
# pool_size: 풀의 크기
# filters: 출력의 차원
# strides: 2개의 리스트/튜풀로 너비와 높이를 준다. 미명시시 pool_size 파라미터 값을 사용한다.
# padding:

# MaxPooling2D
model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size))
model.add(Activation('relu'))

# MaxPooling2D: 맥스풀링 연산을 수행하는 함수
# MaxPooling2D: Dim이 점점커지면 복잡한 수식 간결하게
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))

# Flatten: 수식을 간결하게, 차원축소
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# validation_data: 각 epoch의 끝에서 비용함수와 정확도 등을 평가해보기 위한 데이터
# verbose: (0):silent모드, (1)진행상황보여줌, 디폴트임 (2)하나의 epoch마다 출력
# shuffle: boolean, epoch 단위로 훈련용 데이터를 섞을 것인가의 여부
# fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, 
# validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)


model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=0, validation_data=(X_test, Y_test), shuffle=True)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
