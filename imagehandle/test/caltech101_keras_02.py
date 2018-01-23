import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.pooling import MaxPooling2D

import os

# 카테고리 지정하기
categories = []
root_dir = '../test'

result = os.listdir(root_dir)
for item in result:
    if os.path.isdir(item):
        categories.append(item)

# 이미지가 들어 있는 폴더
# categories = ["chair","camera","butterfly","elephant"]
nb_classes = len(categories)


# 이미지 크기 지정하기
image_w = 64
image_h = 64

# 이미지 데이터를 불러오기 --- (※1)
X_train, X_test, y_train, y_test = np.load('newobj.npy')

# 읽어온 데이터 정규화하기
X_train = X_train.astype('float') / 256
X_test = X_test.astype('float') / 256

print('X_train shape:', X_train.shape)

# CNN 모델 구축하기
model = Sequential()
model.add(Convolution2D(32, 3, 3,border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # --- (※3) 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes)) # 4개의 클래스

model.add(Activation('softmax')) # 다분류라 소프트맥스 사용함

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# 모델 훈련하기
model.fit(X_train, y_train, batch_size=32, nb_epoch=50)

# 모델 평가하기
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])