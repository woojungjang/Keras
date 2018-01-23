from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout


batch_size = 128
num_classes = 10
epochs = 12


# ==============================================================================
# prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ==============================================================================
# build model
# (model code from http://iostream.tistory.com/111)
model = Sequential()


# Glorot uniform initializer, also called Xavier uniform initializer.
# https://keras.io/initializers/#glorot_uniform
# 5 layers
model.add(Dense(256, input_dim=784,
                kernel_initializer='glorot_uniform', activation='relu')) #1
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer='glorot_uniform', activation='relu')) #2
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer='glorot_uniform', activation='relu')) #3
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer='glorot_uniform', activation='relu')) #4
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax')) #5

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# fit메소드(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
# callbacks 매개변수: fitting 이후에 적용할 함수
# validation_split: 훈련에 반영시키지 않을 데이터의 비율(0.0 <= 비율 <= 1.0)
 
# callback에 대한 설명 https://keras.io/models/sequential/
# callback 훈련후 불러들임. cf)java의 awt/ swing/ listener (후속함수)가 콜백함수로 불러들임

# fit 함수
# epoch숫자만큼 훈련 시킨다. fit이 훈련시키는것
# History Object를 반환한다.
# x: training data
# y: label data
# batch_size: 정수 또는 None, 기본값: 32
# epochs: 정수모델을 학습시키기 위한 epoch숫자
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.2) # 60000 중 12000이면 48000 


# ==============================================================================
# predict
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
'''
Test loss: 0.0742975851574
Test accuracy: 0.9811
'''
