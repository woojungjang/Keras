from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

x_data = np.array([[1, 2, 1, 1],
                   [2, 1, 3, 2],
                   [3, 1, 3, 4],
                   [4, 1, 5, 5],
                   [1, 7, 5, 5],
                   [1, 2, 5, 6],
                   [1, 6, 6, 6],
                   [1, 7, 7, 7]],
                  dtype=np.float32)
y_data = np.array([[0, 0, 1], # y가 2, 2, 2, 1, 1, 1, 0, 0이다
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]],
                  dtype=np.float32)

nb_classes = 3 

model = Sequential()
model.add(Dense(units = nb_classes, input_shape=(4,))) #units=아웃풋
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', #문자넣어도됨
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=1000)
print('history 객체') #history 작업 모든기록이 들어있는 keras내부의 history
print('history.on_epoch_begin(200):', history.on_epoch_begin(200))

print(model.predict_classes(np.array([[1, 2, 1, 1]])))
print(model.predict_classes(np.array([[1, 2, 5, 6]])))
