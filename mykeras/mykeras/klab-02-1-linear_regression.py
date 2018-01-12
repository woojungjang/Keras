#http://blog.daum.net/lonemoon/87 머신러닝 라이브러리설명
# 1. tensorflow 
# 래퍼 라이브러리
# 라이브러리 랩으로 쌓듯 내부로 집어 넣음
# 
# 3. 캐라스 
# 단순화된 인터패이스

#preference > install keras
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential


x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

model = Sequential()
model.add(Dense(1, input_dim=1))

sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)

# prints summary of the model to the terminal
model.summary()

model.fit(x_train, y_train, epochs=200)

y_predict = model.predict(np.array([5]))

print(y_predict)