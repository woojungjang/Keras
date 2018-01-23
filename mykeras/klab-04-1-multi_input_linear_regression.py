from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, adam
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
#shape(5,3)

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

model = Sequential()
model.add(Dense(input_dim=3, units=1))
# 입력의 갯수 input_dim, units은 출력의 shape

model.add(Activation('linear'))
#활성화함수 선형차원
rmsprop = adam(lr=0.01)
#RMSprop adagrad의 단점을 보완한 옵티마이져

model.compile(loss='mse', optimizer=rmsprop)
model.fit(x_data, y_data, epochs=10000) #fit 트레이닝

y_predict = model.predict(np.array([[95., 100., 80]]))
print(y_predict)
