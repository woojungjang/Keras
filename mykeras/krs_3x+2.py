import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

x = [1.0, 1.5, 2.0, 2.5, 3.0]
y = [4.9, 6.6, 8.0, 9.5, 11.0]

model = Sequential()
model.add(Dense(1, input_dim=1))

sgd = optimizers.SGD(lr = 0.1)

model.compile(optimizer=sgd, loss='mse')

model.summary()

model.fit(x, y, epochs= 200)

predict = model.predict(x = np.array([5.0]))
print(predict)

print(model.get_weights())


