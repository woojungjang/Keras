from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def xxx(data, testing_row = 5):
    return x_train, x_test, y_train, y_test

xy = np.loadtxt('data-01-test-score.csv', delimiter=',')

x_train, x_test, y_train, y_test = xxx(data, 5)


x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]



print("x_data", x_data)
print("y_data", y_data)

x = len(x_data[1])
print('input_dim:', x)


model = Sequential()
model.add(Dense(input_dim=x, units=1))
#input_dim=3

model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_data, y_data, epochs=2000)

print("0, 2, 1", model.predict(np.array([[0, 2, 1]])))
print("0, 9, -1", model.predict(np.array([[0, 9, -1]])))
