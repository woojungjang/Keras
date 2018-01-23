# Lab 7 Learning rate and Evaluation
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
np.random.seed(777)  # for reproducibility

x_data = np.array([[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
          [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]])
y_data =  np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

# Evaluation our model using this test dataset
x_test =  np.array([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test =  np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

inputDim = x_data.shape[1]
myUnit = y_data.shape[1]

model = Sequential()
model.add(Dense(units = myUnit, input_dim=inputDim)) #input_dim=3 입력개수
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=200)

predictions = model.predict(x_test)

print('_______________________')
print(predictions)
print('_______________________')


score = model.evaluate(x_test, y_test)

print('Prediction: ', [np.argmax(prediction) for prediction in predictions])
print('Cost:', score[0])
print('Accuracy: ', score[1])
