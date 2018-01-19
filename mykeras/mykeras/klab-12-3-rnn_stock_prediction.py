# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import numpy as np
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# brew install graphviz
# pip3 install graphviz
# pip3 install pydot-ng
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

timesteps = seq_length = 7
data_dim = 5

# Open,High,Low,Close,Volume
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
# 가장 최근 데이터가 위로 오겠끔
xy = xy[::-1]  # reverse order (chronically ordered)

# 정규화작업
# very important. It does not work without it.
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)

x = xy # 엑셀파일의 모든열
print(x)
 
y = xy[:, [-1]]  # 마지막열

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing (70:30)으로 훈련이랑 테스트용 나눔 (dataX[0:70])
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
# 출력 데이터 셋
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# 입력 데이터 셋
model = Sequential()
model.add(LSTM(1, input_shape=(seq_length, data_dim), return_sequences=False))
# model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

# Store model graph in png
# (Error occurs on in python interactive shell)
# plot_model(model, to_file=os.path.basename(__file__) + '.png', show_shapes=True)

print(trainX.shape, trainY.shape)
model.fit(trainX, trainY, epochs=200)

# make predictions
testPredict = model.predict(testX)

# inverse values
# testPredict = scaler.transform(testPredict)
# testY = scaler.transform(testY)

# print(testPredict)
plt.plot(testY) # 정답그래프
plt.plot(testPredict) # 예측한 그래프
plt.show()
