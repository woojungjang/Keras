# krs_csvReaderEx03.py
# csv 파일을 읽어 와서 하단의 5행만 테스트용으로 학습시켜보도록 한다.
# 이것을 함수 형태로 작성한다.

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from mykeras.mykeras.myfunction import getDataSet

filename = './score2.csv'
# loadtxt : 데이터를 튜플 형식으로 반환해준다.
data = np.loadtxt(filename, dtype=np.float32, delimiter=',')

x_train, x_test, y_train, y_test = getDataSet(data = data, testing_row = 5)

model = Sequential()


x = len(x_test[1])
print('input_dim:', x)


model = Sequential()
model.add(Dense(input_dim=x, units=1))

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=10000)

for item in x_test :
    result = model.predict( np.array([item]))
    print( result )
    
    
    
    