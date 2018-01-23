#http://blog.daum.net/lonemoon/87 머신러닝 라이브러리설명
# 1. tensorflow 
# 래퍼 라이브러리
# 라이브러리 랩으로 쌓듯 내부로 집어 넣고 최소코딩만 적용함
# 
# 3. 캐라스 
# 단순화된 인터패이스
# 순서 acfp add compile fit predict
#preference > install keras
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential


x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#모델객체를 생성한다.
model = Sequential()
model.add(Dense(1, input_dim=1)) 

sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd) #자바에서 하듯 컴파일

# prints summary of the model to the terminal
model.summary()

model.fit(x_train, y_train, epochs=200) #fit가 training method

y_predict = model.predict(np.array([5]))

print(y_predict)