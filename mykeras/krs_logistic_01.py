# 파일: krs_logistic_01.py
from keras.models import Sequential 
from keras.layers.core import Dense
import numpy as np
from keras.optimizers import SGD

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] #6행 3열
y_data = [[0],[0],[0],[1],[1],[1]]

model = Sequential()

# 출력(units) 1개, 입력(input_dim) 2개
# 활성화 함수는 시그모이드
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))


learning_rate = 0.1

# sgd: 옵티마이저 객체
sgd = SGD(lr= learning_rate)

# binary_crossentropy는 이항 분류시 사용되는 손실 함수
model.compile(optimizer=sgd, loss='binary_crossentropy')

model.fit(x_data, y_data, epochs = 2000)

print('[2, 1]', model.predict_classes(np.array([[2, 1]])))
print('[6, 5]',model.predict_classes(np.array([[6, 5]])))

