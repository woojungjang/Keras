# krs_softmax_04.py

# 엑셀 파일로 처리(iris_data.csv)하기
# 함수 getDataProp()
#       훈련용 데이터와 테스트용 데이터를 비율로 나눠 주는 함수  

# 함수 printCategory() :
# 결과 수치 데이터를 이용하여 종의 이름을 출력해주는 함수
# 종의 분류 : 

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from mykeras.mykeras.myfunction_prop import getDataProp, printCategory

nb_classes = 7

filename = './iris_data.csv'

data = np.loadtxt(filename, dtype=np.int32, delimiter=',')

x_train, x_test, y_train, y_test = \
    getDataProp(data, testing_rate = 0.2, \
            one_hot= True, \
            num_classes = nb_classes )

inputDim = x_train.shape[1]

model = Sequential()
model.add(Dense(units = nb_classes, input_shape=(inputDim,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000)
    
namelist = ['flowerA','flowerB', 'flowerC']
result = model.predict_classes(x_test)
printCategory(namelist, result )


print('--------------')
for prediction, answer in zip(result, x_test) :
    print('예측치:', prediction, ', 정답:', answer)