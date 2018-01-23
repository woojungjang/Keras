# 파일: krs_logistic_02.py
from keras.models import Sequential 
from keras.layers.core import Dense
import numpy as np
from keras.optimizers import SGD, Adam
from mykeras.mykeras.myfunction import getDataSet

# 엑셀파일로 01 다시풀기
# 참조 파일: myfunction.py파일을 이용하기
# 엑셀파일: logistic02.csv
filename = './logistic02.csv'
data = np.loadtxt(filename, dtype=np.int32, delimiter=',')

x_train, x_test, y_train, y_test = \
    getDataSet(data, testing_row = 2)
    
    
model = Sequential()

# 출력(units) 1개, 입력(input_dim) 2개
# 활성화 함수는 시그모이드
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))


learning_rate = 0.1

# sgd: 옵티마이저 객체
adam = Adam(lr= learning_rate)

# binary_crossentropy는 이항 분류시 사용되는 손실 함수
model.compile(optimizer=adam, loss='binary_crossentropy')

model.fit(x_train, y_train, epochs = 2000)

print(model.predict_classes(x_test))

print('[2, 1]', model.predict_classes(np.array([[2, 1]]))) 
print('[6, 5]',model.predict_classes(np.array([[6, 5]])))


def getCategory(datalist):
    mylist = ['참치', '꽁치']
    for item in range(len(datalist)):
        print( datalist[item], mylist[ (int)(datalist[item]) ] )
        
getCategory(predict)


