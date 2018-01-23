# krs_softmax_03.py
# 엑셀파일(data-04-z00.csv)로처리하기
# 함수 getDataProp()
#         훈련용 데이터와 테스트용 데이터를 나눠 주는 함수
# 함수 printCategory():
# 수치 데이터를 이용하여 종의 이름을 출력해주는 함수
# 종의 분류: 강아지, 고양이, 치타, 코끼리, 사슴, 노루, 돼지

# https://github.com/fchollet/keras/tree/master/examples
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from mykeras.mykeras.myfunction2 import getDataProp
import numpy as np

nb_classes = 7
filename = './data-04-zoo.csv'
data = np.loadtxt(filename, dtype=np.int32, delimiter='.')

# Predicting animal type based on various features

x_train, x_test, y_train, y_test = \
    getDataSet(data, testing_row = 2, one_hot = True, num_classes=nb_classes)

model = Sequential()
model.add(Dense(units = nb_classes, input_shape=(16, ))) #softmax
model.add(Activation('softmax'))
model.summary()


adam = Adam(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=2000)

history = model.fit(x_train, y_train, epochs=1000)


#문자열로 입히기, 함수만듬
def getClassName(mydata):
    namelist = ['강아지', '고양이', '치타','코끼리','사슴','노루','돼지']
    for idx in mydata :
        print( namelist[idx], '(', idx, ')')
            
result = model.predict_classes(x_test)
getClassName(result)



print("1, 6, 6, 6", model.predict_classes(np.array([[1, 6, 6, 6]]))) #predict_classes 분류 함수예측
print("1, 7, 7, 7", model.predict_classes(np.array([[1, 7, 7, 7]]))) 


pred = model.predict_classes(x_train)
for p, y in zip(pred, y_train):
    print("prediction: ", p, " true Y: ", y)
    
    
pred = model.predict_classes(x_test)
for p, y in zip(pred, y_train):
    print("prediction: ", p, " true Y: ", y)
    
    
    
    
    
    
    
    
    
    
    
    
    



x_data = data.split[:, 0:-1]
y_data = data.split[:, [-1]] - 1


def getClassName(mydata):
    namelist = ['강아지', '고양이', '치타','코끼리','사슴','노루','돼지']
    for idx in mydata :
        print( namelist[idx], '(', idx, ')')
            
result = model.predict_classes(x_test)
getClassName(result)



print(x_data.shape, y_data.shape)

nb_classes = 7
y_one_hot = np_utils.to_categorical(y_data, nb_classes)

model = Sequential()
model.add(Dense(nb_classes, input_shape=(16,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_data, y_one_hot, epochs=1000)

# Let's see if we can predict
pred = model.predict_classes(x_data)
for p, y in zip(pred, y_data):
    print("prediction: ", p, " true Y: ", y)
