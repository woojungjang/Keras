# krs_softmax_02.py

# klab-06-1-softmax.py 파일을 변형하여 엑셀파일로 처리해보세요.
# 엑셀파일: softmax02.csv

# np_utils.to_categorical( y데이터, nb_classes)

from keras.models import Sequential 
from keras.layers.core import Dense, Activation
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from mykeras.mykeras.myfunction import getDataSet
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer


nb_classes = 3
filename = './softmax02.csv'
data = np.loadtxt(filename, dtype=np.int32, delimiter=',')



x_train, x_test, y_train, y_test = \
    getDataSet(data, testing_row = 2, one_hot = True, num_classes=nb_classes)


#y = np.array([[2],[0],[1]], dtype=np.float32)

model = Sequential()
model.add(Dense(units = nb_classes, input_shape=(4, ))) #softmax
model.add(Activation('softmax'))
model.summary()


adam = Adam(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=2000)

history = model.fit(x_train, y_train, epochs=1000)
#문자열로 입히기, 함수만듬
def getClassName(mydata):
    namelist = ['강아지', '고양이', '토끼']
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
    
    
    
