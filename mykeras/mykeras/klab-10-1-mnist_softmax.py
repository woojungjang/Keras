from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import initializers
from keras.utils import np_utils

# 검색: keras backend(텐서플로우나 티아노, 별칭 K)를 쓸수있는 방법이다.
from keras import backend as K

from keras.datasets import mnist

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()


# Dense 검색: Core layers
# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# use_bias=True 사용하겠다. 바이어스 디폴트값이 True
# kernel_initializer, bias_initializer 많이 씀
# bias_initializer = 'zeros'이므로 기본 값은 0이다. 
# kernel_initializer 가중치 w에 대한 초기자라고 보면됨
# keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# kernel_constraint: w b 값 제한조건 넣기 위한 함수 예)W B값으로 음수는 안된다
# Regularizer: 오버피팅이나 아웃라이어에 대해 패널티를 주어 표준화시키는것

model.add(Dense(units=nb_classes, input_dim=img_rows * img_cols,
                kernel_initializer=initializers.random_normal(stddev=0.01), #평균 0.0 표준편차 0.01를 가지는 랜덤값
                use_bias=True))
model.add(Activation('softmax'))

model.summary()

adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=training_epochs)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
Test score: 0.265781824633
Test accuracy: 0.9268
'''
