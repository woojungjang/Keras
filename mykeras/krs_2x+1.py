import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
#a c f p순서

x = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [3.1, 4.1, 4.9, 6.1, 6.9, 8.2, 9.1]

# 모델 객체를 생성한다. Sequential()가 하나의 클래스 작업대상: 가설=모델=그래프 같은말임
model = Sequential()

# Sequential: 선형으로 만든 layer
# add() 메소드를 이용하여 필요한 연산을 추가한다. 
# Dense: NN Layer를 조밀하게 연결시켜 응축해주는 역할 
# Dense는 Core layer에서 검색바람
    # input_dim: 입력의 차원
    # activation: 활성화 함수 지정
    # units: output의 차원수
model.add(Dense(1, input_dim=1))

# 옵티마이져 객체 구하기, GD = GradientDescent
sgd = optimizers.SGD(lr = 0.1)

# 필요한 정보를 입력하고, 컴파일 loss 손실함수
# compile 메소드 매개 변수
# optimize: 옵티마이저를 지정한다.
# loss: 손실함수지정
# metrics: 훈련/테스트 하는 모델이 평가하는 지표들의 목록
# 예시: metircs = {'output_result': 'accuracy'}
# activation: 활성화함수
model.compile(optimizer=sgd, loss='mse') #mse mean square error
# loss의 종류는 keras사이트에서 검색은 losses로

model.summary() #모델 정보 간략히 보기

# epoch숫자만큼 훈련 시킨다. fit이 훈련시키는것
# History Object를 반환한다.
# x: training data
# y: label data
# batch_size: 정수 또는 None, 기본값: 32
# epochs: 정수모델을 학습시키기 위한 epoch숫자
model.fit(x, y, epochs= 200)

# predict()메소드: 입력 데이터에 대한 출력 예측치를 배열 형태로 생성해준다.
# x: 입력 데이터
# batch_size: 기본값 32
# verbose: 0 또는 1
predict = model.predict(x = np.array([5.0])) #prediction w, b는 내부에서 최적화해줌
print(predict)

# model 바깥으로 정확도나 손실함수의 값을 가져올 수 있는 방법은?
# 

print(model.get_weights()) #array앞에것이 w, 뒤에것이 b
#print(model.set_weights()) 값 넣어주는 함수 #https://keras.io/models/about-keras-models/

# model 관련 메소드 정리
# get_weights(): w와 b의 값을 반환해준다.
# to_json(): 제이슨 형식(파일)으로 반환해준다.
# summary(): 간략한 형식으로 결과보여주기
# summary()가 keras.utils.print_summary(model)와 동일한지 확인필요

# 모델의 구성 설정 정보들을 사전형식으로 반환해준다.
print(model.get_config())

[{'class_name': 'Dense',
  'config': {'W_constraint': None,
   'W_regularizer': None,
   'activation': 'linear',
   'activity_regularizer': None,
   'b_constraint': None,
   'b_regularizer': None,
   'batch_input_shape': (None, 500),
   'init': 'glorot_uniform',
   'input_dim': None,
   'input_dtype': 'float32',
   'name': 'dense_1',
   'output_dim': 32,
   'trainable': True}},
 {'class_name': 'Dense',
  'config': {'W_constraint': None,
   'W_regularizer': None,
   'activation': 'softmax',
   'activity_regularizer': None,
   'b_constraint': None,
   'b_regularizer': None,
   'init': 'glorot_uniform',
   'input_dim': None,
   'name': 'dense_2',
   'output_dim': 10,
   'trainable': False}}]
