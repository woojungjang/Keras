from keras.utils import np_utils

def getDataProp(data, testing_rate = 0.2, one_hot = False, num_classes = -1): 
    # data.shape는 tuple 자료형인데, 인덱싱이 가능하다.
    # one_hot: 원핫인코딩 여부 지정, False면 하지 않겠다.
    # num_classes 클래스갯수
    table_row = data.shape[0]
    training_rate = 1.0 - testing_rate
    table_col = data.shape[1]
 
    print(table_row)
    
    # testing_row = 5 # 테스트 용 데이터 셋 개수
    # 훈련용 데이터 셋 개수
    training_row = table_row - testing_row
    
    # table_col : 엑셀 파일의 컬럼 수
    table_col = data.shape[1] # 열의 갯수
    print(table_col)
    
    y_column = 1
    x_column = table_col - y_column # 입력 데이터의 컬럼 갯수
    
    x_train = data[ 0:training_row, 0:x_column ]
    y_train = data[ 0:training_row, x_column:(x_column+1) ]
    
    if one_hot == False:
        pass
    else: # one hot 인코딩이 필요한 경우
        if num_classes >= 1 : 
            y_train = np_utils.to_categorical(y_train, num_classes)
    
    x_test  = data[training_row:,0:x_column ]
    y_test  = data[training_row:, x_column:(x_column+1) ]
    
    return x_train, x_test, y_train, y_test

    ## 여기가 함수의 끝 ####################################
    
    
#     
# (50*4)
# table_col:4 전체 컬럼수 
# 
# x_column:3
# y_column:1
# 
# table_row:50 전체 행수
# 
# training_row: 45
# testing_row:5
# 
# data.shape 
# >> (50,4)
# 
# 
# x_row:3
# 
# table_row: 50
#  = data.shape[0]
# 
# data.shape[1]
#  = table_col (전체컬럼수)
# 
# testing_row = 5
# training_row = table_row - testing_row
# 
# def x_y_r_c(data, testing_row=5) 디폴트가 5라는말
# x_train
# y_test
# 연산
# 반환(x_train, x_test, y_train, y_test)
# a, b, c, d = x_y_r_c(data, 5)


