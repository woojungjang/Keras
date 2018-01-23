# http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html
# npy 학습된결과를 저장해놓은 바이너리파일.

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

# 분류 대상 카테고리 선택하기 --- (※1)
caltech_dir = "./"

# 이미지가 들어 있는 폴더
categories = temp
# categories = ["chair","camera","butterfly","elephant"]
nb_classes = len(categories)

# 이미지 크기 지정 --- (※2)
# RGB24 비트, 64*64 픽셀
image_w = 64
image_h = 64
pixels = image_w * image_h * 3 #3은 RGB

# 이미지 데이터 읽어 들이기 --- (※3)
X = [] # 실제 이미지 데이터
Y = [] # 이미지를 설명하는 레이블 데이터, one hot encoding 된 것
for idx, cat in enumerate(categories): # 카테고리갯수만큼 enumerate
    # 레이블 지정 --- (※4)
    # 요소 갯수가 categories 갯수(예시는 5개이다.)인 리스트 만들기
    label = [0 for i in range(nb_classes)] # i가 0부터 3까지 네번반복 
    #이 줄이 실행되면 label=[0,0,0,0] label[1]=1이면  label=[0,1,0,0] 
    label[idx] = 1 
    # 이미지 --- (※5), cat: chair, camera등등
    image_dir = caltech_dir + "/" + cat # 해당 폴더에서
    # print('image_dir : ', image_dir) # ./image/camera 등등

    # 확장자가 jpg인 파일들
    # glob 모듈은 특정 디렉토리 내의 파일을 목록을 얻어올 수 있다.
    files = glob.glob(image_dir+"/*.jpg") # 확장자가 jpg인 항목만..
    for i, f in enumerate(files):
        img = Image.open(f) # --- (※6)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img) # 배열 데이터로 변환
        X.append(data)
        Y.append(label)
        # if i % 10 == 0:
        #     print(i, "\n", data)

X = np.array(X) # 배열형태로 바꿈
Y = np.array(Y)

# 무작위로 데이터를 학습 전용 데이터와 테스트 전용 데이터 구분한다. --- (※7)
X_train, X_test, y_train, y_test = train_test_split(X, Y) #

xy = (X_train, X_test, y_train, y_test)
print(X_train.shape)
# print(xy)

# np.save : 압축되지 않은 raw 형식의 바이너리 파일을 만들어 준다.
np.save("./5obj.npy", xy) # 데이터를 두번째 매개변수로 넣어주면 .npy파일생성

print("ok,", len(Y))