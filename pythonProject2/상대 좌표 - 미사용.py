import json
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# json 파일에서 좌표값 읽어오기
with open('''./a.json''', 'r') as f:
    data = json.load(f)
X = np.array(data['coords'])

# 손가락 좌표값 학습을 위한 라벨 생성 (0~4 사이의 정수)
#print(data['lable'])
y = np.array(data['lable'])
#print(X)
#X를 [거리, 좌표]로 변환하기
b=[]
for i in range(len(X)):
    b.append([])
    for j in range(0,21):
        b[i].append( [np.linalg.norm(X[i][j]-X[i][0])/450,math.atan2(X[i][j][1]-X[i][0][1], X[i][j][0]-X[i][0][0])])
#print(b)
b=np.array(b)
# 모델 구성
model = Sequential([
    Flatten(input_shape=(21, 2)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(11, activation='softmax')
])

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(b, y, epochs=5, batch_size=1)

# 모델 저장
model.save('my_model.h5')

...
