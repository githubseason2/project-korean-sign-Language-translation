import tensorflow as tf
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
np.set_printoptions(threshold=np.inf)
with open('a.json', 'r') as f:
    data = json.load(f)

coords = []
labels = []
for item in data:
    coords.append(item['coords'])
    labels.append(item['label'][0])

# LabelEncoder를 사용하여 레이블을 정수로 인코딩
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 2)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(40, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(coords, encoded_labels, test_size=0.3)

# 리스트를 넘파이 배열로 변환
X_train = np.array(X_train)
X_test = np.array(X_test)

model.fit(X_train, y_train, epochs=20)

loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

# 원래 문자열 레이블과 해당하는 정수 레이블 출력
print('Original Labels:', labels)
print('Encoded Labels:', encoded_labels)
model.save('my_model.h5')