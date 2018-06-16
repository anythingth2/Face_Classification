import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras import Sequential
from keras.layers.core import Dense,Dropout


resnet = ResNet50()

model = Sequential()
model.add(resnet)
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('weights/resnet_weight.h5')


cam = cv2.VideoCapture(0)
while not cam.isOpened():
    print('camera is not opened')

while cam.isOpened():
    _,frame = cam.read()
    h,w = frame.shape[:2]
    predict_img = frame[:,(w-h)//2:w-(w-h)//2]
    predict_img = cv2.resize(predict_img,(224,224))
    predict_img = predict_img.astype('float32')/255
    predicted = model.predict(np.array([predict_img]))[0]
    print(predicted)

    cv2.imshow('f',frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()