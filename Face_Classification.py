import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras import Sequential,Model
from keras.models import model_from_json
from keras.layers.core import Dense,Dropout


model_path = 'model/resnet_model.json'


with open(model_path,'r') as f:
    modelJson = f.read()
    model = model_from_json(modelJson)


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('weights/resnet_weight.hdf5')


cam = cv2.VideoCapture(0)
while not cam.isOpened():
    print('camera is not opened')

while cam.isOpened():
    _,frame = cam.read()
    h,w = frame.shape[:2]
    predict_img = frame[:,(w-h)//2:w-(w-h)//2]
    # predict_img = frame
    predict_img = cv2.resize(predict_img,(224,224))
    predict_img = predict_img.astype('float32')/255

    predicted = model.predict(np.array([predict_img]))
    print(predicted)

    cv2.imshow('frame',frame)
    cv2.imshow('predict_img',predict_img)
    cv2.waitKey(1)
cv2.destroyAllWindows()