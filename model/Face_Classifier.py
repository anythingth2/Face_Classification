from keras.applications.resnet50 import ResNet50
from keras import Sequential,Model
from keras.layers.core import Dense,Dropout,Flatten,Activation
import datasets_preprocessing as datasets_preprocessing

model_path = 'model/resnet_model.json'

if __name__ == '__main__':
    print('load datasets')
    datasets, class_labels = datasets_preprocessing.load_datasets(sample=0.01,is_augment=False)

    print('creating model')
    resnet = ResNet50()

    output = resnet.output
    output = Activation('relu')(output)
    output = Dropout(0.25)(output)
    output = Dense(2,activation='softmax')(output)


    model = Model(resnet.input,output)

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    with open(model_path,'w') as f:
        f.write(model.to_json())

    history = model.fit(datasets,class_labels,epochs=20,batch_size=32)

    model.save_weights('weights/resnet_weight.hdf5')
