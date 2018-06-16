from keras.applications.resnet50 import ResNet50
from keras import Sequential
from keras.layers.core import Dense,Dropout
import datasets_preprocessing as datasets_preprocessing


if __name__ == '__main__':
    print('load datasets')
    datasets, class_labels = datasets_preprocessing.load_datasets(sample=0.5)

    print('random augmentation datasets')
    for i in range(len(datasets)):
        datasets[i] = datasets_preprocessing.augment(datasets[i])
        print('augment {} %'.format(i/len(datasets)))

    # print('normalize it!')
    # datasets = datasets_preprocessing.normalize(datasets)

    # print('shuffling')
    # datasets, class_labels = datasets_preprocessing.shuffle_datasets(datasets, class_labels)

    print('creating model')
    resnet = ResNet50()

    model = Sequential()
    model.add(resnet)
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history = model.fit(datasets,class_labels,epochs=25,batch_size=32)

    model.save_weights('weights/resnet_weight.hdf5')
