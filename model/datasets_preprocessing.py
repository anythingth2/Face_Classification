import numpy as np
import cv2
import os
import random
import time
from imgaug import augmenters as iaa
from keras.utils import to_categorical
datasets_path = 'datasets'
datasets_aug_path = 'datasets_aug'
IMG_SHAPE = (224, 224)


def load_datasets(sample=1,is_augment=False):
    class_labels_path = os.listdir(datasets_path)
    print('Import {} class'.format(len(class_labels_path)))
    time.sleep(1)
    datasets = []
    class_labels = []
    for label in class_labels_path:

        image_set_path = os.path.join(datasets_path, label)
        image_set_names = os.listdir(image_set_path)

        for image_name in image_set_names:
            if not (random.random() < sample):
                continue

            image_path = os.path.join(image_set_path, image_name)

            print('loading {} class at {}'.format(label, image_path))
            img = cv2.imread(image_path)
            if is_augment:
                img = augment(img)

            img = normalize(img)
            datasets.append(img)
            class_labels.append(1 if label == 'faces' else 0)

    datasets = np.array(datasets)
    class_labels = np.array(class_labels)
    class_labels = to_categorical(class_labels,2)

    datasets,class_labels = shuffle_datasets(datasets,class_labels)

    return (datasets, class_labels)


def augment(image):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    aug_seq = iaa.Sequential([
        iaa.Noop(),
        sometimes(iaa.Affine(translate_percent={
            'x': (-0.2, 0.2), 'y': (-0.2, 0.2)
        })),
        sometimes(iaa.Affine(rotate=(-45, 45))),
        sometimes(iaa.Add()),
        sometimes(iaa.GaussianBlur(0.5))
    ])


    return aug_seq.augment_image(image)


def normalize(image: np.ndarray):
    return image.astype('float32') / 255


def shuffle_datasets(datasets, labels):
    shuffle = np.array(range(len(labels)))
    np.random.shuffle(shuffle)

    return (datasets[shuffle], labels[shuffle])


