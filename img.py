import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = f"{os.getenv('HOME')}/Source/traindata/train"
CATEGORIES = ["dogs", "cats"]
IMG_SIZE = 50
train_data = []

def readimg():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_array  = cv2.imread(os.path.join(path, img))
            plt.imshow(img_array, cmap="gray")
            plt.show()
            print(img_array)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            plt.imshow(new_array, cmap="gray")
            plt.show()
            break
        break

def create_trainning_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                train_data.append([new_array, class_num])

            except Exception as ex:
                pass

if __name__ == "__main__":
    create_trainning_data()
    print(len(train_data))
    random.shuffle(train_data)
    for sample in train_data:
        print(sample[1])
