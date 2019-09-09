import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

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
        print(CATEGORIES[sample[1]])

    X = []
    y = []

    # for features, label in train_data:
    #     x.append(features)
    #     y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open("X.pickle", "wb")
    pickle.dumps(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dumps(y, pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)
    print(X[1])
