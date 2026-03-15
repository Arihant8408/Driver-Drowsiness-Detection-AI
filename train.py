import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

data = []
labels = []

dataset_path = "dataset"

classes = os.listdir(dataset_path)

print("Classes:", classes)

for label, folder in enumerate(classes):

    path = os.path.join(dataset_path, folder)

    for img in tqdm(os.listdir(path)):

        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (64, 64))

        data.append(image)
        labels.append(label)

data = np.array(data) / 255.0
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2
)

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)))
model.add(MaxPool2D())

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(len(classes), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

model.save("model.h5")

print("MODEL SAVED")