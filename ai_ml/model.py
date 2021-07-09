import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
# from tqdm.notebook import tqdm
from imageio import imread


def read_grayscale_pngs(path, width=20, height=13):
    path = Path(path)
    if not path.exists():
        print("Path doesn't exist")
        return None

    # print(len([name for name in os.listdir('{}/.'.format(path)) if os.path.isfile(name)]))
    num_files = len(list(path.glob('**/*.png'))) # Calculate amount of files in directory
    # num_files = len([f for f in path.iterdir() if path.joinpath(f).is_file()]) # Calculate amount of files in directory

    ids = np.empty(num_files)
    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(path.glob('**/*.png')):
        images[i] = np.array(imread(image_path))[:, :, 0] # Pixel data: It's grayscale so take only Red values from [R, G, B, A]
    return images


legal = read_grayscale_pngs("../data_processing/out/legal")
illegal = read_grayscale_pngs("../data_processing/out/illegal")

legal_test = read_grayscale_pngs("testing/legal")
illegal_test = read_grayscale_pngs("testing/illegal")


X_train = np.concatenate((legal, illegal))
Y_train = np.concatenate((np.full(len(legal), 0), np.full(len(illegal), 1)))

X_test = np.concatenate((legal_test, illegal_test))
Y_test = np.concatenate((np.full(len(legal_test), 0), np.full(len(illegal_test), 1)))


# model = keras.Sequential()
# model.add(layers.Flatten(input_shape=(13,20)))
# model.add(layers.Dense(1, activation="relu"))
# # model.add(layers.Dense(2, activation="relu"))
# model.add(layers.Dense(1, activation="sigmoid"))

# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# model.fit(X_train, Y_train, shuffle=True, epochs=10)

# loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
# print("loss: {}, accuracy: {}%".format(round(loss, 2), round(accuracy*100, 2)))



layer_sizes = []
accuracies = []

i = 1
while i <= 256:
    print(i)
    
    
    keras.backend.clear_session()
    model = keras.Sequential()
    # model.add(layers.Flatten(input_shape=(13,20)))
    model.add(layers.Dense(i, activation="relu"))
    # model.add(layers.Dense(100, activation="relu"))
    # model.add(layers.Dense(20, activation="relu"))

    model.add(layers.Dense(1,  activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, Y_train, shuffle=True, batch_size=320, epochs=10)

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    
    layer_sizes.append(i)
    accuracies.append(accuracy)
    # print("loss: {}, accuracy: {}%".format(round(loss, 2), round(accuracy*100, 2)))

    i = 2*i

print(layer_sizes)
print(accuracies)


keras.backend.clear_session()
model = keras.Sequential()
# model.add(layers.Flatten(input_shape=(13,20)))
model.add(layers.Dense(1, activation="relu"))
# model.add(layers.Dense(100, activation="relu"))
# model.add(layers.Dense(20, activation="relu"))

model.add(layers.Dense(1,  activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, Y_train, shuffle=True, batch_size=320, epochs=10)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(accuracy)