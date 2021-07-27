# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
from imageio import imread
from tensorflow.python.keras.engine.input_layer import Input


def read_grayscale_pngs(path, width=20, height=13):
    path = Path(path)
    if not path.exists():
        print("Path {} doesn't exist".format(path))
        return None

    # print(len([name for name in os.listdir('{}/.'.format(path)) if os.path.isfile(name)]))
    num_files = len(list(path.glob('**/*.png'))) # Calculate amount of files in directory
    # num_files = len([f for f in path.iterdir() if path.joinpath(f).is_file()]) # Calculate amount of files in directory

    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(sorted(path.glob('**/*.png'), key=lambda f: int(f.stem))):
        images[i] = np.array(imread(image_path))[:, :, 0] # Pixel data: It's grayscale so take only Red values from [R, G, B, A]
    return images


legal = np.concatenate((read_grayscale_pngs("out/legal/orig"), read_grayscale_pngs("out/legal/mirrored")))
illegal = np.concatenate((read_grayscale_pngs("out/illegal/orig"), read_grayscale_pngs("out/illegal/mirrored")))

legal_test = read_grayscale_pngs("testing_recurrent/legal")
illegal_test = read_grayscale_pngs("testing_recurrent/illegal")


X_train = np.concatenate((legal, illegal))
X_train = X_train / 255.0
Y_train = np.concatenate((np.full(len(legal), 0), np.full(len(illegal), 1)))

X_test = np.concatenate((legal_test, illegal_test))
X_test = X_test / 255.0
Y_test = np.concatenate((np.full(len(legal_test), 0), np.full(len(illegal_test), 1)))




conv_filters = 14
kernel_size = 4
relus = 6
dropout = 0.31

# Reccurent
keras.backend.clear_session()
model = keras.Sequential()

model.add(layers.InputLayer((13,20), name="input"))
model.add(layers.Reshape((13,20,1), input_shape=(13,20)))
model.add(layers.Conv2D(conv_filters, kernel_size, input_shape=(13,20,1), activation="relu"))
model.add(layers.Dropout(dropout))
model.add(layers.Flatten())
model.add(layers.Dense(relus,  activation="relu"))

model.add(layers.Dense(1,  activation="sigmoid", name="output"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
dot_img_file = 'tmp/CNN.pdf'
keras.utils.plot_model(model, to_file=dot_img_file, rankdir="LR")

# model.fit(X_train, Y_train, batch_size=10, epochs=10, verbose=False)


# loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

# f = open("logs/cnn-dropouts", 'a')
# f.write("{} {} {}% {} kernels {} relus dropout {}\n".format(conv_filters, round(loss, 2), round(accuracy*100, 1), kernel_size, relus, dropout))
# f.close()
# print("{} {} {}% {} kernels {} relus {} dropout\n".format(conv_filters, round(loss, 2), round(accuracy*100, 1), kernel_size, relus, dropout))
