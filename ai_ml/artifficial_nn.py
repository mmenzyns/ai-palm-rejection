
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
from imageio import imread


def read_grayscale_pngs(path, width=20, height=13):
    path = Path(path)
    if not path.exists():
        print("Path {} doesn't exist".format(path))
        return None

    num_files = len(list(path.glob('**/*.png'))) # Calculate amount of files in directory
    images = np.empty((num_files, height, width))

    for i, image_path in enumerate(sorted(path.glob('**/*.png'), key=lambda f: int(f.stem))):
        images[i] = np.array(imread(image_path))[:, :, 0] # Pixel data: It's grayscale so take only Red values from [R, G, B, A]
    return images


legal = np.concatenate((
    read_grayscale_pngs("out/legal/orig"), 
    read_grayscale_pngs("out/legal/mirrored"),
    read_grayscale_pngs("out/legal/rotated5.0"),
    read_grayscale_pngs("out/legal/rotated-5.0"),
))
illegal = np.concatenate((
    read_grayscale_pngs("out/illegal/orig"), 
    read_grayscale_pngs("out/illegal/mirrored"),
    read_grayscale_pngs("out/illegal/shifted"),
    read_grayscale_pngs("out/illegal/rotated5.0"),
    read_grayscale_pngs("out/illegal/rotated-5.0")
))

legal_test = read_grayscale_pngs("testing/legal")
illegal_test = read_grayscale_pngs("testing/illegal")

X_train = np.concatenate((legal, illegal))
X_train = X_train / 255.0
Y_train = np.concatenate((np.full(len(legal), 0), np.full(len(illegal), 1)))

X_test = np.concatenate((legal_test, illegal_test))
X_test = X_test / 255.0
Y_test = np.concatenate((np.full(len(legal_test), 0), np.full(len(illegal_test), 1)))


relus = 100

keras.backend.clear_session()
model = keras.Sequential()

model.add(layers.InputLayer((13,20), name="input"))
model.add(layers.Flatten())
model.add(layers.Dense(relus,  activation="relu"))

model.add(layers.Dense(1,  activation="sigmoid", name="output"))

model.compile(loss="binary_crossentropy",  metrics=["binary_accuracy"])

dot_img_file = 'figures/ann.pdf'
keras.utils.plot_model(model, to_file=dot_img_file, rankdir="LR")
model.fit(X_train, Y_train, shuffle=True, batch_size=20, epochs=20,
             verbose=0, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
print(accuracy)