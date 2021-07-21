import numpy as np
from pathlib import Path
from imageio import imread
import pandas as pd

from sklearn.svm import SVC, LinearSVC
import marginal
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.mixture import BayesianGaussianMixture

from sklearn.ensemble import RandomForestClassifier

def read_grayscale_pngs(path, width=20, height=13):
    path = Path(path)
    if path is None:
        return None

    if not path.exists():
        raise ValueError("Path {} doesn't exist".format(path))

    num_files = len(list(path.glob('**/*.png'))) # Calculate amount of files in directory
    if num_files == 0:
        print("Path {} doesn't contain any images".format(path))
        return None

    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(sorted(path.glob('**/*.png'), key=lambda f: int(f.stem))):
        images[i] = np.array(imread(image_path))[:, :, 0] # Pixel data: It's grayscale so take only Red values from [R, G, B, A]

    return images



legal = np.concatenate((
    read_grayscale_pngs("out/legal/orig"),
    read_grayscale_pngs("testing_recurrent/legal"),
    read_grayscale_pngs("testing/legal") 
    # read_grayscale_pngs("out/legal/mirrored"), 
    # read_grayscale_pngs("out/legal/rotated-20.0"),
    # read_grayscale_pngs("out/legal/rotated20.0"),

    # read_grayscale_pngs("out/legal/rotated-2.0"),
    # read_grayscale_pngs("out/legal/rotated2.0"),

    # read_grayscale_pngs("out/legal/rotated-3.0"),
    # read_grayscale_pngs("out/legal/rotated3.0"),

    # read_grayscale_pngs("out/legal/rotated-3.0"),
    # read_grayscale_pngs("out/legal/rotated3.0"),

    # read_grayscale_pngs("out/legal/rotated-3.0"),
    # read_grayscale_pngs("out/legal/rotated3.0"),

    # read_grayscale_pngs("out/legal/rotated-4.0"),
    # read_grayscale_pngs("out/legal/rotated4.0"),

    # read_grayscale_pngs("out/legal/rotated-5.0"),
    # read_grayscale_pngs("out/legal/rotated5.0"),

    # read_grayscale_pngs("out/legal/rotated-6.0"),
    # read_grayscale_pngs("out/legal/rotated6.0"),

    # read_grayscale_pngs("out/legal/rotated-7.0"),
    # read_grayscale_pngs("out/legal/rotated7.0"),

    # read_grayscale_pngs("out/legal/rotated-10.0"),
    # read_grayscale_pngs("out/legal/rotated10.0"),

    # read_grayscale_pngs("out/legal/rotated-15.0"),
    # read_grayscale_pngs("out/legal/rotated15.0"),

    # read_grayscale_pngs("out/legal/rotated-18.0"),
    # read_grayscale_pngs("out/legal/rotated18.0"),

    # read_grayscale_pngs("out/legal/rotated-22.0"),
    # read_grayscale_pngs("out/legal/rotated22.0"),

    # read_grayscale_pngs("out/legal/rotated-25.0"),
    # read_grayscale_pngs("out/legal/rotated25.0"),

    # read_grayscale_pngs("out/legal/rotated-30.0"),
    # read_grayscale_pngs("out/legal/rotated30.0"),

    # read_grayscale_pngs("out/legal/rotated-40.0"),
    # read_grayscale_pngs("out/legal/rotated40.0"),

))
illegal = np.concatenate((
    read_grayscale_pngs("out/illegal/orig"),
    read_grayscale_pngs("out/illegal/mirrored"),
    read_grayscale_pngs("testing_recurrent/illegal"),
    read_grayscale_pngs("testing/illegal")
    # read_grayscale_pngs("out/illegal/mirrored"), 

    # read_grayscale_pngs("out/illegal/rotated-20.0"),
    # read_grayscale_pngs("out/illegal/rotated20.0"),

    # read_grayscale_pngs("out/illegal/rotated-2.0"),
    # read_grayscale_pngs("out/illegal/rotated2.0"),

    # read_grayscale_pngs("out/illegal/rotated-3.0"),
    # read_grayscale_pngs("out/illegal/rotated3.0"),

    # read_grayscale_pngs("out/illegal/rotated-3.0"),
    # read_grayscale_pngs("out/illegal/rotated3.0"),

    # read_grayscale_pngs("out/illegal/rotated-3.0"),
    # read_grayscale_pngs("out/illegal/rotated3.0"),

    # read_grayscale_pngs("out/illegal/rotated-4.0"),
    # read_grayscale_pngs("out/illegal/rotated4.0"),

    # read_grayscale_pngs("out/illegal/rotated-5.0"),
    # read_grayscale_pngs("out/illegal/rotated5.0"),

    # read_grayscale_pngs("out/illegal/rotated-6.0"),
    # read_grayscale_pngs("out/illegal/rotated6.0"),

    # read_grayscale_pngs("out/illegal/rotate0.0"),
    # read_grayscale_pngs("out/illegal/rotated20
    # read_grayscale_pngs("out/illegal/rotated-10.0"),
    # read_grayscale_pngs("out/illegal/rotated10.0"),

    # read_grayscale_pngs("out/illegal/rotated-15.0"),
    # read_grayscale_pngs("out/illegal/rotated15.0"),

    # read_grayscale_pngs("out/illegal/rotated-18.0"),
    # read_grayscale_pngs("out/illegal/rotated18.0"),

    # read_grayscale_pngs("out/illegal/rotated-22.0"),
    # read_grayscale_pngs("out/illegal/rotated22.0"),

    # read_grayscale_pngs("out/illegal/rotated-25.0"),
    # read_grayscale_pngs("out/illegal/rotated25.0"),

    # read_grayscale_pngs("out/illegal/rotated-30.0"),
    # read_grayscale_pngs("out/illegal/rotated30.0"),

    # read_grayscale_pngs("out/illegal/rotated-40.0"),
    # read_grayscale_pngs("out/illegal/rotated40.0"),

))

legal_test = read_grayscale_pngs("testing_recurrent/legal")
illegal_test = read_grayscale_pngs("testing_recurrent/illegal")

legal_features = pd.DataFrame({
    "std": np.std(legal, axis=(1,2)),
    # "mean": np.mean(legal, axis=(1,2)),
    "mmeanx": np.array([marginal.mean(image, dim='x', meanNN_TF=False) for image in legal]),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in legal]),
    "target": 0
})

illegal_features = pd.DataFrame({
    "std": np.std(illegal, axis=(1,2)),
    # "mean": np.mean(illegal, axis=(1,2)),
    "mmeanx": np.array([marginal.mean(image, dim='x', meanNN_TF=False) for image in illegal]),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in illegal]),
    "target": 1
})

features = pd.concat((illegal_features, legal_features))

X_train = features.drop('target', axis=1)
y_train = features['target']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, "RandomForest.joblib")

legal_test_features = pd.DataFrame({
    "std": np.std(legal_test, axis=(1,2)),
    "mmeanx": np.array([marginal.mean(image, dim='x', meanNN_TF=False) for image in legal_test]),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in legal_test]),
    "target": 0
})
illegal_test_features = pd.DataFrame({
    "std": np.std(illegal_test, axis=(1,2)),
    "mmeanx": np.array([marginal.mean(image, dim='x', meanNN_TF=False) for image in illegal_test]),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in illegal_test]),
    "target": 1
})
features = pd.concat((illegal_test_features, legal_test_features))
X_test = features.drop('target', axis=1)
y_test = features['target']

score = clf.score(X_test, y_test)
print(score)