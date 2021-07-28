import numpy as np
from pathlib import Path
from imageio import imread
import pandas as pd

import marginal
from sklearn.metrics import confusion_matrix, accuracy_score

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


legal = read_grayscale_pngs("out/legal/orig")
illegal = read_grayscale_pngs("out/illegal/orig")

legal_test = read_grayscale_pngs("testing/legal")
illegal_test = read_grayscale_pngs("testing/illegal")

features_list = []
for i, dataset in enumerate((legal, illegal, legal_test, illegal_test)):
    features = pd.DataFrame({
        "min": np.min(dataset, axis=(1,2)),
        "max": np.max(dataset, axis=(1,2)),
        "mean": np.mean(dataset, axis=(1,2)),
        "var": np.var(dataset, axis=(1,2)),
        "sum": np.sum(dataset, axis=(1,2)),
        "ptp": np.ptp(dataset, axis=(1,2)),
        "std": np.std(dataset, axis=(1,2)),
        "trace": np.trace(dataset, axis1=1, axis2=2),

        "mmeanx": np.array([marginal.mean(image, dim='x', meanNN_TF=False) for image in dataset]),
        "mmeanxTF": np.array([marginal.mean(image, dim='x', meanNN_TF=True) for image in dataset]),

        "msdx": np.array([marginal.std(image, dim='x', meanNN_TF=False) for image in dataset]),
        "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in dataset]),

        "mmeany": np.array([marginal.mean(image, dim='y', meanNN_TF=False) for image in dataset]),
        "mmeanyTF": np.array([marginal.mean(image, dim='y', meanNN_TF=True) for image in dataset]),

        "msdy": np.array([marginal.std(image, dim='y', meanNN_TF=False) for image in dataset]),
        "msdyTF": np.array([marginal.std(image, dim='y', meanNN_TF=True) for image in dataset]),

        "target": 0 if i % 2 == 0 else 1
    })
    features_list.append(features)

legal_features, illegal_features, legal_test_features, illegal_test_features = tuple(features_list)


features = pd.concat((illegal_features, legal_features))
X_train = features.drop('target', axis=1)
y_train = features['target']

model = SVC()
model.fit(X_train, y_train)

features = pd.concat((illegal_test_features, legal_test_features))
X_test = features.drop('target', axis=1)
y_test = features['target']

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print("accuracy: {}, true positive {}, false positive {}".format(acc, conf[0,0]/(conf[0,0] + conf[0,1]), conf[0,1]/(conf[1,0] + conf[1,1])))
