#!/usr/bin/env python

import v4l2
from fcntl import ioctl
import mmap
import os
import array
import numpy as np
from pathlib import Path
from imageio import imread
import pandas as pd
import marginal
import pickle

from sklearn.svm import SVC

class StreamCfg:
    def __init__(self, buffer, fd, bufinfo, buf_type, width, height):
        self.buffer = buffer
        self.fd = fd
        self.bufinfo = bufinfo
        self.buf_type = buf_type
        self.width = width
        self.height = height


def init_stream(path="/dev/v4l-touch0"):
    fd = os.open(path, os.O_RDWR)

    # Get capability
    cap = v4l2.v4l2_capability()
    if ioctl(fd, v4l2.VIDIOC_QUERYCAP, cap) < 0:
        raise BufferError("Couldn't get device capability")

    print("Driver: ", cap.driver.decode('ascii'))
    print("Card: ", cap.card.decode('ascii'))
    print("Bus Info: ", cap.bus_info.decode('ascii'))

    # Cycle until there is no input left which raises OSError
    inp = v4l2.v4l2_input()
    inp.index = 0
    try:
        while ioctl(fd, v4l2.VIDIOC_ENUMINPUT, inp) == 0:
            print("Input: ", inp.index, inp.name.decode('ascii'))
            inp.index += 1
    except OSError:
        pass

    bufrequest = v4l2.v4l2_requestbuffers()
    bufrequest.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    bufrequest.memory = v4l2.V4L2_MEMORY_MMAP
    bufrequest.count = 1

    if ioctl(fd, v4l2.VIDIOC_REQBUFS, bufrequest) < 0:
        raise BufferError("Couldn't request a buffer")

    bufinfo = v4l2.v4l2_buffer()
    bufinfo.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    bufinfo.memory = v4l2.V4L2_MEMORY_MMAP
    bufinfo.index = 0

    if ioctl(fd, v4l2.VIDIOC_QUERYBUF, bufinfo) < 0:
        raise BufferError("Couldn't get buffer info")

    buffer = mmap.mmap(fd, bufinfo.length, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bufinfo.m.offset)

    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    if ioctl(fd, v4l2.VIDIOC_G_FMT, fmt) < 0:
        raise BufferError("Couldn't get buffer format")

    print("height: {} width {}".format(fmt.fmt.pix.height, fmt.fmt.pix.width))

    # Create a buffer so it doesn't fail
    buf_type = array.array('I', [bufinfo.type])
    if ioctl(fd, v4l2.VIDIOC_STREAMON, buf_type) < 0:
        raise BufferError("Couldn't enable buffer streaming")

    return StreamCfg(buffer, fd, bufinfo, buf_type, fmt.fmt.pix.width, fmt.fmt.pix.height)


def get_image(cfg):
    # Get frame
    if ioctl(cfg.fd, v4l2.VIDIOC_QBUF, cfg.bufinfo) < 0:
        print("qbuf rip")

    if ioctl(cfg.fd, v4l2.VIDIOC_DQBUF, cfg.bufinfo) < 0:
        print("dqbuf rip")

    dtype_size = 2
    return np.frombuffer(cfg.buffer, dtype=np.int16, count=cfg.bufinfo.length // dtype_size).reshape(cfg.height, cfg.width)


def close_stream(cfg):
    if ioctl(cfg.fd, v4l2.VIDIOC_STREAMOFF, cfg.buf_type) < 0:
        raise BufferError("Couldn't disable buffer streaming")
    os.close(cfg.fd)


def read_grayscale_pngs(path, width=20, height=13):
    path = Path(path)
    if not path.exists():
        print("Path doesn't exist")
        return None

    # Calculate amount of files in directory
    num_files = len(list(path.glob('**/*.png')))
    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(path.glob('**/*.png')):
        # Pixel data: It's grayscale so take only Red values from [R, G, B, A]
        images[i] = np.array(imread(image_path))[:, :, 0]
    return images


legal = np.concatenate((read_grayscale_pngs("out/legal/orig"), read_grayscale_pngs("out/legal/mirrored"), read_grayscale_pngs("out/legal/shifted"), read_grayscale_pngs("out/legal/shift_mirrored")))
illegal = np.concatenate((read_grayscale_pngs("out/illegal/orig"), read_grayscale_pngs("out/illegal/mirrored"), read_grayscale_pngs("out/illegal/shifted"), read_grayscale_pngs("out/illegal/shift_mirrored")))

legal_test = np.concatenate((read_grayscale_pngs("testing_recurrent/legal"), read_grayscale_pngs("testing/legal")))
illegal_test = np.concatenate((read_grayscale_pngs("testing_recurrent/illegal"), read_grayscale_pngs("testing/illegal")))

X_train = np.empty((len(legal) + len(illegal), 2))
X_test = np.empty((len(legal_test) + len(illegal_test), 2))



legal_features = pd.DataFrame({
    "std": np.std(legal, axis=(1,2)),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in legal]),
    "target": 0
})

illegal_features = pd.DataFrame({
    "std": np.std(illegal, axis=(1,2)),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in illegal]),
    "target": 1
})

features = pd.concat((illegal_features, legal_features))


X_train = features.drop('target', axis=1)
y_train = features['target']

clf = SVC()
clf.fit(X_train, y_train)

pickle.dump(clf, open("SVM_pi.obj", 'wb'))

legal_test_features = pd.DataFrame({
    "std": np.std(legal_test, axis=(1,2)),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in legal_test]),
    "target": 0
})
illegal_test_features = pd.DataFrame({
    "std": np.std(illegal_test, axis=(1,2)),
    "msdxTF": np.array([marginal.std(image, dim='x', meanNN_TF=True) for image in illegal_test]),
    "target": 1
})
features = pd.concat((illegal_features, legal_features))
X_test = features.drop('target', axis=1)
y_test = features['target']

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# cfg = init_stream()
# image = get_image(cfg)

# close_stream(cfg)
