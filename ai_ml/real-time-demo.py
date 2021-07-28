#!/usr/bin/env python

import v4l2
from fcntl import ioctl
import mmap
import os
import sys
import time

import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import array
import numpy as np
from pathlib import Path
from imageio import imread
import pandas as pd

import marginal
import joblib


class StreamCfg:
    def __init__(self, buffer, fd, bufinfo, buf_type, width, height):
        self.buffer = buffer
        self.fd = fd
        self.bufinfo = bufinfo
        self.buf_type = buf_type
        self.width = width
        self.height = height


def start_stream(path="/dev/v4l-touch0"):
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

    buffer = mmap.mmap(fd,
                       bufinfo.length,
                       flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE,
                       offset=bufinfo.m.offset)

    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    if ioctl(fd, v4l2.VIDIOC_G_FMT, fmt) < 0:
        raise BufferError("Couldn't get buffer format")

    print("height: {} width {}".format(fmt.fmt.pix.height, fmt.fmt.pix.width))

    # Create a buffer so it doesn't fail
    buf_type = array.array('I', [bufinfo.type])
    if ioctl(fd, v4l2.VIDIOC_STREAMON, buf_type) < 0:
        raise BufferError("Couldn't enable buffer streaming")

    return StreamCfg(buffer, fd, bufinfo, buf_type, fmt.fmt.pix.width,
                     fmt.fmt.pix.height)


def get_image(cfg):
    # Get frame
    if ioctl(cfg.fd, v4l2.VIDIOC_QBUF, cfg.bufinfo) < 0:
        print("qbuf rip")

    if ioctl(cfg.fd, v4l2.VIDIOC_DQBUF, cfg.bufinfo) < 0:
        print("dqbuf rip")

    dtype_size = 2
    return np.frombuffer(cfg.buffer,
                         dtype=np.int16,
                         count=cfg.bufinfo.length // dtype_size).reshape(
                             cfg.height, cfg.width)


def close_stream(cfg):
    if ioctl(cfg.fd, v4l2.VIDIOC_STREAMOFF, cfg.buf_type) < 0:
        raise BufferError("Couldn't disable buffer streaming")
    os.close(cfg.fd)


def read_grayscale_pngs(path, width=20, height=13):
    path = Path(path)
    if path is None:
        return None

    if not path.exists():
        raise ValueError("Path {} doesn't exist".format(path))

    num_files = len(list(
        path.glob('**/*.png')))  # Calculate amount of files in directory
    if num_files == 0:
        print("Path {} doesn't contain any images".format(path))
        return None

    images = np.empty((num_files, 13, 20))

    for i, image_path in enumerate(
            sorted(path.glob('**/*.png'), key=lambda f: int(f.stem))):
        images[i] = np.array(
            imread(image_path)
        )[:, :,
          0]  # Pixel data: It's grayscale so take only Red values from [R, G, B, A]

    return images


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def update_result(self, image, result):
        self.redraw_figure(image)
        self.label['text'] = "Reject" if result == 1 else "Accept"
        self.label['fg'] = "red" if result == 1 else "green"
        self.update()

    def redraw_figure(self, image):
        sns.heatmap(np.flipud(image),
                    xticklabels=0,
                    yticklabels=0,
                    cbar=False,
                    square=True,
                    vmin=0,
                    vmax=255,
                    ax=self.ax)
        self.canvas.draw()

    def close(self):
        global exit_flag
        exit_flag = True
        root.destroy()

    def create_widgets(self):
        self.label = tk.Label(self, font=("Arial", 30, "bold"))
        self.label.pack(side=tk.TOP)

        fig = plt.figure(figsize=(10, 5))
        self.ax = fig.subplots()
        self.canvas = FigureCanvasTkAgg((fig), master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=True, fill="both")

        self.quit = tk.Button(self,
                              text="Quit",
                              font=("Arial", 20),
                              command=self.close)
        self.quit.pack(side=tk.BOTTOM)


def window_close():
    global exit_flag
    exit_flag = True


model = joblib.load("RandomForest.joblib")
root = tk.Tk()
app = Application(master=root)
root.protocol("WM_DELETE_WINDOW", window_close)
exit_flag = False

cfg = start_stream()
try:
    while True:
        image = get_image(cfg)
        image = np.clip(image, -10, 245) + 10
        features = pd.DataFrame(
            {
                "std": np.std(image),
                "mmeanx": marginal.mean(image, dim='x', meanNN_TF=False),
                "msdxTF": marginal.std(image, dim='x', meanNN_TF=True),
            },
            index=[0])

        result = model.predict(features)[0]
        print(result)
        app.update_result(image, result)

        if exit_flag:
            close_stream(cfg)
            break

        time.sleep(0.1)
except KeyboardInterrupt:
    close_stream(cfg)
    sys.exit(0)