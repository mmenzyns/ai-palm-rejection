import pandas as pd
import os
from glob import glob
import csv
import numpy as np

class TouchpadData:

    width = None
    height = None
    size = None
    min_val = float('inf')
    max_val = float('-inf')

    def __init__(self, path=None):
        if not os.path.exists(path):
            raise Exception

        self.data = []
        # Todo, load recursively
        if os.path.isdir(path):
            for filename in glob(path + '/*'):
                self.data.extend(self.load_file(filename))
        elif os.path.isfile(path):
            self.data = self.load_file(path)
        else:
            raise Exception('unexpected error')

    def load_file(self, filename):
        reader = csv.reader(open(filename, 'r'), delimiter='\t')

        row_num = 0
        data = []
        for row in reader:
            if row_num == 0:
                new_width = int(row[1])
                if self.width is None:
                    self.width = new_width
                else:
                    assert(self.width == new_width)
                new_height = int(row[3])
                if self.height is None:
                    self.height = new_height
                else:
                    assert(self.height == new_height)

            if row_num == 1:
                data.extend(row)

            row_num += 1
        self.size = self.height * self.width
        return pd.to_numeric(data, errors='ignore', downcast='integer').reshape((-1, self.height, self.width))
    