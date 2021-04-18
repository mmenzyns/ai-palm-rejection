import pandas as pd
import os
from glob import glob
import csv

class TouchpadData:

    width = None
    height = None
    size = None
    min_val = float('inf')
    max_val = float('-inf')
    data = None

    def __init__(self, path=None):
        if not os.path.exists(path):
            raise Exception
        
        data = []
        for filename in glob(path + '/*'):
            reader = csv.reader(open(filename, 'r'), delimiter='\t')

            row_num = 0
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

                if row_num == 2:
                    new_min_val = int(row[1])
                    if new_min_val < self.min_val:
                        self.min_val = new_min_val
                    
                    new_max_val = int(row[3])
                    if new_max_val > self.max_val:
                        self.max_val = new_max_val

                row_num += 1
            self.size = self.height * self.width
        self.data = pd.to_numeric(data, errors='ignore', downcast='integer').reshape(-1, self.height, self.width)