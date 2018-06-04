import os
from mxnet import io, ndarray
import mxnet
import numpy
from struct import unpack
import time


def load(path):
    start = time.time()
    res = []
    with open(path, 'rb') as f:
        while True:
            byte = f.read(8192)
            if byte:
                i = unpack('b' * len(byte), byte)
                res += i
            else:
                break
    res = res
    end = time.time()
    # print(path)
    return res


def loadpath():
    dirs = ['windows10', 'windows7', 'xp', 'download']
    train = []
    num = 0
    for directory in dirs:
        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                if path.endswith('.exe') and os.path.getsize(path) <= 1024 * 1024:
                    train.append((load(path), 1))
                    num += 1
                    print(num)
    for root, _, files in os.walk('malware'):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith('.exe') and os.path.getsize(path) <= 1024 * 1024:
                train.append((load(path), 0))
                num += 1
                print(num)
    return train


def get_iter(dataset, batch_size):
    n = len(dataset)
    res = []
    for i in range(n // batch_size):
        x = []
        y = []
        for j in range(batch_size):
            data = dataset[i * batch_size + j]
            x.append(data[0])
            y.append(data[1])
        res.append((x, y))
    return res
