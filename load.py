import os
from mxnet import io, ndarray
import mxnet
import numpy
from struct import unpack
import time
import random


def load(path):
    res = []
    with open(path, 'rb') as f:
        while True:
            byte = f.read(8192)
            if byte:
                i = unpack('b' * len(byte), byte)
                res += i
            else:
                break
    return res


def loadpath():
    with open('test.txt', 'r') as f:
        testfiles = f.read().splitlines()
    with open('train.txt', 'r') as f:
        trainfiles = f.read().splitlines()
    test = [[], []]
    for f in testfiles:
        if f.find('malware') != -1:
            test[0].append(load(f))
        else:
            test[1].append(load(f))
    train = [[], []]
    for f in trainfiles:
        if f.find('malware') != -1:
            train[0].append(load(f))
        else:
            train[1].append(load(f))
    return train, test


def get_iter(dataset, batch_size):
    benign, malware = dataset
    n = (len(benign) + len(malware)) // batch_size
    random.shuffle(benign)
    random.shuffle(malware)
    size1 = benign // n
    size2 = malware // n
    res = []
    for i in range(n):
        x = []
        y = []
        for j in range(size1):
            data = benign[i * size1 + j]
            x.append(data)
            y.append(0)
        for j in range(size2):
            data = malware[i * size2 + j]
            x.append(data)
            y.append(1)
        res.append((x, y))
    return res
