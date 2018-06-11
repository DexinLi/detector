import os
from mxnet import io, ndarray
import mxnet
import numpy
from struct import unpack
import time


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
    test = []
    for f in testfiles:
        if f.find('malware') != -1:
            test.append((load(f), 0))
        else:
            test.append((load(f), 1))
    train = []
    for f in trainfiles:
        if f.find('malware') != -1:
            train.append((load(f), 0))
        else:
            train.append((load(f), 1))
    return train, test


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


def get_gan_iter(dataset, batch_size):
    n = len(dataset)
    res = []
    for i in range(n // batch_size):
        real = []
        fake = []
        for j in range(batch_size):
            data = dataset[i * batch_size + j]
            if data[1] == 0:
                real.append(data[0])
            else:
                fake.append(data[0])
        res.append((real, fake))
    return res
