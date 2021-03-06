from mxnet.gluon import nn
import mxnet


class GLU(nn.HybridBlock):
    def __init__(self, channels, kernel_size, stride):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv1D(channels, kernel_size, stride)
        self.conv2 = nn.Conv1D(channels, kernel_size, stride)

    def hybrid_forward(self, F, X, *args, **kwargs):
        X = X.transpose((0, 2, 1))
        Y1 = self.conv1(X)
        Y2 = self.conv2(X)
        Z = mxnet.nd.multiply(Y1, mxnet.nd.sigmoid(Y2))
        Z = Z.transpose((0, 2, 1))
        return Z.reshape((Z.shape[0], 1, -1))


class Payload(nn.HybridBlock):
    def __init__(self, ctx):
        super(Payload, self).__init__()
        self.layer = mxnet.nd.random_uniform(0, 255, 1024 * 1024, ctx)

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = x + self.layer
        return y.clip(0, 256)


def get_netD():
    netD = nn.Sequential()
    netD.add(nn.Embedding(256, 8),
             GLU(channels=128, kernel_size=512, stride=512),
             nn.MaxPool1D(128, 128),
             nn.Dense(128),
             nn.Dense(2))
    return netD


def get_netD1():
    netD = nn.Sequential()
    netD.add(nn.Conv1D(channels=8, kernel_size=4, strides=1, activation='relu'),
             nn.MaxPool1D(pool_size=4, strides=1),
             nn.Conv1D(channels=128, kernel_size=512, strides=512, activation='relu'),
             nn.MaxPool1D(pool_size=4, strides=4),
             nn.Conv1D(channels=256, kernel_size=4, strides=4, activation='relu'),
             nn.MaxPool1D(pool_size=4, strides=4),
             nn.Dense(128),
             nn.Dense(2))
    return netD


def get_netG():
    netG = nn.Sequential()
    netG.add(nn.Embedding(256, 8),
             GLU(channels=128, kernel_size=512, stride=512),
             nn.MaxPool1D(128, 128),
             nn.Dense(1024))
    return netG


def get_netG1(ctx):
    netG = nn.Sequential()
    netG.add(
        Payload(ctx)
    )
    return netG
