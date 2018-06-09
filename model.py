from mxnet.gluon import nn
import mxnet


class GLU(nn.Block):
    def __init__(self, channels, kernel_size, stride):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv1D(channels, kernel_size, stride)
        self.conv2 = nn.Conv1D(channels, kernel_size, stride)

    def forward(self, X):
        X = X.transpose((0, 2, 1))
        Y1 = self.conv1(X)
        Y2 = self.conv2(X)
        Z = mxnet.nd.multiply(Y1, mxnet.nd.sigmoid(Y2))
        return Z.reshape((Z.shape[0], 1, -1))


def get_netD():
    netD = nn.Sequential()
    netD.add(nn.Embedding(256, 8),
             GLU(channels=128, kernel_size=512, stride=512),
             nn.MaxPool1D(128, 128),
             nn.Dense(128),
             nn.Dense(2))
    return netD
