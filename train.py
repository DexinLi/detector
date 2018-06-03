from mxnet.gluon import nn, utils as gutils
from mxnet import gluon, init, ndarray, autograd
import mxnet
import time
import os
import numpy
import load

ctx = mxnet.gpu()


def _get_batch(batch, ctx):
    """return features and labels on ctx"""

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    features, labels = batch
    features = [[load.load(path)] for path in features]
    features = ndarray.array(features)
    labels = ndarray.array(labels)

    return (gutils.split_and_load(features, ctx),

            gutils.split_and_load(labels, ctx),

            features.shape[0])
    # return (features,labels,features.shape[0])


def evaluate_accuracy(data_iter, net, ctx=[mxnet.cpu()]):
    """Evaluate accuracy of a model on the given data set."""

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]

    acc = ndarray.array([0])

    n = 0

    for batch in data_iter:

        features, labels, batch_size = _get_batch(batch, ctx)

        for X, y in zip(features, labels):
            y = y.astype('float32')

            acc += (net(X).argmax(axis=1) == y).sum().copyto(mxnet.cpu())

            n += y.size

        acc.wait_to_read()

    return acc.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""

    print("training on", ctx)

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]

    for epoch in range(1, num_epochs + 1):

        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0

        start = time.time()

        i = 0
        for batch in train_iter:
            i += 1

            Xs, ys, batch_size = _get_batch(batch, ctx)

            ls = []

            with autograd.record():

                y_hats = [net(X) for X in Xs]

                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]

            for l in ls:
                l.backward()

            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()

                                  for y_hat, y in zip(y_hats, ys)])

            train_l_sum += sum([l.sum().asscalar() for l in ls])

            trainer.step(batch_size)

            n += batch_size

            m += sum([y.size for y in ys])

            if print_batches and (i + 1) % print_batches == 0:
                print("batch %d, loss %f, train acc %f" % (

                    n, train_l_sum / n, train_acc_sum / m

                ))

        test_acc = evaluate_accuracy(test_iter, net, ctx)

        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" % (

            epoch, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start

        ))
        net.save_params('param')


net = nn.Sequential()
net.add(
    nn.Conv1D(channels=8, kernel_size=4, padding=0, activation='sigmoid'),
    nn.MaxPool1D(strides=2),
    nn.Conv1D(channels=64, kernel_size=512, strides=512, activation='relu'),
    nn.MaxPool1D(pool_size=4, strides=4),
    nn.Conv1D(channels=128, kernel_size=8, strides=4, activation='relu'),
    nn.MaxPool1D(pool_size=4, strides=4),
    nn.Dense(128, activation='relu'),
)

if(os.path.exists('param')):
    net.load_params('param',ctx=ctx)
else:
    net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.008})
import random

data = load.loadpath()
random.shuffle(data)
batch_size = 8
train_data = load.get_iter(data[100:], batch_size=batch_size, ctx=ctx)
test_data = load.get_iter(data[:100], batch_size=batch_size, ctx=ctx)
train(train_data, test_data, net, loss, trainer, ctx, 10, 10)
