from mxnet.gluon import nn, utils as gutils
from mxnet import gluon, init, ndarray, autograd
import mxnet
import time
import os
import numpy
import random
import load


def get_ctx():
    ctx = mxnet.gpu()
    try:
        ndarray.array([0], ctx)
    except:
        ctx = mxnet.cpu()
    return ctx


def _get_batch(batch, ctx):
    """return features and labels on ctx"""

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    features, labels = batch
    features = [[data + (1024 * 1024 - len(data)) * [0]] for data in features]

    return (gutils.split_and_load(features, ctx),

            gutils.split_and_load(labels, ctx),

            len(features))
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


def train(train_data, test_data, batch_size, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""

    print("training on", ctx)

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    test_iter = load.get_iter(test_data, batch_size)
    for epoch in range(1, num_epochs + 1):
        random.shuffle(train_data)
        train_iter = load.get_iter(train_data, batch_size)

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
        net.save_params("train_acc %.3f-test_acc %.3f_param" % (

            train_acc_sum / m, test_acc))


net = nn.Sequential()
net.add(
    nn.Conv1D(channels=8, kernel_size=4, padding=0, activation='sigmoid'),
    nn.MaxPool1D(strides=2),
    nn.Conv1D(channels=64, kernel_size=512, strides=512, activation='relu'),
    nn.MaxPool1D(pool_size=4, strides=4),
    nn.Conv1D(channels=128, kernel_size=8, strides=4, activation='relu'),
    nn.MaxPool1D(pool_size=4, strides=4),
    nn.Dropout(0.5),
    nn.Dense(128, activation='relu'),
    nn.Dense(2)
)
ctx = get_ctx()
if (os.path.exists('param')):
    net.load_params('param', ctx=ctx)
else:
    net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
scheduler = mxnet.lr_scheduler.FactorScheduler(100, 0.9)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.008, 'wd': 2e-4, 'lr_scheduler': scheduler, 'momentum': 0.9})
train_data, test_data = load.loadpath()
batch_size = 32
train(train_data, test_data, batch_size, net, loss, trainer, ctx, 20, 10)
