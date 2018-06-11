from mxnet.gluon import nn, utils as gutils
from mxnet import gluon, init, ndarray, autograd
import mxnet
import time
import os
import model
import numpy
import load


def get_ctx():
    ctx = mxnet.gpu(1)
    try:
        ndarray.array([0], ctx)
    except:
        ctx = mxnet.cpu()
    return ctx


def add_noise(x):
    noise = []
    for i in range(1024):
        noise.append(random.uniform(0, 255))
    return x + noise + (1024 * 1023 - len(x)) * [0]


def get_real_in(real, ctx):
    real = [data + [0] * (1024 * 1024 - len(data)) for data in real]
    return ndarray.array(real, ctx=ctx)


def _get_fake_in_D(fake, netG, ctx):
    n = len(fake)
    fake_in = [add_noise(x) for x in fake]
    padding = [[0] * (1024 * 1023 - len(x)) for x in fake]
    fake_out = netG(ndarray.array(fake_in, ctx=ctx)).detach()
    res = ndarray.zeros((n, 1024 * 1024), ctx=ctx)
    for x, y, z, i in zip(fake, fake_out, padding, range(n)):
        pre = ndarray.zeros(len(x), ctx=ctx)
        suf = ndarray.array(z, ctx=ctx)
        data = ndarray.concat(pre, y, suf, dim=0)
        res[i] = data
    return res


def _get_fake_in_G(fake, netG, ctx):
    n = len(fake)
    fake_in = [add_noise(x) for x in fake]
    padding = [[0] * (1024 * 1023 - len(x)) for x in fake]
    fake_out = netG(ndarray.array(fake_in, ctx=ctx))
    res = ndarray.zeros((n, 1024 * 1024), ctx=ctx)
    for x, y, z, i in zip(fake, fake_out, padding, range(n)):
        pre = ndarray.zeros(len(x), ctx=ctx)
        suf = ndarray.array(z, ctx=ctx)
        data = ndarray.concat(pre, y, suf, dim=0)
        res[i] = data
    return res


def evaluate_accuracy(data_iter, netD, netG, ctx):
    acc = ndarray.array([0])

    for batch in data_iter:
        real, fake = batch
        real_in = get_real_in(real, ctx=ctx)
        real_label = ndarray.zeros(len(real), ctx=ctx)
        fake_in = _get_fake_in_D(fake, netG, ctx)
        fake_label = ndarray.ones(len(fake), ctx=ctx)
        acc += (netD(real_in).argmax(axis=1) == real_label).sum().copyto(mxnet.cpu())
        acc += (netD(fake_in).argmax(axis=1) == fake_label).sum().copyto(mxnet.cpu())

        acc.wait_to_read()

    return acc.asscalar() / 1000


def train(train_data, test_data, batch_size, netD, netG, loss, trainerD, trainerG, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""

    print("training on", ctx)

    for epoch in range(1, num_epochs + 1):
        random.shuffle(train_data)
        train_iter = load.get_gan_iter(train_data, batch_size=batch_size)
        test_iter = load.get_gan_iter(test_data, 25)

        dis_l_sum, disc_acc_sum, n, dm = 0.0, 0.0, 0.0, 0.0
        gen_l_sum, gen_acc_sum, gm = 0.0, 0.0, 0.0

        start = time.time()

        i = 0
        for batch in train_iter:
            i += 1

            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################

            real, fake = batch
            real_in = get_real_in(real, ctx)
            real_label = ndarray.zeros(len(real), ctx=ctx)
            fake_in = _get_fake_in_D(fake, netG, ctx)
            fake_label = ndarray.ones(len(fake), ctx=ctx)

            with autograd.record():

                y_hats = netD(real_in)
                disc_acc_sum += (y_hats.argmax(axis=1) == real_label).sum().asscalar()
                ls = loss(y_hats, real_label)

                y_hats = netD(fake_in)
                disc_acc_sum += (y_hats.argmax(axis=1) == fake_label).sum().asscalar()
                ls += loss(y_hats, fake_label)
                dis_l_sum += ls.sum().asscalar()
                ls *= 0.5
                ls.backward()

            trainerD.step(len(fake) + len(real))

            n += len(real) + len(fake)

            dm += len(real) + len(fake)

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z)))
            ###########################

            fake_in = _get_fake_in_G(fake, netG, ctx)
            fake_label = ndarray.zeros(len(fake), ctx=ctx)

            with autograd.record():

                y_hats = netD(fake_in)

                ls = loss(y_hats, fake_label)

            ls.backward()

            gen_acc_sum += (y_hats.argmax(axis=1) == fake_label).sum().asscalar()

            gen_l_sum += ls.sum().asscalar()

            trainerG.step(len(fake))

            gm += len(fake)

            if print_batches and (i + 1) % print_batches == 0:
                print("batch %d, dis_loss %f, dis acc %f,gen_loss %f, gen acc %f" % (

                    n, dis_l_sum / n, disc_acc_sum / dm, gen_l_sum / n, gen_acc_sum / gm

                ))
        test_acc = evaluate_accuracy(test_iter, netD, netG, ctx)

        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" % (

            epoch, dis_l_sum / n, disc_acc_sum / dm, test_acc, time.time() - start

        ))


ctx = get_ctx()
netD = model.get_netD()
netG = model.get_netG()
if os.path.exists('paramD'):
    netD.load_params('paramD', ctx=ctx)
else:
    netD.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)

if os.path.exists('paramG'):
    netG.load_params('paramG', ctx=ctx)
else:
    netG.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)

loss = gluon.loss.SoftmaxCrossEntropyLoss()

trainerD = gluon.Trainer(netD.collect_params(), 'sgd', {'learning_rate': 0.008})
trainerG = gluon.Trainer(netG.collect_params(), 'sgd', {'learning_rate': 0.008})
import random

traindata, testdata = load.loadpath()
batch_size = 32
train(traindata, testdata, batch_size, netD, netG, loss, trainerD, trainerG, ctx, 30, 10)
