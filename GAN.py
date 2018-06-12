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


def _get_fake_in_D(fake, netG, ctx):
    n = len(fake)
    fake_in = [add_noise(x) for x in fake]
    padding = [[0] * (1024 * 1023 - len(x)) for x in fake]
    fake_in = ndarray.array(fake_in, ctx)
    fake_out = netG(fake_in).detach()
    res = [[0] * (1024 * 1024 - len(x)) for x in fake]
    res = ndarray.array(res, ctx)
    for x, y, z, i in zip(fake, fake_out, padding, range(n)):
        pre = ndarray.zeros(len(x), ctx=ctx)
        suf = ndarray.array(z, ctx=ctx)
        data = ndarray.concat(pre, y, suf, dim=0)
        res[i] = data + res[i]
    return res


def _get_fake_in_G(fake, netG, ctx):
    n = len(fake)
    fake_in = [add_noise(x) for x in fake]
    padding = [[0] * (1024 * 1023 - len(x)) for x in fake]
    fake_in = ndarray.array(fake_in, ctx)
    fake_out = netG(fake_in)
    res = [[0] * (1024 * 1024 - len(x)) for x in fake]
    res = ndarray.array(res, ctx)
    for x, y, z, i in zip(fake, fake_out, padding, range(n)):
        pre = ndarray.zeros(len(x), ctx=ctx)
        suf = ndarray.array(z, ctx=ctx)
        data = ndarray.concat(pre, y, suf, dim=0)
        res[i] = data + res[i]
    return res


def get_iter1(batch, ctx):
    fake = []
    for i in range(len(batch[1])):
        if batch[1][i] == 1:
            fake.append(batch[0][i])
    feature, label = batch
    feature = [data + [0] * (1024 * 1024 - len(data)) for data in feature]
    return fake, ndarray.array(feature, ctx), ndarray.array(label, ctx)


def get_iter(batch, ctx):
    fake = []
    feature = []
    label = []
    for i in range(len(batch[1])):
        if batch[1][i] == 1 and i % 2 == 0:
            fake.append(batch[0][i])
        else:
            feature.append(batch[0][i])
            label.append(batch[1][i])
    feature = [data + [0] * (1024 * 1024 - len(data)) for data in feature]
    return fake, ndarray.array(feature, ctx), ndarray.array(label, ctx)


ctx = get_ctx()
netD0 = model.get_netD()
netD0.load_params('paramD0', ctx=ctx)


def evaluate_accuracy(data_iter, netD, netG, ctx):
    acc = ndarray.array([0])
    acc1 = ndarray.array([0])
    n = 0
    for batch in data_iter:
        fake, feat, label = get_iter1(batch, ctx)
        fake_in = _get_fake_in_D(fake, netG, ctx)
        fake_label = ndarray.ones(len(fake), ctx=ctx)
        acc += (netD(feat).argmax(axis=1) == label).sum().copyto(mxnet.cpu())
        acc += (netD(fake_in).argmax(axis=1) == fake_label).sum().copyto(mxnet.cpu())
        acc1 += (netD0(fake_in).argmax(axis=1) == fake_label).sum().copyto(mxnet.cpu())
        n += len(fake)

        acc.wait_to_read()
        acc1.wait_to_read()

    return acc.asscalar() / (1000 + n), acc1.asscalar() / n


def train(train_data, test_data, batch_size, netD, netG, loss, trainerD, trainerG, ctx, num_epochs, print_batches=None):
    """Train and evaluate a model."""

    print("training on", ctx)
    c = 0.01

    for epoch in range(1, num_epochs + 1):
        train_iter = load.get_iter(train_data, batch_size=batch_size)
        test_iter = load.get_iter(test_data, 25)

        dis_l_sum, disc_acc_sum, n, dm = 0.0, 0.0, 0.0, 0.0
        gen_l_sum, gen_acc_sum, gm = 0.0, 0.0, 0.0

        start = time.time()

        i = 0
        for batch in train_iter:
            i += 1

            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################

            fake, feat, label = get_iter1(batch, ctx)
            fake_in = _get_fake_in_D(fake, netG, ctx)
            fake_label = ndarray.ones(len(fake), ctx=ctx)
            Xs = ndarray.concat(feat, fake_in, dim=0)
            ys = ndarray.concat(label, fake_label, dim=0)

            with autograd.record():

                y_hats = netD(Xs)
                disc_acc_sum += (y_hats.argmax(axis=1) == ys).sum().asscalar()
                ls = loss(y_hats, ys)
                dis_l_sum += ls.sum().asscalar()
                ls.backward()

            trainerD.step(len(fake) + len(feat))
            # params = netD.collect_params()
            # for x in params:
            #     p = params[x].data()
            #     ndarray.clip(p, -c, c, out=p)

            n += len(feat)

            dm += len(feat) + len(fake)

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

            trainerG.step(len(fake), True)

            gm += len(fake)

            if print_batches and (i + 1) % print_batches == 0:
                print("batch %d, dis_loss %f, dis acc %f,gen_loss %f, gen acc %f" % (

                    n, dis_l_sum / dm, disc_acc_sum / dm, gen_l_sum / gm, gen_acc_sum / gm

                ))
        test_acc, test_acc1 = evaluate_accuracy(test_iter, netD, netG, ctx)

        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f,test acc1 %.3f, time %.1f sec" % (

            epoch, dis_l_sum / dm, disc_acc_sum / dm, test_acc, test_acc1, time.time() - start

        ))
        netD.save_params("%d-test-acc_%.3fD" % (epoch, test_acc))
        netG.save_params("%d-test-acc_%.3fG" % (epoch, test_acc1))


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
schedulerD = mxnet.lr_scheduler.FactorScheduler(100, 0.88)
schedulerG = mxnet.lr_scheduler.FactorScheduler(100, 0.9)
trainerD = gluon.Trainer(netD.collect_params(), 'sgd',
                         {'learning_rate': 0.01, 'wd': 1.5e-4, 'lr_scheduler': schedulerD, 'momentum': 0.9})
trainerG = gluon.Trainer(netG.collect_params(), 'sgd',
                         {'learning_rate': 0.01, 'wd': 1.5e-4, 'lr_scheduler': schedulerG, 'momentum': 0.9})
import random

traindata, testdata = load.loadpath()
batch_size = 25
train(traindata, testdata, batch_size, netD, netG, loss, trainerD, trainerG, ctx, 30, 10)
