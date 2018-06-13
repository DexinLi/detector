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


def get_fake(batch):
    fake = []
    for i in range(len(batch[1])):
        if batch[1][i] == 1:
            data = batch[0][i]
            fake.append(batch[0][i])
    return fake


def get_fake_out(fake, netG, ctx):
    fake_in = [x + [0] * (1024 * 1024 - len(x)) for x in fake]
    fake_in = ndarray.array(fake_in, ctx)
    fake_out = netG(fake_in)
    for i in range(len(fake)):
        l = len(fake[i])
        prefix = ndarray.zeros(l, ctx)
        payload = fake_out[i].take([l, l + 1024])
        suffix = ndarray.zeros(1024 * 1023 - l, ctx)
        fake_out[i] = fake_out[i] + ndarray.concat(prefix, payload, suffix)
    return fake_out


def get_iter(batch, ctx):
    label = []
    feature = [data + [0] * (1024 * 1024 - len(data)) for data in batch]
    return ndarray.array(feature, ctx), ndarray.array(label, ctx)


ctx = get_ctx()
netD0 = model.get_netD()
netD0.load_params('paramD0', ctx=ctx)


def evaluate_accuracy(data_iter, netD, netG, ctx):
    acc = ndarray.array([0])
    acc1 = ndarray.array([0])
    n = 0
    for batch in data_iter:
        fake = get_fake(batch)
        fake_out = get_fake_out(fake, netG, ctx)
        feat, label = get_iter(batch, ctx)
        fake_label = ndarray.ones(len(fake_out), ctx=ctx)
        acc += (netD(feat).argmax(axis=1) == label).sum().copyto(mxnet.cpu())
        acc1 += (netD0(fake_out).argmax(axis=1) == fake_label).sum().copyto(mxnet.cpu())
        n += len(fake)

        acc.wait_to_read()
        acc1.wait_to_read()

    return acc.asscalar() / 1000, acc1.asscalar() / n


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

            # fake_in = _get_fake_in_D(fake, netG, ctx)
            # fake_label = ndarray.ones(len(fake), ctx=ctx)
            # Xs = ndarray.concat(feat, fake_in, dim=0)
            # ys = ndarray.concat(label, fake_label, dim=0)
            #
            # with autograd.record():
            #
            #     y_hats = netD(Xs)
            #     disc_acc_sum += (y_hats.argmax(axis=1) == ys).sum().asscalar()
            #     ls = loss(y_hats, ys)
            #     dis_l_sum += ls.sum().asscalar()
            #     ls.backward()

            # trainerD.step(len(fake) + len(feat))
            # params = netD.collect_params()
            # for x in params:
            #     p = params[x].data()
            #     ndarray.clip(p, -c, c, out=p)

            # n += len(feat)
            #
            # dm += len(feat) + len(fake)

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z)))
            ###########################

            fake = get_fake(batch)
            fake_out = get_fake_out(fake,netG,ctx)
            fake_label = ndarray.zeros(len(fake), ctx=ctx)

            with autograd.record():

                y_hats = netD(fake_out)

                ls = loss(y_hats, fake_label)

            ls.backward()

            gen_acc_sum += (y_hats.argmax(axis=1) == fake_label).sum().asscalar()

            gen_l_sum += ls.sum().asscalar()

            trainerG.step(len(fake), True)

            gm += len(fake)

            # if print_batches and (i + 1) % print_batches == 0:
            #     print("batch %d, dis_loss %f, dis acc %f,gen_loss %f, gen acc %f" % (
            #
            #         n, dis_l_sum / dm, disc_acc_sum / dm, gen_l_sum / gm, gen_acc_sum / gm
            #
            #     ))
            if print_batches and (i + 1) % print_batches == 0:
                print("batch %d, gen_loss %f, gen acc %f" % (

                    gm, gen_l_sum / gm, gen_acc_sum / gm

                ))
        test_acc, test_acc1 = evaluate_accuracy(test_iter, netD, netG, ctx)

        # print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f,test acc1 %.3f, time %.1f sec" % (
        #
        #     epoch, dis_l_sum / dm, disc_acc_sum / dm, test_acc, test_acc1, time.time() - start
        #
        # ))
        print("epoch %d, test acc %.3f,test acc1 %.3f, time %.1f sec" % (

            epoch, test_acc, test_acc1, time.time() - start

        ))
        # netD.save_params("%d-test-acc_%.3fD" % (epoch, test_acc))
        netG.save_params("%d-test-acc_%.3fG" % (epoch, test_acc1))


netD = model.get_netD()
netG = model.get_netG1(ctx)
if os.path.exists('paramD'):
    netD.load_params('paramD', ctx=ctx)
else:
    netD.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)

if os.path.exists('paramG'):
    netG.load_params('paramG', ctx=ctx)
else:
    netG.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
schedulerD = mxnet.lr_scheduler.FactorScheduler(100, 0.9)
schedulerG = mxnet.lr_scheduler.FactorScheduler(100, 0.9)
trainerD = gluon.Trainer(netD.collect_params(), 'sgd',
                         {'learning_rate': 0.01, 'wd': 1.5e-4, 'lr_scheduler': schedulerD, 'momentum': 0.9})
trainerG = gluon.Trainer(netG.collect_params(), 'sgd',
                         {'learning_rate': 0.01, 'wd': 1.5e-4, 'lr_scheduler': schedulerG, 'momentum': 0.9})
import random

traindata, testdata = load.loadpath()
batch_size = 25
train(traindata, testdata, batch_size, netD, netG, loss, trainerD, trainerG, ctx, 30, 10)
