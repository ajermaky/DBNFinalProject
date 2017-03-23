import sys

import numpy

from src.data import load_data, prep_theano_data
from src.dbn.dbn_loop import test_DBN
from src.svm import run_svm


def aws(name):

    datasets = load_data('mnist.pkl.gz')

    numpy_rng = numpy.random.RandomState(123)


    def getEvenData(data,sample_sizes):
        trainx, trainy = data[0]
        validx, validy = data[1]
        testx, testy = data[2]

        num_of_train_rows = trainx.shape[0]
        num_of_valid_rows = validx.shape[0]
        num_of_test_rows = testx.shape[0]
        train_indices = range(num_of_train_rows)
        valid_indices = range(num_of_valid_rows)
        test_indices = range(num_of_test_rows)
        numpy_rng.shuffle(range(num_of_train_rows))
        numpy_rng.shuffle(range(num_of_valid_rows))
        numpy_rng.shuffle(range(num_of_test_rows))

        tx = trainx[train_indices]
        ty = trainy[train_indices]
        vx = validx[valid_indices]
        vy = validy[valid_indices]
        tex = testx[test_indices]
        tey = testy[test_indices]

        train_each_class = sample_sizes[0]/10
        valid_each_class = sample_sizes[1] / 10
        test_each_class = sample_sizes[2] / 10

        train_new_indices = numpy.where(ty==0)[0][:train_each_class]
        valid_new_indices = numpy.where(vy==0)[0][:valid_each_class]
        test_new_indices = numpy.where(tey==0)[0][:test_each_class]

        for i in range(1,10):
            train_new_indices = numpy.union1d(train_new_indices,numpy.where(ty==i)[0][:train_each_class])

            valid_new_indices = numpy.union1d(valid_new_indices, numpy.where(vy == i)[0][:valid_each_class])
            test_new_indices = numpy.union1d(test_new_indices, numpy.where(tey == i)[0][:test_each_class])

        newdata = ((tx[train_new_indices],ty[train_new_indices]),
                (vx[valid_new_indices],vy[valid_new_indices]),
                (tex[test_new_indices],tey[test_new_indices]))

        return newdata

    prelr = .1
    flr=.1
    s = (5000,1000,1000)
    l = [1000,1000,1000]
    k=1
    name = "2017_03_22_new_data_samples"
    fd = open('data/' + name + '_dbn.log', 'w+')
    fe = open('data/' + name + '_svm.log', 'w+')
    for j in range(3):
        fd.write("_______RUN: %d" % j)
        fe.write("_______RUN: %d" % j)
        for s in [(1000,1000,1000),(2000,1000,1000),(3000,1000,1000),(4000,1000,1000),(5000,1000,1000)]:
            data = getEvenData(datasets,s)
            theano_datasets = prep_theano_data(data)
            print('done prepping data')
            fd.write("\n\n==========Sample Size: %s ===========\n\n" % str(s))
            fe.write("\n\n==========Sample Size: %s ===========\n\n" % str(s))
            test_DBN(pretraining_epochs=100, pretrain_lr=prelr, k=k,
                     training_epochs=1000, finetune_lr=flr,
                     datasets=theano_datasets, batch_size=10,
                     hidden_layers=l, fd=fd, normal_distro=False)

            for c in [10, 100]:
                fe.write("\n---------------C: %f -------------\n" %c)

                run_svm(digits=data, C=c, fd=fe)

    fd.close()
    fe.close()



    prelr = .1
    flr=.1
    s = (5000,1000,1000)
    l = [1000,1000,1000]
    k=1
    name = "2017_03_22_new_data_samples_valid_500"
    fd = open('data/' + name + '_dbn.log', 'w+')
    fe = open('data/' + name + '_svm.log', 'w+')
    for j in range(3):
        fd.write("_______RUN: %d" % j)
        fe.write("_______RUN: %d" % j)
        for s in [(5000,500,500),(5000,1000,1000),(5000,1500,1500),(5000,2000,2000)]:
            data = getEvenData(datasets,s)
            theano_datasets = prep_theano_data(data)
            print('done prepping data')
            fd.write("\n\n==========Sample Size: %s ===========\n\n" % str(s))
            fe.write("\n\n==========Sample Size: %s ===========\n\n" % str(s))
            test_DBN(pretraining_epochs=100, pretrain_lr=prelr, k=k,
                     training_epochs=1000, finetune_lr=flr,
                     datasets=theano_datasets, batch_size=10,
                     hidden_layers=l, fd=fd, normal_distro=False)

            for c in [10, 100]:
                fe.write("\n---------------C: %f -------------\n" %c)

                run_svm(digits=data, C=c, fd=fe)

    fd.close()
    fe.close()

    prelr = .1
    flr=.1
    s = (5000,1000,1000)
    l = [1000,1000,1000]
    k=1
    name = "2017_03_22_new_data_prelr"
    fd = open('data/' + name + '_dbn.log', 'w+')
    fe = open('data/' + name + '_svm.log', 'w+')
    for j in range(3):
        fd.write("_______RUN: %d" % j)
        fe.write("_______RUN: %d" % j)
        for prelr in [.5,.1,.07,.04,.01]:
            data = getEvenData(datasets,s)
            theano_datasets = prep_theano_data(data)
            print('done prepping data')
            fd.write("\n\n==========Prelr: %s ===========\n\n" % str(prelr))
            fe.write("\n\n==========Prelr: %s ===========\n\n" % str(prelr))
            test_DBN(pretraining_epochs=100, pretrain_lr=prelr, k=k,
                     training_epochs=1000, finetune_lr=flr,
                     datasets=theano_datasets, batch_size=10,
                     hidden_layers=l, fd=fd, normal_distro=False)

            for c in [10, 100]:
                fe.write("\n---------------C: %f -------------\n" %c)

                run_svm(digits=data, C=c, fd=fe)

    fd.close()
    fe.close()


    prelr = .1
    flr=.1
    s = (5000,1000,1000)
    l = [1000,1000,1000]
    k=1
    name = "2017_03_22_new_data_layers"
    fd = open('data/' + name + '_dbn.log', 'w+')
    fe = open('data/' + name + '_svm.log', 'w+')
    for j in range(3):
        fd.write("_______RUN: %d" % j)
        fe.write("_______RUN: %d" % j)
        for l in [[1000],[1000,1000],[1000,1000,1000]]:
            data = getEvenData(datasets,s)
            theano_datasets = prep_theano_data(data)
            print('done prepping data')
            fd.write("\n\n==========Layer Size: %s ===========\n\n" % str(l))
            fe.write("\n\n==========Layer Size: %s ===========\n\n" % str(l))
            test_DBN(pretraining_epochs=100, pretrain_lr=prelr, k=k,
                     training_epochs=1000, finetune_lr=flr,
                     datasets=theano_datasets, batch_size=10,
                     hidden_layers=l, fd=fd, normal_distro=False)

            for c in [10, 100]:
                fe.write("\n---------------C: %f -------------\n" %c)

                run_svm(digits=data, C=c, fd=fe)

    fd.close()
    fe.close()
    # datasets = ((trainx[:100],trainy[:100]),(validx[:50],validy[:50]),(testx[:50],testy[:50]))


if __name__=="__main__":
    name = sys.argv[1]
    aws(name)