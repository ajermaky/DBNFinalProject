from __future__ import print_function, division

import timeit
import numpy
import sys
import os
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from src.data import load_data, prep_theano_data
from src.dbn.DBN import DBN


def test_DBN(pretrain_lr=0.01, pretraining_epochs=100,
             k=1, finetune_lr=0.1, training_epochs=1000,
             hidden_layers=[1000,1000,1000],
             datasets=None, batch_size=10,fd=sys.stdout):

    if datasets is None:
        datasets = prep_theano_data(load_data('mnist.pkl.gz'))


    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    numpy_rng = numpy.random.RandomState(123)

    fd.write('... building the model\n')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng,
              n_ins=28 * 28,
              hidden_layers_size=hidden_layers,
              n_outs=10)

    fd.write('... getting the pretraining functions\n')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    num_of_train_rows = (train_set_x.shape.eval())[0]
    num_of_valid_rows = (valid_set_x.shape.eval())[0]
    indices = range(num_of_train_rows)
    numpy_rng.shuffle(range(num_of_train_rows))
    train = train_set_x[indices[:num_of_valid_rows]]
    valid = valid_set_x
    fd.write('... pre-training the model\n')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs

        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            fd.write('Pre-training layer %i, epoch %d, cost %s\n' % (i, epoch,numpy.mean(c, dtype='float64')))
            if(epoch%5==0):
                train_free_energy = dbn.rbm_layers[i].free_energy(train)
                valid_free_energy = dbn.rbm_layers[i].free_energy(valid)
                fd.write('Pre-training layer %i, epoch %d, representative training free energy %s\n' % (i, epoch,numpy.mean(train_free_energy.eval(),dtype='float64')))
                fd.write('Pre-training layer %i, epoch %d, validation free energy %s\n' % (i, epoch,numpy.mean(valid_free_energy.eval(),dtype='float64')))

        valid = T.dot(valid,dbn.rbm_layers[i].W)
        train = T.dot(train, dbn.rbm_layers[i].W)

    end_time = timeit.default_timer()

    fd.write('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    fd.write('... getting the finetuning functions')
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    fd.write('... finetuning the model\n')
    # early-stopping parameters

    # look as this many examples regardless
    patience = 4 * n_train_batches

    # wait this much longer when a new best is found
    patience_increase = 2.

    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    # go through this many minibatches before checking the network on
    # the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                fd.write('epoch %i, minibatch %i/%i, validation error %f %%\n' % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    fd.write(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%\n') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    fd.write(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           'with test performance %f %%\n'
           ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    fd.write('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm\n' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    datasets = load_data('mnist.pkl.gz')
    fd = open('dbn.log','w+')
    # trainx,trainy = datasets[0]
    # validx,validy = datasets[1]
    # testx,testy = datasets[2]
    #
    # datasets = ((trainx[:100],trainy[:100]),(validx[:50],validy[:50]),(testx[:50],testy[:50]))
    theano_datasets = prep_theano_data(datasets)

    test_DBN(pretraining_epochs=100, pretrain_lr=0.01, k=1,
             training_epochs=1000,finetune_lr=0.1,
             datasets=theano_datasets, batch_size=10,
             hidden_layers=[1000,1000,1000],fd=fd)
    fd.close()