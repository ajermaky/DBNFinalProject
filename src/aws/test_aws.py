import numpy

from src.data import load_data, prep_theano_data
from src.dbn.dbn_loop import test_DBN
from src.svm import run_svm


def test_aws():
    datasets = load_data('mnist.pkl.gz')
    trainx, trainy = datasets[0]
    validx, validy = datasets[1]
    testx, testy = datasets[2]
    numpy_rng = numpy.random.RandomState(123)

    num_of_train_rows = trainx.shape[0]
    num_of_valid_rows = validx.shape[0]
    num_of_test_rows = testx.shape[0]
    train_indices = range(num_of_train_rows)
    valid_indices = range(num_of_valid_rows)
    test_indices = range(num_of_test_rows)
    numpy_rng.shuffle(range(num_of_train_rows))
    numpy_rng.shuffle(range(num_of_valid_rows))
    numpy_rng.shuffle(range(num_of_test_rows))

    samples = [(10, 10, 10), (50, 10, 10), (100, 10, 10)]
    fd = open('data/aws_dbn_test.log', 'w+')
    fe = open('data/aws_svm_test.log', 'w+')

    for s in samples:
        data = ((trainx[train_indices[:s[0]]], trainy[train_indices[:s[0]]]), (validx[valid_indices[:s[1]]], validy[valid_indices[:s[1]]]),
                (testx[test_indices[:s[2]]], testy[test_indices[:s[2]]]))
        theano_datasets = prep_theano_data(data)

        fd.write("\n\n==========Samples: %s ===========\n\n" % str(s))
        fe.write("\n\n==========Samples: %s ===========\n\n" % str(s))
        test_DBN(pretraining_epochs=100, pretrain_lr=0.01, k=1,
                 training_epochs=1000, finetune_lr=0.1,
                 datasets=theano_datasets, batch_size=10,
                 hidden_layers=[100, 100, 100], fd=fd)

        for c in [.01, .1, 1, 10, 100]:
            fe.write("\n---------------C: %f -------------\n" %c)
            run_svm(digits=data,C=c, fd=fe)



    fd.close()
    fe.close()

if __name__ == "__main__":
    test_aws()
