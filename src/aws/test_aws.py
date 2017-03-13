from src.data import load_data, prep_theano_data
from src.dbn.dbn_loop import test_DBN
from src.svm import run_svm


def test_aws():
    datasets = load_data('mnist.pkl.gz')
    fd = open('data/aws_dbn_test.log', 'w+')
    trainx,trainy = datasets[0]
    validx,validy = datasets[1]
    testx,testy = datasets[2]

    datasets = ((trainx[:100],trainy[:100]),(validx[:50],validy[:50]),(testx[:50],testy[:50]))
    theano_datasets = prep_theano_data(datasets)

    test_DBN(pretraining_epochs=100, pretrain_lr=0.01, k=1,
             training_epochs=1000, finetune_lr=0.1,
             datasets=theano_datasets, batch_size=10,
             hidden_layers=[100, 100, 100], fd=fd)
    fd.close()
    fd = open('data/aws_svm_test.log','w+')
    run_svm(digits=datasets,fd=fd)
    fd.close()

if __name__=="__main__":
    test_aws()