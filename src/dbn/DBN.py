import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams

from src.HiddenLayer import HiddenLayer
from src.LogisticRegression import LogisticRegression
from src.rbm.RBM import RBM


class DBN(object):
    def __init__(
            self,
            numpy_rng,
            theano_rng = None,
            n_ins=784,
            hidden_layers_size = [500,500],
            n_outs = 10,
            normal = False,
    ):

        self.sigmoid_layers=[]
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_size)

        assert self.n_layers>0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(np.random.randint(2**30))

        self.theano_rng = theano_rng
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in range(self.n_layers):
            if i==0:
                input_size =n_ins
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
                input_size= hidden_layers_size[i-1]


            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in = input_size,
                n_out=hidden_layers_size[i],
                activation=T.nnet.sigmoid
            )

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            rbm_layer = RBM(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_size[i],
                W=sigmoid_layer.W,
                hbias=sigmoid_layer.b,
                normal=normal
            )

            self.rbm_layers.append(rbm_layer)
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_size[-1],
            n_out=n_outs
        )
        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self,train_set_x,batch_size,k):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        batch_begin = index*batch_size

        batch_end = batch_begin+batch_size

        pretrain_fns=[]
        for rbm in self.rbm_layers:
            cost,updates = rbm.get_cost_updates(learning_rate,persistent=None,k=k)

            fn = theano.function(
                inputs=[index, theano.In(learning_rate,value=.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x:train_set_x[batch_begin:batch_end]
                }
            )

            pretrain_fns.append(fn)

        return pretrain_fns


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: train_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: test_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: valid_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score