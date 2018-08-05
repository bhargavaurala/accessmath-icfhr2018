
# ==========================================================================
#  Code here is a modified/adapted version of theano tutorials found at
#     http://deeplearning.net/tutorial/mlp.html
# ==========================================================================

import pickle
import timeit
import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        self.conf_n_in = n_in
        self.conf_n_out = n_out

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.n_neurons = n_out

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)

        # parameters of the model
        self.params = [self.W, self.b]

        self.conf_rng = rng
        self.conf_n_in = n_in
        self.conf_n_out = n_out
        self.conf_activation=activation

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    BatchSize = 20
    RandomGenerator = np.random.RandomState(125)
    L1_reg = 0.0
    L2_reg = 0.0001
    N_Epochs = 1000
    LearningRate = 0.01
    Patience = 10000        # look as this many examples regardless
    PatienceIncrease = 2    # wait this much longer when a new best is found
    ImprovementThreshold = 0.995  # a relative improvement of this much is considered significant

    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: list(int)
        :param n_hidden: number of hidden units per hidden layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        assert isinstance(n_hidden , list) and len(n_hidden) > 0

        input = T.matrix('x')

        # One single hidden layer for now with n_hidden neurons ...
        self.hiddenLayers = []

        for idx, layer_hidden in enumerate(n_hidden):
            if idx == 0:
                layer_input = input
                layer_n_in = n_in
            else:
                layer_input = self.hiddenLayers[-1].output
                layer_n_in = self.hiddenLayers[-1].n_neurons

            layer = HiddenLayer(
                rng=MLP.RandomGenerator,
                input=layer_input,
                n_in=layer_n_in,
                n_out=layer_hidden,
                activation=activation
            )

            self.hiddenLayers.append(layer)

        # Output layer is a logistic regressor using Softmax
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=self.hiddenLayers[-1].n_neurons,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = abs(self.logRegressionLayer.W).sum()
        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()

        for idx, layer_hidden in enumerate(n_hidden):
            self.L1 += abs(self.hiddenLayers[idx].W).sum()
            self.L2_sqr += (self.hiddenLayers[idx].W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.logRegressionLayer.params
        for idx, layer_hidden in enumerate(n_hidden):
            self.params += self.hiddenLayers[idx].params

        # keep track of model input
        self.input = input

        # final function once trained ...
        self.trained_f = None

        self.conf_n_in = n_in
        self.conf_n_hidden = n_hidden
        self.conf_n_out = n_out
        self.conf_activation = activation

    def fit(self, data_x, data_y, prc_validation=0.20, verbose=True):
        if verbose:
            print("Preparing for training ....")

        # randomly split validation data ...
        training_data, validation_data = MLP.split_training_validation(data_x, data_y, prc_validation)

        in_training_x, in_training_y = training_data
        in_validation_x, in_validation_y = validation_data

        training_x, training_y = MLP.shared_datasets(in_training_x, in_training_y)
        validation_x, validation_y = MLP.shared_datasets(in_validation_x, in_validation_y)

        n_batches_training = in_training_x.shape[0] // MLP.BatchSize
        n_batches_validation = in_validation_x.shape[0] // MLP.BatchSize

        mb_index = T.lscalar()
        y = T.ivector('y')

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = (
            self.negative_log_likelihood(y)
            + MLP.L1_reg * self.L1
            + MLP.L2_reg * self.L2_sqr
        )

        validate_model = theano.function(
            inputs=[mb_index],
            outputs=self.errors(y),
            givens={
                self.input: validation_x[mb_index * MLP.BatchSize:(mb_index + 1) * MLP.BatchSize],
                y: validation_y[mb_index * MLP.BatchSize:(mb_index + 1) * MLP.BatchSize]
            }
        )

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = [
            (param, param - MLP.LearningRate * gparam) for param, gparam in zip(self.params, gparams)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[mb_index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: training_x[mb_index * MLP.BatchSize: (mb_index + 1) * MLP.BatchSize],
                y: training_y[mb_index * MLP.BatchSize: (mb_index + 1) * MLP.BatchSize]
            }
        )

        if verbose:
            print("Executing training ....")

        patience = MLP.Patience
        validation_frequency = min(n_batches_training, patience // 2)

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < MLP.N_Epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_batches_training):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_batches_training + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i in range(n_batches_validation)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_batches_training,
                            this_validation_loss * 100.0
                        ), end='\r'
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * MLP.ImprovementThreshold:
                            patience = max(patience, iter * MLP.PatienceIncrease)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

        self.trained_f = theano.function([self.input], self.logRegressionLayer.y_pred)

        end_time = timeit.default_timer()

        if verbose:
            print(('\nOptimization complete. Best validation score of %f %% obtained at iteration %i') %
              (best_validation_loss * 100., best_iter + 1))
            print(('Total time %.2fm' % ((end_time - start_time) / 60.)))

    def count_params(self):
        count = 0
        for param in self.params:
            count += param.get_value(borrow=True).size

        return count


    def predict(self, data_x):
        if self.trained_f is None:
            raise Exception("MLP has not been trained")

        prediction = self.trained_f(data_x)

        return prediction

    def compute_hidden_layers_output(self, data_x):
        hidden_layers_function = theano.function([self.input], [self.hiddenLayers[-1].output])

        raw_output = hidden_layers_function(data_x)

        return np.array(raw_output[0])

    def save(self, out_filename):
        # save all parameters required to rebuilt the network (specially after training)
        out_file = open(out_filename, "wb")

        # first, save the most general parameters of the network
        pickle.dump(self.conf_n_in, out_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.conf_n_hidden, out_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.conf_n_out, out_file, pickle.HIGHEST_PROTOCOL)

        if self.conf_activation == T.tanh:
            pickle.dump("tanh", out_file, pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception("Saving current activation function is not yet supported")

        # for each hidden layer, save parameters
        for idx, layer in enumerate(self.hiddenLayers):
            pickle.dump(layer.W.get_value(), out_file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(layer.b.get_value(), out_file, pickle.HIGHEST_PROTOCOL)

        # save output layer parameters
        pickle.dump(self.logRegressionLayer.W.get_value(), out_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.logRegressionLayer.b.get_value(), out_file, pickle.HIGHEST_PROTOCOL)

        # save whether or not the network has been trained ..
        pickle.dump(self.trained_f is None, out_file, pickle.HIGHEST_PROTOCOL)

        out_file.close()

    @staticmethod
    def Load(in_filename):
        in_file = open(in_filename, "rb")

        n_in = pickle.load(in_file)
        n_hidden = pickle.load(in_file)
        n_out = pickle.load(in_file)

        # re-create network ....
        activation_name = pickle.load(in_file)
        if activation_name == "tanh":
            activation_function = T.tanh
        else:
            raise Exception("Loading activation function {0:s} is not yet supported".format(activation_name))

        loaded_mlp = MLP(n_in, n_hidden, n_out, activation_function)

        # for each hidden layer, load parameters
        for idx, layer in enumerate(loaded_mlp.hiddenLayers):
            layer.W.set_value(pickle.load(in_file))
            layer.b.set_value(pickle.load(in_file))

        loaded_mlp.logRegressionLayer.W.set_value(pickle.load(in_file))
        loaded_mlp.logRegressionLayer.b.set_value(pickle.load(in_file))

        trained_network = pickle.load(in_file)
        if trained_network:
            loaded_mlp.trained_f = theano.function([loaded_mlp.input], loaded_mlp.logRegressionLayer.y_pred)

        in_file.close()

        return loaded_mlp


    @staticmethod
    def split_training_validation(data_x, data_y, prc_validation):
        if isinstance(data_x, list):
            data_x = np.array(data_x)
        if isinstance(data_y, list):
            data_y = np.array(data_y)

        n_samples = data_x.shape[0]

        # generate a random split of data
        random_vals = np.random.rand(n_samples).tolist()
        validation_vals = (np.arange(n_samples) < n_samples * prc_validation).astype(np.int32).tolist()
        validation_indices = np.array(sorted(zip(random_vals, validation_vals)))[:, 1]

        training_x = data_x[validation_indices == 0, :]
        training_y = data_y[validation_indices == 0]

        validation_x = data_x[validation_indices > 0, :]
        validation_y = data_y[validation_indices > 0]

        return (training_x, training_y), (validation_x, validation_y)

    @staticmethod
    def shared_datasets(data_x, data_y):
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)

        return shared_x, T.cast(shared_y, 'int32')

    @staticmethod
    def per_layer_fit(n_in, n_hidden, n_out, data_x, data_y, prc_validation=0.20):
        previous_mlp = None
        for idx in range(len(n_hidden)):
            current_hidden = n_hidden[:idx + 1]
            print("=> Currently Training: " + str(current_hidden))
            current_mlp = MLP(n_in, current_hidden, n_out)

            if previous_mlp is not None:
                # use the previously learn weights for the first N-1 layers ...
                for pre_idx, layer in enumerate(previous_mlp.hiddenLayers):
                    current_mlp.hiddenLayers[pre_idx].W.set_value(layer.W.get_value())
                    current_mlp.hiddenLayers[pre_idx].b.set_value(layer.b.get_value())

            # train the new network with the added layer ...
            current_mlp.fit(data_x, data_y, prc_validation)

            previous_mlp = current_mlp

        # return the last trained MLP
        return previous_mlp

    @staticmethod
    def fix_layer_per_layer_fit(n_in, n_hidden, n_out, data_x, data_y, prc_validation=0.20):
        single_layer_mlps = []
        last_x = None
        for idx in range(len(n_hidden)):
            print("=> Currently Training: " + str(n_hidden[:idx + 1]))

            if idx == 0:
                layer_n_in = n_in
                current_x = data_x
            else:
                layer_n_in = n_hidden[idx - 1]
                current_x = single_layer_mlps[-1].compute_hidden_layers_output(last_x)

            current_mlp = MLP(layer_n_in, [n_hidden[idx]], n_out)
            current_mlp.fit(current_x, data_y, prc_validation)
            single_layer_mlps.append(current_mlp)

            last_x = current_x

        final_mlp = MLP(n_in, n_hidden, n_out)
        # copy all weights
        for idx in range(len(n_hidden)):
            # hidden layer weights
            layer = single_layer_mlps[idx].hiddenLayers[0]
            final_layer = final_mlp.hiddenLayers[idx]
            final_layer.W.set_value(layer.W.get_value())
            final_layer.b.set_value(layer.b.get_value())

            if idx == len(n_hidden) - 1:
                # copy the regression layer too....
                final_mlp.logRegressionLayer.W.set_value(single_layer_mlps[idx].logRegressionLayer.W.get_value())
                final_mlp.logRegressionLayer.b.set_value(single_layer_mlps[idx].logRegressionLayer.b.get_value())

        # final_mlp.trained_f = theano.function([final_mlp.input], final_mlp.logRegressionLayer.y_pred)
        final_mlp.fit(data_x, data_y)

        return final_mlp
