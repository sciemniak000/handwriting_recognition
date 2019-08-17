import random
import numpy as np


def relu(x):
    """Activation function that prevent
    negative values from occuring"""
    return max(0, x)


"""relu applied to numpy matrices"""
real_relu = np.vectorize(relu)


def der_relu(x):
    """Derivative of relu"""
    if x > 0:
        return 1
    else:
        return 0


"""der_relu for numpy matrices"""
real_der_relu = np.vectorize(der_relu)


class ConvolutionLayer:
    """Type of layer which counts 'amount_of_filters' convolutions on its input,
    thus it counts 'amount_of_filters activations of size 'size_of_activations'"""

    def __init__(self, amount_of_filters, size_of_activations, size_of_filter, previous=None):
        self.amount_of_filters = amount_of_filters
        self.size_of_activations = size_of_activations
        self.size_of_filter = size_of_filter

        self.previous = previous
        self.activations = np.zeros((amount_of_filters, size_of_activations[0], size_of_activations[1]))
        self.filters = np.random.random((amount_of_filters, size_of_filter[0], size_of_filter[1]))

    def count_activations(self, input=None):
        """if there's an input, then this is the first layer and the input is the image we try to recognize"""
        if input is not None:
            for i in range(self.amount_of_filters):
                """size of filters should be odd, so there is central cell
                stepx and stepy are the amount of rows and columns that we cut from the borders due to the
                size of the filters;
                for example when the filter is 3x3, in the output activations there will be 1 less cell on each side
                than in the input image"""
                stepx = int((self.filters[i].shape[0] - 1) / 2)
                stepy = int((self.filters[i].shape[1] - 1) / 2)

                for x in range(stepx, input.shape[0] - stepx):
                    for y in range(stepy, input.shape[1] - stepy):

                        """count every element of output activations,
                        simple convolution with relu applied to the result"""
                        self.activations[i, x - stepx, y - stepy] = relu(sum(np.ndarray.flatten(np.multiply(self.filters[i],
                                                            input[x - stepx:x + stepx + 1, y - stepy:y + stepy + 1]))))

        else:
            for i in range(self.amount_of_filters):
                """stepx and stepy are explained in the previous if block"""
                stepx = int((self.filters[i].shape[0] - 1)/2)
                stepy = int((self.filters[i].shape[1] - 1)/2)
                for x in range(stepx, self.previous.activations[i].shape[0] - stepx):
                    for y in range(stepy, self.previous.activations[i].shape[1] - stepy):

                        """just as in the previous if block,
                        but instead of the input image we take output of the previous layer
                        (it means that this isn't the first layer"""
                        self.activations[i, x - stepx, y - stepy] = relu(sum(np.ndarray.flatten(np.multiply(self.filters[i],
                                                self.previous.activations[i, x - stepx:x + stepx + 1, y - stepy:y +
                                                stepy + 1]))))

    def set_previous(self, prev):
        self.previous = prev


class PoolLayer:
    """This is max pooling. It takes 2D input (output of previous layer) and chooses the max value of the mini-block
    of size 'size_of_pooling' (there are 'amount_of_filters' 2D arrays in the input and in the output activations);
    the mini-blocks are made by moving the pooling window on the input by step of size 'stride'"""
    def __init__(self, amount_of_filters, size_of_activations, size_of_pooling, stride, previous = None):
        self.amount_of_filters = amount_of_filters
        self.size_of_activations = size_of_activations
        self.size_of_pooling = size_of_pooling
        self.stride = stride
        # self.method = method

        self.activations = np.zeros((amount_of_filters, size_of_activations[0], size_of_activations[1]))
        self.previous = previous

    def count_activations(self):
        for n in range(self.amount_of_filters):

            for i in range(0, self.size_of_activations[0], self.stride):
                for j in range(0, self.size_of_activations[1], self.stride):
                    self.activations[n, i, j] = max(np.ndarray.flatten(self.previous.activations[n, i:i + self.size_of_pooling[0],
                                                    j:j + self.size_of_pooling[1]]))

    def set_previous(self, prev):
        self.previous = prev


class OutputLayer:
    """This layer is just the threshold between 2D layers and 1D layers,
    it takes the 2D output of the previous layer and puts it as the output in 1D"""
    def __init__(self, previous=None):
        self.previous = previous
        self.activations = None
        self.z = None

    def count_activations(self):
        self.activations = np.ndarray.flatten(self.previous.activations)
        self.z = self.activations

    def set_previous(self, prev):
        self.previous = prev
        self.activations = np.ndarray.flatten(self.previous.activations)


class FullyConnectedLayer:
    """Most classical feedforward layer"""
    def __init__(self, size):
        self.activations = np.zeros(size)
        if self.activations.ndim > 1:
            raise IndexError
        self.previous = None
        self.weights = None
        self.biases = np.random.random(size)

        """z will be useful when training the net;
        z = a^l-1 * w + b"""
        self.z = None

    def set_previous(self, prev):
        self.previous = prev
        if prev.activations.ndim > 1:
            raise IndexError

        """weights should be initialized but their size is known only when the previous is set"""
        self.weights = np.random.random((self.previous.activations.size, self.activations.size))

    def count_activations(self):
        self.activations = real_relu(np.dot(self.previous.activations, self.weights) + self.biases)
        self.z = np.dot(self.previous.activations, self.weights) + self.biases


class ConvolutionalNetwork:
    """This is just the container for the layers given in Python list"""
    def __init__(self, layers):
        self.layers = layers
        for i in range(1, len(layers)):
            layers[i].set_previous(layers[i - 1])

    def count_activations(self, image):
        """Counts activations iteratively through all the layers and returns activation of the last layer
        which should be a vector with 10 elements, one for each digit"""
        self.layers[0].count_activations(image)
        for n in range(1, len(self.layers)):
            self.layers[n].count_activations()
        return self.layers[len(self.layers) - 1].activations

    def determine_number(self, image):
        """Calls count_activations and then checks, what digit did the net recognize"""
        act = self.count_activations(image)
        result = 0
        maxi = act[0]
        for i in range(1, 10):
            if act[i] > maxi:
                maxi = act[i]
                result = i

        return result

    def train_network(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Trains network with stochastic gradient descent method.
        'training_data' is a list of tuples (image, label), labels should be vector of size 10,
        full of zeros except for the node responsible for the correct digit, should it be 1.
        The training will be held in 'epochs' turns,
        in each turn, in each epoch, the training data will be shuffled and divided
        into batches of size 'mini_batch_size',
        then the algorithm should be run over each batch separately.
        'eta' determines how fast we're going to change the values of the net (risk of jumping over the right values)
        'test_data' is optional, it should has the same format as 'training_data'
        if test data is provided, after each epoch the result of training should be checked and printed
        in form of amount of recognized images from test data"""

        n = len(training_data)

        for ep in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in batches:
                """the training algorithm is run here"""
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                self.test_the_net(ep, test_data)

    def update_mini_batch(self, batch, eta):
        """Update the values of the net according to specified batch of training data"""

        """Store deltas for all the images from the barch, sum them in those lists below.
        They are all None and size of list of layers,
        when counting, some of the Nones will be replaced by the sums, only some of them,
        this may seem strange but it simplifies applying of those values later"""
        conv_weights = [None] * len(self.layers)
        full_biases = [None] * len(self.layers)
        full_weights = [None] * len(self.layers)

        for sets in batch:

            """First gather the error of each layer"""
            errors = []

            """Cost function is cross-enthropy function,
            for the delta of last fully connected layer just subtract the expected values from actual values;
            
            Notice, this line of code counts the feedforward activations
            for all the net ('count_activations' function)"""
            errors.append(np.subtract(self.count_activations(sets[0]), sets[1]))

            """Backpropagate the error over all the layers except for the input layer;
            there's no need to count error of the input layer because it will not be needed to count the new weights"""
            for layer in range(-2, -len(self.layers) - 1, -1):

                """The way of counting the error varies according to the type of the layer"""

                if type(self.layers[layer]) is FullyConnectedLayer or type(self.layers[layer]) is OutputLayer:
                    """Error equals to transposed weights of next layer multiplied by error of the following
                    layer, multiplied elementwise by derivative of activation function of z factor of the
                    actual layer"""
                    errors.append(np.multiply(np.dot(errors[-layer - 2], np.transpose(self.layers[layer + 1].weights)
                                                        ), real_der_relu(self.layers[layer].z)))

                else:
                    lay = self.layers[layer]

                    if type(self.layers[layer + 1]) is OutputLayer:
                        er = errors[-1]
                        errors.append(np.reshape(er, lay.activations.shape))

                    elif type(self.layers[layer + 1]) is PoolLayer:
                        """Error is just an array of zeros where you put the error of the following layer
                        in such a way, that the indices of the maximum element in activations of previous layer and
                        the error of this layer are the same indices"""
                        previous_error = errors[-1]
                        errors.append(np.zeros(self.layers[layer].activations.shape))

                        nlay = self.layers[layer + 1]
                        for x in range(previous_error.shape[0]):
                            for y in range(previous_error.shape[1]):
                                for z in range(previous_error.shape[2]):
                                    temp = nlay.previous.activations[x, y*nlay.stride:y*nlay.stride + nlay.size_of_pooling[0],
                                           z*nlay.stride:z*nlay.stride + nlay.size_of_pooling[1]]

                                    ind = np.argmax(temp)
                                    first = int(ind/temp.shape[0])
                                    second = int(ind % temp.shape[0])

                                    errors[-1][x, y*nlay.stride + first, z*nlay.stride + second] = previous_error[x, y, z]

                    elif type(self.layers[layer + 1]) is ConvolutionLayer:
                        """For the Convolution Layer we iterate over the error of the next layer and at the same time
                        we move window size of filter over the current activations of the current layer.
                        Because of the shape of the window each cell is connected with one cell of the filter in every step;
                        We multiply the corresponding activation cells by that one error cell of the next layer,
                        we sum all the corresponding results for each filter cell and in the end we acquire the error
                        for the current layer
                        
                        Notice that there are 'number_of_filters' filters of size 'size_of_filters' in ConvolutionLayer"""

                        """Notice that it's theoretically possible that the next layer is a 1D OutputLayer"""

                        previous_error = errors[-1]

                        nlay = self.layers[layer + 1]

                        errors.append(np.zeros(lay.activations.shape))

                        for x in range(previous_error.shape[0]):
                            for y in range(previous_error.shape[1]):
                                for z in range(previous_error.shape[2]):
                                    errors[-1][x, y:y+nlay.size_of_filter[0], z:z+nlay.size_of_filter[1]] += np.multiply(
                                        nlay.filters[x], previous_error[x, y, z])

            """We iterated through layers from end to beginning but we appended the error which means that it is
            in wrong order, that's why we reverse it"""
            errors = list(reversed(errors))

            """Again iterate over the layers but now count the deltas knowing the errors.
            First sum the deltas up, then make the average, then use the average to calculate new weights and biases
            
            Notice that there's no error for the first layer, so it is shifted in regard of layers"""
            for i in range(len(self.layers)):
                if type(self.layers[i]) is ConvolutionLayer:
                    lay = self.layers[i]

                    """errors[i] is error of the next layer because of the shift
                    theoretically it's possible that next layer is in 1D"""
                    if errors[i].ndim is 1:

                        """Wait a moment, what the fuck, this is a mistake that I don't want to fix right now.
                        Just be sure that there is always a pooling layer after a convolution layer"""
                        # next_error = np.reshape(errors[i], self.layers[i + 1].activations.shape)
                        raise IndexError
                    else:
                        next_error = errors[i]

                    delta_w = np.zeros((lay.amount_of_filters, lay.size_of_filter[0], lay.size_of_filter[1]))

                    if i is not 0:
                        prev_activ = lay.previous.activations

                        for x in range(next_error.shape[0]):
                            for y in range(next_error.shape[1]):
                                for z in range(next_error.shape[2]):
                                    delta_w[x] += np.multiply(
                                        prev_activ[x, y:y + lay.size_of_filter[0], z:z + lay.size_of_filter[1]],
                                        next_error[x, y, z])

                    else:
                        prev_activ = sets[0]
                        prev_activ = np.reshape(prev_activ, (prev_activ.shape[0], prev_activ.shape[1]))

                        for x in range(next_error.shape[0]):
                            for y in range(next_error.shape[1]):
                                for z in range(next_error.shape[2]):
                                    delta_w[x] += np.multiply(
                                        prev_activ[y:y + lay.size_of_filter[0], z:z + lay.size_of_filter[1]],
                                        next_error[x, y, z])

                    if conv_weights[i] is not None:
                        conv_weights[i] += delta_w
                    else:
                        conv_weights[i] = delta_w

                elif type(self.layers[i]) is FullyConnectedLayer:
                    lay = self.layers[i]

                    """Remember, errors[i - 1] correspond with layers[i]"""
                    if full_biases[i] is not None:
                        full_biases[i] += errors[i]
                    else:
                        full_biases[i] = errors[i]

                    delta_w = np.dot(np.reshape(lay.previous.activations, (lay.previous.activations.shape[0], 1)),
                                     np.reshape(errors[i], (1, errors[i].shape[0])))

                    if full_weights[i] is not None:
                        full_weights[i] += delta_w
                    else:
                        full_weights[i] = delta_w

        """Make the average of what you've summed up for all the images and correct the values of the net.
        The counted values are derivatives of cost function over the biases or weights for specific layers.
        To correct the values you take those derivatives, make them negative and scale them, then add them to
        the old weights or biases"""
        for i in range(len(self.layers)):
            if conv_weights[i] is not None:
                conv_weights[i] = np.divide(conv_weights[i], len(batch))

            if full_weights[i] is not None:
                full_weights[i] = np.divide(full_weights[i], len(batch))

            if full_biases[i] is not None:
                full_biases[i] = np.divide(full_biases[i], len(batch))

            if type(self.layers[i]) is ConvolutionLayer:
                self.layers[i].filters += np.multiply(conv_weights[i], -eta)

            elif type(self.layers[i]) is FullyConnectedLayer:
                self.layers[i].weights += np.multiply(full_weights[i], -eta)
                self.layers[i].biases += np.multiply(full_biases[i], -eta)

    def test_the_net(self, ep, test_data):
        counter = 0
        for test in test_data:
            act = test[1]
            expected = 0
            maxi = act[0]
            for i in range(1, 10):
                if act[i] > maxi:
                    maxi = act[i]
                    expected = i
            if self.determine_number(test[0]) == expected:
                counter += 1

        print("epoch {0}: {1}/{2}".format(ep, counter, len(test_data)))

