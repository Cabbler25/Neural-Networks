import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def feedforward_Propogation(W1, W2, data):
    # input -> hiddenlayer -> outputlayer

    data = data.transpose()

    # add bias term
    data = np.concatenate((data, np.ones((1, np.size(data, 1)))))

    # eq 1
    # simple linear combination
    eq_one = np.dot(W1, data)

    # eq 2
    # sigmoid activation function
    eq_two = sigmoid(eq_one)

    # add bias term
    eq_two = np.concatenate((eq_two, np.ones((1, np.size(eq_two, 1)))))

    # eq 3
    # another simple linear combination
    # this time with hiddenlayer rather than input
    eq_three = np.dot(W2, eq_two)

    # eq 4
    # sigmoid activation function
    eq_four = sigmoid(eq_three)

    hiddenlayer = eq_two
    outputlayer = eq_four
    # return data back in original dims with added bias term
    # eq two is hidden layer, eq four is output layer
    return hiddenlayer, outputlayer, data.transpose()

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines
    # Sigmoid = 1 / 1 + e^(-x)
    return 1 / (1 + np.exp(-1 * z))

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem)
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args

    # First reshape 'params' vector into 2 matrices of weights W1 and W2
    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # in general,
    # x = train_data
    # y = labels,
    # z = hiddenlayer_outputs,
    # o = outputlayer_outputs,

    ### feedforward propogation, eq 1-4 ###
    train_label = train_label.astype(int)
    hiddenlayer_outputs, outputlayer_outputs, train_data = feedforward_Propogation(W1, W2, train_data)

    ### error backpropogation, eq 5-17 ###

    # convert labels using 1ofK encoding
    labels = np.zeros((np.size(train_label), n_class))
    labels[np.arange(np.size(train_label)), train_label] = 1
    #for i in range(np.size(train_label)): works but better w.o for loop
    #    labels[i][train_label[i]] = 1
    labels = labels.transpose()

    # eq 5, error function
    # don't do the whole eq since it's used in eq 6 - 7
    # only do inner k summation
    # labels * ln(outputlayer) + (1 - labels) * ln(1-outputlayer)
    #eq_five = np.sum(label * np.log(output_layer_output) + (1 - label) * np.log(1 - output_layer_output))
    #print(np.shape(eq_five))
    #print(eq_five)
    eq_five = labels * np.log(outputlayer_outputs) + (1 - labels) * np.log(1 - outputlayer_outputs)

    # eq 6 - 7
    # error for each input x
    # 1 / n * negative summation of eq five
    n = train_data.shape[0]
    eq_seven = -np.sum(eq_five)
    eq_six = (1 / n) * eq_seven
    #print(eq_six)

    # eq 8 - 9
    # derivative of obj function with respect to W2 (outputlayer)
    # (o - y) dot z
    # or (outputlayer - labels) dot hiddenlayer
    #eq_nine = (outputlayer_outputs - label) * hiddenlayer_outputs
    helper = outputlayer_outputs - labels # we use a helper just because eq_eight is to be reused
    eq_eight = np.dot(hiddenlayer_outputs, helper.transpose())

    # eq 10-12
    # derivative of obj function with respect to weights W1 (hiddenlayer)
    # dot product of x comes last so shapes match up
    # also chop off last row, dont need gradient for bias terms
    # dot(x,(1 - hiddenlayer) * hiddenlayer * ((dot(outputlayer - label, W2))
    #print(np.shape((1-hiddenlayer_outputs)*hiddenlayer_outputs))
    a = (1 - hiddenlayer_outputs) * hiddenlayer_outputs
    #b = np.dot((outputlayer_outputs - labels), W2) W2 must go first
    b = np.dot(W2.transpose(), (outputlayer_outputs - labels))
    c = a * b
    eq_twelve = np.dot(c, train_data)[0:-1]

    # 13-14 gradient is dealt with in nnscript

    # regularization
    # eq 15
    # eq_six + (lamda/2n) * (sum of (W1^2 + W2^2))
    # this is also our objective function!
    eq_fifteen = eq_six + (lambdaval / (2 * n)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    obj_val = eq_fifteen

    # eq 16
    # obj func partial derivative for hidden -> output layer w/ respect to W2
    # (1 / n) * (summation of (eq_eight + (lamda * W2)))
    eq_sixteen = (1 / n) * (eq_eight.transpose() + (lambdaval * W2))

    # eq 17
    # obj func partial derivative for input -> hidden layer w/ respect to W1
    # identical as 16 but for W1 using eq_twelve
    eq_seventeen = (1 / n) * (eq_twelve + (lambdaval * W1))

    # obj_grad is a vector, not a matrix, so we must flatten
    # eq_17 must come first when flattening
    # as correct order is input -> hidden -> output
    obj_grad = np.concatenate((eq_seventeen.flatten(), eq_sixteen.flatten()))

    return (obj_val, obj_grad)

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels
    '''

    label = np.zeros((data.shape[0],))
    # Your code here
    hiddenlayer_output, outputlayer_output, data = feedforward_Propogation(W1, W2, data)
    idx = 0
    for i in outputlayer_output.T:
        label[idx] = np.argmax(i)
        idx += 1
    return label

