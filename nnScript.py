'''
Neural Network Script Starts here
'''
import matplotlib.pyplot as plt
import time
from nnFunctions import *
# you may experiment with a small data set (mnist_sample.pickle) first
#filename = 'mnist_sample.pickle'
filename = 'mnist_all.pickle'
#filename = 'AI_quick_draw.pickle'
train_data, train_label, test_data, test_label = preprocess(filename)
'''
with open('params.pickle', 'rb') as f:
    data = pickle.load(f)
    n_hidden = data[0]
    W1 = data[1]
    W2 = data[2]
    lambdaval = data[3]
'''

#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in output unit
n_class = 10

''' Default script (runs once) '''

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# initialize the weights into some random matrices
initial_W1 = initializeWeights(n_input, n_hidden)
initial_W2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_W1.flatten(), initial_W2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# Reshape nnParams from 1D vector into W1 and W2 matrices
W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
print("training done!")

# Test the computed parameters

# find the accuracy on Training Dataset
predicted_label = nnPredict(W1, W2, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# find the accuracy on Testing Dataset
predicted_label = nnPredict(W1, W2, test_data)
print('\n Test set Accuracy:    ' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

#params = [50, W1, W2, 25]
#pickle.dump(params, open('params.pickle', 'wb'))
''' Finished '''

'''
# Code taken from
# https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
# Simply plots graph with points and best fit line
def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    return a, b

# For testing optimal lamda vals
n_hidden = 50
accuracy = np.zeros(13)
lambdavals = np.zeros(13)
for i in range(0,13):
    lambdaval = i * 5
    lambdavals[i] = lambdaval

    start = time.time()
    # set the number of nodes in hidden unit (not including bias unit)
    # n_hidden += 4
    # n_hidden_vals[i] = n_hidden
    # initialize the weights into some random matrices
    initial_W2 = initializeWeights(n_hidden, n_class)
    initial_W1 = initializeWeights(n_input, n_hidden)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_W1.flatten(), initial_W2.flatten()), 0)
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # Reshape nnParams from 1D vector into W1 and W2 matrices
    W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    print("training done!")
    end = time.time()
    complete = end - start
    # times[i] = complete

    # Test the computed parameters

    # find the accuracy on Training Dataset
    predicted_label = nnPredict(W1, W2, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    # find the accuracy on Testing Dataset
    predicted_label = nnPredict(W1, W2, test_data)
    accuracy[i] = 100 * np.mean((predicted_label == test_label).astype(float))
    print('\n Test set Accuracy:    ' + str(accuracy[i]) + '%')

# Plotting for lambdavals
a, b = best_fit(accuracy, lambdavals)
plt.plot(accuracy, lambdavals, 'ro')
yfit = [a + b * xi for xi in accuracy]
plt.plot(accuracy, yfit)
plt.ylabel('Lambdaval')
plt.xlabel('Test Set Accuracy (%)')
plt.title(filename)
plt.show()

optimal_lambdaval = lambdavals[np.argmax(accuracy)]

# For testing optimal hidden units
n_hidden = 0
lambdaval = optimal_lambdaval
n_hidden_vals = np.zeros(5)
accuracy = np.zeros(5)
times = np.zeros(5)
for i in range(0, 5):
    start = time.time()
    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden += 4
    n_hidden_vals[i] = n_hidden
    # initialize the weights into some random matrices
    initial_W2 = initializeWeights(n_hidden, n_class)
    initial_W1 = initializeWeights(n_input, n_hidden)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_W1.flatten(), initial_W2.flatten()), 0)
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # Reshape nnParams from 1D vector into W1 and W2 matrices
    W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    print("training done!")
    end = time.time()
    complete = end - start
    times[i] = complete

    # Test the computed parameters

    # find the accuracy on Training Dataset
    predicted_label = nnPredict(W1, W2, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    # find the accuracy on Testing Dataset
    predicted_label = nnPredict(W1, W2, test_data)
    accuracy[i] = 100 * np.mean((predicted_label == test_label).astype(float))
    print('\n Test set Accuracy:    ' + str(accuracy[i]) + '%')
    #print('\n Test set Accuracy:    ' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# Plotting for hidden units
a, b = best_fit(times, n_hidden_vals)
plt.plot(times, n_hidden_vals, 'ro')
yfit = [a + b * xi for xi in times]
plt.plot(times, yfit)
plt.ylabel('Hidden Units')
plt.xlabel('Training Time (seconds)')
plt.title(filename)
plt.show()

# Plotting for hidden units based on accuracy
a, b = best_fit(accuracy, n_hidden_vals)
plt.plot(accuracy, n_hidden_vals, 'ro')
yfit = [a + b * xi for xi in accuracy]
plt.plot(accuracy, yfit)
plt.ylabel('Hidden Units')
plt.xlabel('Test Set Accuracy (%)')
plt.title(filename)
plt.show()
'''
