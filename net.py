import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import argparse


def init_parameters(structure):
    wArraySize = len(structure)
    parameters = {}
    for i in range(1,wArraySize):
        parameters['W'+str(i)] = np.random.random((structure[i],structure[i-1])) * 0.01
        parameters['b'+str(i)] = np.zeros((structure[i],1))
    return parameters

def simpleRound(Z):
    return np.around(Z),Z

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-1.0 * Z)),Z

def relu(Z):
    return np.maximum(0,Z),Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def simpleRound_backword(dA, cache):
    Z = cache
    return Z 

def forward_propagation(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = forward_propagation(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = forward_propagation(A_prev,W,b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(
            A_prev,
            parameters["W"+str(l)],
            parameters["b"+str(l)],
            activation = "relu")
        caches.append(cache)
    A_prev = A
    AL, cache = linear_activation_forward(
        A_prev,
        parameters["W"+str(int(len(parameters)/2))],
        parameters["b"+str(int(len(parameters)/2))],
        activation = "sigmoid")
    caches.append(cache)       
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1.0/m*np.sum(Y*np.log(AL)+(1.0-Y)*np.log(1.0-AL))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1.0/m*np.dot(dZ,np.transpose(A_prev))
    db = 1.0/m*np.sum(dZ,axis=1,keepdims = True)
    dA_prev = np.dot(W.T,dZ)    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)   
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL = - (np.divide(Y, AL) - np.divide(1.0 - Y, 1.0 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
        dAL,
        current_cache,
        "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l+2)],
            current_cache,
            "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) / 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate* grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate* grads["db" + str(l+1)]
    return parameters


def get_data(file):
    with open(file, "r") as fileData:
        Data = fileData.read()
        index = Data.find("[")
        tmpstring = Data[0:index]
        size = int(tmpstring)
        Data = Data[index+1:-1]
        Data = Data.split("\n")
        DataX = np.zeros((((size+1)*2),len(Data)))
        DataY = np.zeros((size+1,len(Data)))
        for i in range(len(Data)):
            Data[i] = Data[i].replace("[", "").replace("]","").replace("'","").strip()
            Data[i] = re.sub(' +', ' ',Data[i])
            Data[i] = Data[i].split(" ")
            for j in range(len(Data[i])):
                Data[i][j] = np.binary_repr(int(Data[i][j]),size+1)
            tmpstring = str(Data[i][0]) + str(Data[i][1])
            for j in range((size+1)*2):
                DataX[j][i] = int(tmpstring[j])
            tmpstring = str(Data[i][2])
            for j in range(size+1):
                DataY[j][i] = int(tmpstring[j])
    return DataX,DataY,size

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculator')
    parser.add_argument('--train_data',
                        type=str,
                        required=True,
                        help='Data, which will be used to train net.')
    parser.add_argument('--test_data',
                        type=str,
                        required=True,
                        help='Data, which will be used to test net.')
    parser.add_argument('--hiden_layers',
                        nargs='*',
                        type=int,
                        default=[20],
                        help='Structure of hiden layers.')
    parser.add_argument('--net_file',
                        type=str,
                        default='new_net',
                        help='File, in which net will be stored.')
    parser.add_argument('--num_of_iterations',
                        type=int,
                        default=10000,
                        help='Number of iterations, that will be performed on training set.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.05,
                        help='Learning rate says, how fast parameters will be changing.')
    parser.add_argument('--plot_file',
                        type=str,
                        default='new_plot',
                        help='File, in which plot will be stored.')

    args = parser.parse_args()
    
    train_data_str = args.train_data
    test_data_str = args.test_data
    hiden_layers = args.hiden_layers
    net_file = args.net_file
    number_of_iterations = args.num_of_iterations
    learning_rate = args.learning_rate
    plot_file = args.plot_file


    print ("Training set: " + train_data_str)
    print ("Test set: " + test_data_str)
    print ("Structure of hiden layers: " + str(hiden_layers))
    print ("File to save net: " + net_file)
    print ("Number of iterations: " + str(number_of_iterations))
    print ("Learning rate: " + str(learning_rate))
    print ("File to save plot: " + plot_file)

    trainDataX,trainDataY,size = get_data(train_data_str)
    testDataX, testDataY, size = get_data(test_data_str)
    structure = [(size+1)*2] + hiden_layers + [size+1]

    steps = 100
    spaces = number_of_iterations / steps
    
    par = init_parameters(structure)
    pars = {}
    pars["0"] = par
    costs = np.zeros(number_of_iterations)
    test_cost = np.zeros(steps)

    for i in range(number_of_iterations):
        AL, caches = L_model_forward(trainDataX,par)
        AL[AL == 0] = 0.0 + 1000.0/sys.maxsize
        AL[AL == 1] = 1.0 - 1000.0/sys.maxsize
        cost = compute_cost(AL, trainDataY)
        grads = L_model_backward(AL, trainDataY,caches)
        par = update_parameters(par, grads, learning_rate)
        pars[str(i+1)] = par
        costs[i] = cost
        if i % spaces == 0:
            AL, caches = L_model_forward(testDataX,par)
            AL[AL == 0] = 0.0 + 1000.0/sys.maxsize
            AL[AL == 1] = 1.0 - 1000.0/sys.maxsize
            cost = compute_cost(AL, testDataY)
            test_cost[i / spaces] = cost
            print (str(i/spaces) + "% completed!")
    
    save_obj(par, net_file)
    iteration = np.arange(0, number_of_iterations,1)
    test_iteration = np.arange(0, steps, 1)
    figure, ax = plt.subplots()
    ax.plot(iteration, costs, 'b', label='Train Set Cost')
    ax.plot(test_iteration * spaces, test_cost, 'r', label='Test Set Cost')
    legend = ax.legend()
    plt.savefig(plot_file)
    print ("100% completed!")