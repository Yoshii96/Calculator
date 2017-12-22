import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import pickle


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
	return 1/(1 + np.exp(-1 * Z)),Z

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
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],activation = "relu")
        caches.append(cache)
    A_prev = A
    AL, cache = linear_activation_forward(A_prev, parameters["W"+str(int(len(parameters)/2))], parameters["b"+str(int(len(parameters)/2))],activation = "sigmoid")
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
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
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


def get_train_data(size):
    with open("trainData.txt", "r") as fileTrainData:
        trainData = fileTrainData.read()
        trainData = trainData[1:-1]
        trainData = trainData.split("\n")
        trainDataX = np.zeros((((size+1)*2),len(trainData)))
        trainDataY = np.zeros((size+1,len(trainData)))
        for i in range(len(trainData)):
            trainData[i] = trainData[i].replace("[", "").replace("]","").replace("'","").strip()
            trainData[i] = re.sub(' +', ' ',trainData[i])
            trainData[i] = trainData[i].split(" ")
            for j in range(len(trainData[i])):
                trainData[i][j] = np.binary_repr(int(trainData[i][j]),size+1)
            tmpstring = str(trainData[i][0]) + str(trainData[i][1])
            for j in range((size+1)*2):
        		if tmpstring[j] == '1':
        			trainDataX[j][i] = 1
        		else:
        			trainDataX[j][i] = 0
            tmpstring = str(trainData[i][2])
            for j in range(size+1):
                if tmpstring[j] == '1':
        			trainDataY[j][i] = 1
                else:
        			trainDataY[j][i] = 0    			
    return trainDataX,trainDataY

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



#main
#main
hidenLayers = [36]
learning_rate = 0.07
number_of_iterations = 20000
size = 8 #TO DO zrobic, zeby nie bylo 5 tylko rozmiar w bitach
#dodajemy liczby 4 bit wiec wynik mze byc 5
structure = [(size+1)*2] + hidenLayers + [size+1]
#print ("structure = ", structure)
par = init_parameters(structure)
pars = {}
pars["0"] = par
costs = np.zeros(number_of_iterations)
#print ("W1 = ", par["W1"].shape)
#print ("b1 = ", par["b1"].shape)
#print ("W2 = ", par["W2"].shape)
#print ("b2 = ", par["b2"].shape)
trainDataX,trainDataY = get_train_data(size)
#print trainDataX.shape
#trainDataY = trainDataY[0:2,:]
for i in range(number_of_iterations):
    AL, caches = L_model_forward(trainDataX,par)
    AL[AL == 0] = 0.000000000001
    AL[AL == 1] = 0.999999999999
    cost = compute_cost(AL, trainDataY)
    grads = L_model_backward(AL, trainDataY,caches)
    par = update_parameters(par, grads, learning_rate)
    pars[str(i+1)] = par
    costs[i] = cost
# test
save_obj(par, "parameters")
iteration = np.arange(0, number_of_iterations,1)
plt.plot(iteration,costs)
plt.savefig("test.png")
plt.show