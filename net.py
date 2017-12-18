import numpy as np
import re
import sys

def init_parameters(structure):
	wArraySize = len(structure)
	parameters = {}
	for i in range(1,wArraySize):
		parameters['W'+str(i)] = np.random.random((structure[i],structure[i-1])) * 0.01
		parameters['b'+str(i)] = np.zeros((structure[i],1))
	return parameters

def sigmoid(Z):
	return 1/(1 + np.exp(-1 * Z)),Z

def relu(Z):
	return np.maximum(0,Z),Z


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
    #print (parameters["W"+str(int(len(parameters)/2))].shape)
    #print (parameters["b"+str(int(len(parameters)/2))].shape)
    #print (str(int(len(parameters)/2)))
    AL, cache = linear_activation_forward(A_prev, parameters["W"+str(int(len(parameters)/2))], parameters["b"+str(int(len(parameters)/2))],activation = "sigmoid")
    caches.append(cache)       
    return AL, caches

def compute_cost(AL, Y):
	print (AL.shape)
	m = Y.shape[0]#zmiana z 1 na 0
	cost = -1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
	cost = np.squeeze(cost)   
	return cos

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,np.transpose(A_prev))
    db = 1/m*np.sum(dZ,axis=1,keepdims = True)
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
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
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
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate* grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate* grads["db" + str(l+1)]
    return parameters


#main
hidenLayers = [10]
learning_rate = 0.01
size = 5 #TO DO zrobic, zeby nie bylo 5 tylko rozmiar w bitach
#dodajemy liczby 4 bit wiec wynik mze byc 5


with open("trainData.txt", "r") as fileTrainData:
    trainData = fileTrainData.read()
    trainData = trainData[1:-1]
    trainData = trainData.split("\n")
    trainDataX = np.zeros((len(trainData),size*2))
    trainDataY = np.zeros((len(trainData),size))
    for i in range(len(trainData)):
    	trainData[i] = trainData[i].replace("[", "").replace("]","").replace("'","").strip()
    	trainData[i] = re.sub(' +', ' ',trainData[i])
    	trainData[i] = trainData[i].split(" ")
    	for j in range(len(trainData[i])):
    		trainData[i][j] = np.binary_repr(int(trainData[i][j]),size)
    	tmpstring = str(trainData[i][0]) + str(trainData[i][1])
    	for j in range(size*2):
    		if tmpstring[j] == '1':
    			trainDataX[i][j] = 1
    		else:
    			trainDataX[i][j] = 0

    	tmpstring = str(trainData[i][2])
    	for j in range(size):
    		if tmpstring[j] == '1':
    			trainDataY[i][j] = 1
    		else:
    			trainDataY[i][j] = 0    			

print trainDataX[0].shape
trainDataX[0] = trainDataX[0].reshape((10,1))



with open("testData.txt", "r") as fileTestData:
    testData = fileTestData.read()
    testData = testData[1:-1]
    testData = testData.split("\n")
    testDataX = np.zeros((len(testData),size*2))
    testDataY = np.zeros((len(testData),size))
    for i in range(len(testData)):
    	testData[i] = testData[i].replace("[", "").replace("]","").replace("'","").strip()
    	testData[i] = re.sub(' +', ' ',testData[i])
    	testData[i] = testData[i].split(" ")
    	for j in range(len(testData[i])):
    		testData[i][j] = np.binary_repr(int(testData[i][j]),size)
    	tmpstring = str(testData[i][0]) + str(testData[i][1])
    	for j in range(size*2):
    		if tmpstring[j] == '1':
    			testDataX[i][j] = 1
    		else:
    			testDataX[i][j] = 0

    	tmpstring = str(testData[i][2])
    	for j in range(size):
    		if tmpstring[j] == '1':
    			testDataY[i][j] = 1
    		else:
    			testDataY[i][j] = 0

structure = [size*2] + hidenLayers + [size]
par = init_parameters(structure)
prev_par = par
for i in range(len(trainDataX)):
	AL, caches = L_model_forward(trainDataX[i],par)
	cost = compute_cost(AL, testDataY[i])
	grads = L_model_backward(AL, testDataY[i],caches)
	par = update_parameters(par, grads, learning_rate)


# test