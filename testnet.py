import pickle
import re
import sys
import numpy as np

def get_test_data(file):
    with open(file, "r") as fileTestData:
        testData = fileTestData.read()
        index = testData.find("[")
        tmpstring = testData[0:index]
        size = int(tmpstring)
        testData = testData[index+1:-1]
        testData = testData.split("\n")
        testDataX = np.zeros((((size + 1)*2),len(testData)))
        testDataY = np.zeros((size+1,len(testData)))
        for i in range(len(testData)):
            testData[i] = testData[i].replace("[", "").replace("]","").replace("'","").strip()
            testData[i] = re.sub(' +', ' ',testData[i])
            testData[i] = testData[i].split(" ")
            for j in range(len(testData[i])):
                testData[i][j] = np.binary_repr(int(testData[i][j]),size+1)
            tmpstring = str(testData[i][0]) + str(testData[i][1])
            for j in range((size + 1)*2):
                testDataX[j][i] = int(tmpstring[j])
            tmpstring = str(testData[i][2])
            for j in range(size + 1):
                testDataY[j][i] = int(tmpstring[j])
    return testDataX,testDataY


def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def sigmoid(Z):
    return 1/(1 + np.exp(-1 * Z))


def relu(Z):
    return np.maximum(0,Z)

def forward_propagation(A,W,b):
    return np.dot(W,A) + b


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = forward_propagation(A_prev,W,b)
        A = sigmoid(Z)
    elif activation == "relu":
        Z = forward_propagation(A_prev,W,b)
        A = relu(Z)
    return A

def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2                    
    for l in range(1, L):
        A_prev = A 
        A = linear_activation_forward(
            A_prev,
            parameters["W"+str(l)],
            parameters["b"+str(l)],
            activation = "relu")
    A_prev = A
    AL = linear_activation_forward(
        A_prev,
        parameters["W"+str(int(len(parameters)/2))],
        parameters["b"+str(int(len(parameters)/2))],
        activation = "sigmoid")
    return AL


#main
if (len(sys.argv) == 1) or (len(sys.argv) > 3):
    print ("Podaj nazwe pliku do testow!")
    sys.exit(0)
par = load_obj("parameters")
print par
testDataX,testDataY = get_test_data(sys.argv[1])
AL = L_model_forward(testDataX,par)
print ("AL  =  " + str(AL))
AL = np.around(AL)
print ("AL after round =  " + str(AL))
AL = np.absolute(AL - testDataY)
print ("AL after absolute =  " + str(AL))
print ("Sum of AL =  " + str(np.sum(AL)))
print ("Accuracy = " + str((AL.shape[0] * AL.shape[1] - np.sum(AL)) * 100 / (AL.shape[0] * AL.shape[1])) + "%")

