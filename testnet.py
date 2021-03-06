import pickle
import re
import sys
import numpy as np
import argparse
from net import get_data, sigmoid, relu, forward_propagation, linear_activation_forward, L_model_forward, load_obj



#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TestCalculator')
    parser.add_argument('--test_data',
                        type=str,
                        required=True,
                        help='Data, which will be used to test net.')
    parser.add_argument('--net_file',
                        type=str,
                        required=True,
                        help='File, which contains net.')

    args = parser.parse_args()
    
    test_data_str = args.test_data
    net_file = args.net_file

    print ("Test set: " + test_data_str)
    print ("File with net: " + net_file)

    par = load_obj(net_file)
    testDataX,testDataY,size = get_data(test_data_str)

    tmp_shape1, shape2 = par["W" + str(1)].shape
    shape1, tmp_shape2 = par["W" + str(len(par)/2)].shape
    if shape1 != size + 1 or shape2 != (size + 1) * 2:
        print ('Structure of net do not match with data!')
        sys.exit(0)

    AL,tmpcache = L_model_forward(testDataX,par)
    AL = np.around(AL)
    AL = np.absolute(AL - testDataY)
    #print ("Errors in AL =  " + str(AL))
    print ("Sum of errors =  " + str(np.sum(AL)))
    print ("Accuracy = " + str((AL.shape[0] * AL.shape[1] - np.sum(AL)) * 100 / (AL.shape[0] * AL.shape[1])) + "%")