import argparse
import numpy as np
from net import load_obj, L_model_forward, sigmoid, relu, forward_propagation, linear_activation_forward

#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculator')
    parser.add_argument('--net_file',
                        type=str,
                        required=True,
                        help='File, which contains net.')

    args = parser.parse_args()
    
    net_file = args.net_file

    print ('File with net: ' + net_file)

    par = load_obj(net_file)
    tmp_shape1, shape2 = par["W" + str(1)].shape
    shape2 = shape2/2 - 1
    range_int = 2**shape2 - 1
    size = shape2

    print ('Range in bits: ' + str(shape2))

    while True:
        first = int(input('Give first number between 0 and ' + str(range_int) + ':   '))
        if first >= 0 and first <= range_int:
            break

    while True:
        second = int(input('Give second number between 0 and ' + str(range_int) + ':  '))
        if second >= 0 and second <= range_int:
            break    

    answer = first + second
    print (str(first) + " + " + str(second) + " = " + str(answer))

    bin_first = np.binary_repr(first,size+1)
    bin_second = np.binary_repr(second,size+1)
    bin_answer = np.binary_repr(answer,size+1)

    print (str(bin_first) + " + " + str(bin_second) + " = " + str(bin_answer))

    DataX = np.zeros(((size+1)*2,1)) 
    tmp_string = str(bin_first) + str(bin_second)
    for i in range(len(tmp_string)):
        DataX[i] = int(tmp_string[i])

    DataY = np.zeros((size+1, 1))
    tmp_string = str(bin_answer)
    for i in range(len(tmp_string)):
        DataY[i] = int(tmp_string[i])

    AL, cathce = L_model_forward(DataX,par)
    
    print ('Output given by net: ' + str(AL))
    AL = np.around(np.transpose(AL))
    error = np.absolute(AL - np.transpose(DataY))
    tmp_string = str(AL)
    tmp_string = tmp_string.replace('[','').replace(']','').replace(' ','').replace('.','')
    print (tmp_string + ' = NET ANSWER')
    tmp_string = str(np.transpose(DataY))
    tmp_string = tmp_string.replace('[','').replace(']','').replace(' ','').replace('.','')
    print (tmp_string + ' = RIGHT ANSER')
    tmp_string = str(error)
    tmp_string = tmp_string.replace('[','').replace(']','').replace(' ','').replace('.','').replace('1','X')
    print (tmp_string + ' = ERROR MARKS')