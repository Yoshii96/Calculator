from random import randint
import numpy as np
import sys
import argparse

#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TestCalculator')
    parser.add_argument('--number_of_bits',
                        type=int,
                        required=True,
                        help='Range in bits.')
    
    args = parser.parse_args()
    
    number_of_bits = args.number_of_bits

    print ("Range in bits: " + str(number_of_bits))

    rangeForGenerator = (2 ** number_of_bits)

    list1 = [[False for x in range(0,rangeForGenerator)] for y in range(0,rangeForGenerator)]
    counter = 0
    matrix = [[0 for x in range(3)] for y in range(int(0.7/1.0 * rangeForGenerator * rangeForGenerator))]

    while counter < int(0.7/1.0 * rangeForGenerator * rangeForGenerator):
        i = randint(0,rangeForGenerator -1)
        j = randint(0,rangeForGenerator -1)
        if (list1[i][j] == False):
            list1[i][j] = True
            matrix[counter][0] = i
            matrix[counter][1] = j
            matrix[counter][2] = i + j
            counter = counter + 1

    matrix = np.asarray(matrix)

    trainData = matrix[ : int(len(matrix) * 0.7)]
    testData = matrix[int(len(matrix) * 0.7):]


    if len(sys.argv) > 2 and str(sys.argv[2]) == "-P":
        print trainData
        print testData


    with open("trainData.txt", "w") as file1:
        file1.write(sys.argv[1])
        file1.write(str(trainData))

    with open("testData.txt", "w") as file2:
        file2.write(sys.argv[1])
        file2.write(str(testData))
