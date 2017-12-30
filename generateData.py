from random import randint
import numpy as np
import argparse

np.set_printoptions(threshold='nan')

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

    matrix = [[0 for x in range(3)] for y in range(int(rangeForGenerator * rangeForGenerator))]

    for i in range(rangeForGenerator * rangeForGenerator):
        matrix[i][0] = i / rangeForGenerator
        matrix[i][1] = i % rangeForGenerator
        matrix[i][2] = matrix[i][0] + matrix[i][1]

    matrix = np.asarray(matrix)
    np.random.shuffle(matrix)

    trainData = matrix[ : int(len(matrix) * 0.5)]
    testData = matrix[int(len(matrix) * 0.5):int(len(matrix) * 0.7)]

    with open("trainData.txt", "w") as file1:
        file1.write(str(args.number_of_bits))
        file1.write(str(trainData))

    with open("testData.txt", "w") as file2:
        file2.write(str(args.number_of_bits))
        file2.write(str(testData))
