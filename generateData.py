from random import randint
import numpy as np
import sys

np.set_printoptions(threshold='nan')

if (len(sys.argv) == 1) or (len(sys.argv) > 3):
    print ("Podaj zakres liczb do wygenerowania w bitach")
    sys.exit(0)

rangeForGenerator = (2 ** int(sys.argv[1]))

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
