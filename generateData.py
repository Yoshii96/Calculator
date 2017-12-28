from random import randint
import numpy as np
import sys

np.set_printoptions(threshold='nan')

if (len(sys.argv) == 1) or (len(sys.argv) > 3):
	print ("Podaj zakres liczb do wygenerowania w bitach")
	sys.exit(0)

rangeForGenerator = (2 ** int(sys.argv[1]))

file1 = open("trainData.txt", "w")
file2 = open("testData.txt", "w")
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


#size = len(np.binary_repr(2*rangeForGenerator))

#for i in range(0,len(matrix)):
#	for j in range(0,3):
#		matrix[i][j] = np.binary_repr(matrix[i][j],size)

matrix = np.asarray(matrix)

trainData = matrix[ : int(len(matrix) * 0.7)]
testData = matrix[int(len(matrix) * 0.7):]


if len(sys.argv) > 2 and str(sys.argv[2]) == "-P":
	print trainData
	print testData

file1.write(str(trainData))
file2.write(str(testData))

file1.close()
file2.close()