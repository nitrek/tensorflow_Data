import numpy as np
import math
import csv
# f = open("trainX.csv")

new_matrix = []

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def column(matrix, i):
    return [row[i] for row in matrix]

def import_data(in_name):
    print("loading training data")
    trainX = csv_to_numpy_array(in_name+".csv", delimiter="\t")
    # trainY = csv_to_numpy_array("trainY.csv", delimiter=",")
    # print("loading test data")
    # testX = csv_to_numpy_array("testX.csv", delimiter=",")
    # testY = csv_to_numpy_array("testY.csv", delimiter=",")
    # print type(trainX)
    # print "\n\n----\n\n"
    transposed = trainX.transpose()


    for row in transposed:
        # print row
        num_rows = len(row)
        num_non_zeros =  np.count_nonzero(row)
        new_row = [el * math.log(num_rows/float(num_non_zeros),10) for el in row]
        new_matrix.append(new_row)
    

def print_out(out_name):
    matrix = np.array(new_matrix).transpose()
    # matrix = matrix.transpose()
    with open(out_name+".csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter = "\t")
        for row in matrix:
            writer.writerow(row)
            # print len(row)



import_data("trainX")
print_out("trainX1")

import_data("testX")
print_out("testX1")

# import_data("dummy")
# print_out("dummy")