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
    # print("loading training data")
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
        if num_non_zeros == 0:
            new_row = [0.0 for el in row]
        else:
            new_row = [el * math.log(num_rows/float(num_non_zeros),10) for el in row]
        new_matrix.append(new_row)
    

def print_out(out_name):
    matrix = np.array(new_matrix).transpose()
    # matrix = matrix.transpose()
    print matrix.shape
    f = open("bagOfWords.csv")
    l = ["\t",]
    for line in f:
        line =  line.strip()
        l.append(line)
    f.close()
    f = open("trainPaths.txt")
    f_names = []
    for line in f:
        line = line.strip().split("/")[-1]
        print line
        f_names.append(line)
    f.close()
    with open(out_name+".csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter = "\t")
        writer.writerow(l)
        count = 0
        for row in matrix:
            # row = f_names[count]+row
            # print type(row)
            # np.insert(row,0,f_names[count])
            output.write(f_names[count]+"\t")
            writer.writerow(row)
            count+=1
            # print len(row)



import_data("trainX")
print_out("trainX1_headers")

# new_matrix = []

# import_data("testX")
# print_out("testX1")

# import_data("dummy")
# print_out("dummy")