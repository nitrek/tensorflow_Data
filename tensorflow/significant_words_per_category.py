import numpy as np
import csv
from operator import itemgetter

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data(name):
    # print("loading training data")
    array = csv_to_numpy_array(name+".csv", delimiter="\t")
    return array

f = open("bagOfWords.csv")
bag = []
for line in f:
	line = line.strip()
	bag.append(line)
f.close()


trainY = import_data("trainY")
trainY = trainY.transpose()

trainX = import_data("trainX")





print trainX.shape
print trainY.shape

x = trainX.shape[1]
y = trainY.shape[0]

final_matrix = np.zeros(shape=(y,x),dtype=float)
print final_matrix.shape
row_index = 0
for row in trainY:
	el_index = 0
	# ones = []
	for el in row:
		if el ==1.0:
			# print len(final_matrix[row_index]), len(trainX[el_index])
			final_matrix[row_index]+=trainX[el_index]
			# ones.append(el_index)
		el_index+=1
	row_index+=1
	# for one in el_index:
		# s = trainX[row_index][one]



# with open("words_after_tfidf.csv", "w") as output:
# 	# first_row = bag
# 	writer = csv.writer(output, lineterminator='\n', delimiter = "\t")
# 	writer.writerow(bag)
# 	for el in final_matrix:
# 	    writer.writerow(el)

def build_dict(name):
	d = {}
	if name == "p":
		for i in range(len(bag)):
			d[bag[i]] = final_matrix[0][i]
	elif name == "i":
		for i in range(len(bag)):
			d[bag[i]] = final_matrix[1][i]
	elif name == "r":
		for i in range(len(bag)):
			d[bag[i]] = final_matrix[2][i]
	else:
		for i in range(len(bag)):
			d[bag[i]] = final_matrix[3][i]
	return d

public_dict = build_dict("p")
internal_dict = build_dict("i")
restricted_dict = build_dict("r")
highly_restricted_dict = build_dict("h_r")

p = sorted(public_dict.items(), key=itemgetter(1), reverse=True)[:20]
i = sorted(internal_dict.items(), key=itemgetter(1), reverse=True)[:20]
r = sorted(restricted_dict.items(), key=itemgetter(1), reverse=True)[:20]
h_r = sorted(highly_restricted_dict.items(), key=itemgetter(1), reverse=True)[:20]

for l in [p, i, r, h_r]:
	for el in l:
		print el
	print "\n\n------------------\n\n"
