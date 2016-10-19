import numpy as np
import os
import re
from collections import Counter
import csv
from nltk.corpus import stopwords

def get_feature_matrix(filePaths, featureDict):
	featureMatrix = np.zeros(shape=(len(filePaths),len(featureDict)),dtype=float)
	# regex = re.compile("\w+")
	stop = stopwords.words("english")
	for i,filePath in enumerate(filePaths):
		with open(filePath) as f:
			raw = f.read().lower()
			# raw = re.sub(regex,'',_raw)
			# tokens = raw.split()
			# tokens = re.findall("\w+", raw)
			tokens = re.findall("[a-z]{2,}", raw)
			new_tokens = []
			for token in tokens:
				if not token in stop:
					new_tokens.append(token)
			fileUniDist = Counter(new_tokens)
			for key,value in fileUniDist.items():
			# print key, value, filePath
				if key in featureDict:
					featureMatrix[i,featureDict[key]] = value
					# print type(featureDict)
	return featureMatrix

def regularize_vectors(featureMatrix):

    for doc in range(featureMatrix.shape[0]):
        totalWords = np.sum(featureMatrix[doc,:],axis=0)
        featureMatrix[doc,:] = np.multiply(featureMatrix[doc,:],(1/totalWords))
    return featureMatrix


if __name__ == "__main__":

	test_dir = "source_files_to_test"

	filenames = os.listdir(test_dir)
	filePaths = []

	for file in filenames:
		f_name = os.path.join(test_dir, file)
		print f_name
		filePaths.append(f_name)

	f = open("bagOfWords.csv")
	words = []
	for line in f:
		line = line.strip()
		words.append(line)
	f.close()

	featureDict = {feature:i for i,feature in enumerate(words)}

	# f = open("testinput.txt")
	# testinput = []
	# for line in f:
	# 	line = line.strip()
	# 	testinput.append(line)
	# f.close()

	testX = get_feature_matrix(filePaths,featureDict)
	testX = regularize_vectors(testX)
	

	with open("to_predict.csv", "w") as output:
		writer = csv.writer(output, lineterminator='\n', delimiter = "\t")
		for el in testX:
			writer.writerow(el)