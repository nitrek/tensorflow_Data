from nltk.corpus import stopwords
import csv
import os
import re
from operator import itemgetter


categories = ["public", "internal", "restricted", "highly_restricted"]
# categories = ["public",]

# directory = "source_files/"

def build(category):
    directory = "source_files/" + category
    filenames = os.listdir(directory)
    f_paths = []
    for file in filenames:
        f_name = os.path.join(directory, file)
        f_paths.append(f_name)
    return f_paths

def read(filePaths):
    stop = stopwords.words("english")
    freq = {}
    for filePath in filePaths:
        print filePath
        with open(filePath) as f:
            raw = f.read().lower()
            # raw = re.sub(regex,'',raw)
            #tokens = re.findall("\w+", raw)
            tokens = re.findall("[a-z]{2,}", raw)
            #tokens = raw.split()
            for token in tokens:
                if not token in stop:
                    if freq.get(token,0) ==0:
                    	freq[token] = 1
                    else:
                    	freq[token]+=1
    return freq

for category in categories:
	print category
	fPaths = build(category)
	freq = read(fPaths)
	l = sorted(freq.items(), key=itemgetter(1), reverse=True)[:10]
	for el in l:
		print el
	print "\n\n--------------------------------\n\n"