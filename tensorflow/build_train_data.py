import re
from collections import Counter
import csv
import os

f_names = []
f_types = {}
categories = ["public", "internal", "restricted", "highly_restricted"]
for category in categories:
    directory = "train/" + category
    filenames = os.listdir(directory)
    for file in filenames:
        f_name = os.path.join(directory, file)
        f_names.append(f_name)
        if category == "public":
            f_types[f_name] = "p"
        elif category == "internal":
            f_types[f_name] = "i"
        elif category == "restricted":
            f_types[f_name] = "r"
        else:
            f_types[f_name] = "h"
        # f = open(f_name)
        # print f.read()
        # f.close()
# public = []
# internal = []
# restricted = []
# highly_restricted = []
#
# f_names= ['data2','data3','data4', 'data3']
freqs = []
def get_list(data):
    return re.compile('\w+').findall(data)

def get_data(fname):
    f = open(fname)
    data = f.read().lower()
    f.close()
    return data

words = set()

def build_set():
    for f in f_names:
        l = get_list(get_data(f))
        freqs.append(Counter(l))
        words.update(l)

def print_out_word_matrix():
    with open("trainX.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        # first_row = ['file'] + list(words)
        # writer.writerow(first_row)
        count = 0
        for f in f_names:
            print f
            row = []
            # row = [f.split("/")[-1]]
            for word in words:
                row.append(freqs[count][word])
            writer.writerow(row)
            count+=1

def print_out_file_matrix():
    with open("trainY.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        # first_row = ["category"] + categories
        # writer.writerow(first_row)
        for file in f_names:
            row = []
            # row = [file.split("/")[-1]]
            if f_types[file] == "p":
                row += [1,0,0,0]
            elif f_types[file] == "i":
                row += [0,1,0,0]
            elif f_types[file] == "r":
                row += [0,0,1,0]
            else:
                row += [0,0,0,1]
            writer.writerow(row)

build_set()
print_out_word_matrix()
print_out_file_matrix()
