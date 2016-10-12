import re
from collections import Counter
import csv
import os

f_names = []
f_types = {}
no_of_words = []

# new_str = re.sub('[^a-zA-Z\n\.]', ' ', string)  #removed special sym and numbers
# new_str = re.sub('[\n]', ' ', new_str)    #removed new line
# new_str = re.sub('[ ]+', ' ', new_str) 


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
    # print len(re.compile('\w+').findall(data))
    return re.findall(r"(?i)\b[a-z]+\b", data)
    # new_str = re.sub('[^a-z]', ' ', string)  #removed special sym and numbers
    # new_str = re.sub('[ ]+', ' ', new_str)

def get_data(fname):
    f = open(fname)
    data = f.read().lower()
    f.close()
    return data

words = set()

def build_set():
    for f in f_names:
        l = get_list(get_data(f))
        # print "from new->"+ str(len(l))
        no_of_words.append(len(l))
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
                row.append(freqs[count][word]/float(no_of_words[count]))
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
                row += [1.0,0.0,0.0,0.0]
            elif f_types[file] == "i":
                row += [0.0,1.0,0.0,0.0]
            elif f_types[file] == "r":
                row += [0.0,0.0,1.0,0.0]
            else:
                row += [0.0,0.0,0.0,1.0]
            writer.writerow(row)

build_set()
print_out_word_matrix()
print_out_file_matrix()
