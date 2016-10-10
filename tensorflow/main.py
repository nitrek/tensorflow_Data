import re
from collections import Counter
import csv
import os

fnames = []
categories = ["public", "internal", "restricted", "highly_restricted"]
for category in categories:
    directory = category
    filenames = os.listdir(directory)
    for file in filenames:
        f_name=os.path.join(directory, file)
        fnames.append(f_name)
        # f = open(f_name)
        # print f.read()
        # f.close()
# public = []
# internal = []
# restricted = []
# highly_restricted = []
#
# fnames= ['data2','data3','data4', 'data3']
freqs = []
def get_list(data):
    return re.compile('\w+').findall(data)

def get_data(fname):
    f = open(fname)
    data = f.read()
    f.close()
    return data

words = set()

def build_set():
    for f in fnames:
        l = get_list(get_data(f))
        freqs.append(Counter(l))
        words.update(l)



def print_out():
    with open("out.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')


        first_row = ['file'] + list(words)
        writer.writerow(first_row)
        count = 0
        for f in fnames:
            row = [f]
            for word in words:
                row.append(freqs[count][word])
            writer.writerow(row)
            count+=1


build_set()
print_out()

