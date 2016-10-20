# conda install nltl
# nltk.download()    -> inside python
import nltk
import os
import re
import itertools
from collections import Counter
from nltk.stem import WordNetLemmatizer


l = WordNetLemmatizer()

#categories = ["public", "internal", "restricted", "highly_restricted"]
categories = ["test"]

def lemmatizing():
    for category in categories:
	#directory = "../tensorflow/source_files/"+category
        directory = "test"
        filenames = os.listdir(directory)
        for file in filenames:
			f_name = os.path.join(directory,file)
			new_data=""
			f = open(f_name,'r')
			data = f.read().lower().split()
            		for words in data:
            			words = l.lemmatize(words,pos='v')				
				words = l.lemmatize(words1,pos='n')
				print(words)
				new_data = new_data + " " + words2 + "\n"
			print(new_data)
			f.close()
			f = open(f_name,'w')
			f.write(new_data)
			f.close
			
				

lemmatizing()
