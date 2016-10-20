# f = open("st.txt")

l = []

# for line in f:
# 	line = line.strip().lower()
# 	line = line.replace("'",'').replace('"','')
# 	l.append(line)
# f.close()

# l = set(l)

# f = open("stop_words.txt", 'w')

# for el in l:
# 	f.write(el+'\n')
# f.close()

# f = open("temp.txt")

# for line in f:
# 	line = line.strip().lower()
# 	words =line.split("\t")
# 	for word in words:
# 		word= word.replace("'",'').replace('"','')
# 		l.append(word)
# f.close()
# # for el in l:
# # 	print el
# f = open("stop_words.txt", 'a')

# for el in l:
# 	f.write(el+'\n')
# f.close()

f = open("stop_words.txt")

for line in f:
	line = line.strip()
	l.append(line)
f.close()

l = set(l)

f = open("stopwords.txt",'w')
for el in l:
	f.write(el+"\n")
f.close()

# from nltk.corpus import stopwords

# w = stopwords.words("english")
# f = open("nltk.txt",'w')
# for word in w:
# 	f.write(word+"\n")
# f.close()