# f = open("bagOfWords.csv")
# bag = []
# for line in f:
# 	line = line.strip()
# 	bag.append(line)
# f.close()


f = open("testX.csv")
l = []
for line in f:
	line = line.strip()
	line = line.replace("\t",",")
	l.append(line)
f.close()

f = open("testX_new.csv","w")
#f.write(",".join(bag))
#f.write("\n")
# print ",".join(bag)
for el in l:
	f.write(el+"\n")
f.close()