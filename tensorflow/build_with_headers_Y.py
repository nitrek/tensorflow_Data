data = []

f = open("trainY.csv")
for line in f:
	line = line.strip().split("\t")
	# print line
	data.append(line)
f.close()

f = open("trainPaths.txt")
f_names = []
for line in f:
    line = line.strip().split("/")[-1]
    # print line
    f_names.append(line)
f.close()

f = open("trainY_with_headers.csv","w")
f.write("\tpublic\tinternal\trestricted\thighly_restricted\n")
for i in range(len(f_names)):
	row = data[i]
	row.insert(0,f_names[i])
	for el in row:
		f.write(el+"\t")
	f.write("\n")
f.close()