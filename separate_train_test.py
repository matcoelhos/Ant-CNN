import csv
from random import shuffle,seed

genlist = []

with open("dataset.csv", "r") as f:
	reader = csv.reader(f)
	i = 0
	for line in reader:
		if i > 0:
			genlist.append(line)
		else:
			header = line
		i+=1

f.close()

seed()
for i in range(4):
	shuffle(genlist)
size = len(genlist)
i = 0

markings = open('test.csv','w', encoding="utf-8")
writer = csv.writer(markings)
writer.writerow(header)

j = 0
while j < int(0.1*size):
	writer.writerow(genlist[i])
	i+=1
	j+=1

markings.close()

markings = open('validation.csv','w', encoding="utf-8")
writer = csv.writer(markings)
writer.writerow(header)

j = 0
while j < int(0.1*size):
	writer.writerow(genlist[i])
	i+=1
	j+=1

markings.close()

markings = open('train.csv','w', encoding="utf-8")
writer = csv.writer(markings)
writer.writerow(header)

while i < size:
	writer.writerow(genlist[i])
	i+=1

markings.close()