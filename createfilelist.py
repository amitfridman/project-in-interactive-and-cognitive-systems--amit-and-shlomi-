import os
filefolder='/home/liel/Downloads/ILSVRC/Data/CLS-LOC/val/'
files=os.listdir(filefolder)
with open('val_list.txt','w') as f:
	for i,fi in enumerate(files):
		if i<5040:
			f.write(fi+' '+'-2'+'\n')
		else:
			break
f.close
