import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms
import torchvision.models as models
import pickle 

imagefolder='/home/liel/lrp_toolbox/lrp_output/val/'
files=os.listdir(imagefolder)
channels=3
h=224
width=224
imagenum=0
lrp_vals={}
images={}
lrp_vals_all={}
for f in files:
	if 'as' in f:

		images[f.split('_as')[0]]=f
		print(f)
	if 'rawhm' in f:
		with open(imagefolder+f,'r') as img:
			imagename=f.split('_raw')[0]
			#print(imagename)
			for i,line in enumerate(img):
				if i>=2:
					l=line.split(' ')[:-1]
					if i-imagenum*h-2>=h:
						imagenum+=1
					for j,val in enumerate(l):
						key=str(imagenum)+','+str(i-imagenum*h-2)+','+str(j)
						if key not in lrp_vals:
							lrp_vals[key]=float(val)
						else:
							lrp_vals[key]+=float(val)
	
				elif  i==0:
					channels=int(line.rstrip('\n'))
				else:
					h,w=line.split(' ')[:2]
					h=int(h)
					w=int(w)
		
			lrp_vals_sorted=sorted(lrp_vals.items(),key=lambda item: item[1],reverse=True)[:50]
			lrp_vals_all[imagename]=lrp_vals_sorted
			lrp_vals={}
		img.close()
		#print(lrp_vals_sorted[:9])
		imagenum=0

label_file='/home/liel/lrp_toolbox/caffe-master-lrp/data/ilsvrc12/val.txt'
labels={}
with open(label_file,'r') as l:
	for line in l:
		img=line.split(' ')
		labels[img[0]]=int(img[1])

googlenet = models.googlenet(pretrained=True)
acc=0
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
googlenet.eval()
original_score=0
scores=[]
for j in range(25):
	acc=0
	for i,img in enumerate(images):
		imgs=Image.open(imagefolder+images[img])
		
		trans=transforms.ToTensor()
		imgs=trans(imgs)
		imgs=normalize(imgs)
		for k in range(j):
			#print(k)
			loc=lrp_vals_all[img][k][0].split(',')
			color=int(loc[0])
			loc=loc[1:]
			
			top,right,left,bottom=(int(loc[0])+4,int(loc[1])+4,int(loc[1])-4,int(loc[0])-4)
			if bottom<0:
				bottom=0
			if right>223:
				right=223
			if left<0:
				left=0
			if top>223:
				top=223
			imgs[:,bottom:top+1,left:right+1]=torch.randint(low=0,high=256,size=(3,top+1-bottom,right+1-left))
		
		
		
		
		imgs=torch.unsqueeze(imgs, 0)
		#print(imgs.size())

		outputs=googlenet(imgs)
		#print(torch.sort(outputs, descending=True)[:10])
		_,indexs = torch.sort(outputs,dim=1,descending=True)
		indexs=indexs[0,:5]
		#print(index,labels[img])
		if i==0:
			print(indexs)
		if torch.tensor(labels[img]) in indexs:
			acc+=1
	if j==0:
		original_score=acc/len(list(images.keys()))
		print(original_score)
	score=acc/len(list(images.keys()))
	scores.append(score/original_score)
	print(score,score/original_score)
pickle.dump(scores,open('scores_lrp_9x9_1.pkl','wb'))
	
	
					
					
					

