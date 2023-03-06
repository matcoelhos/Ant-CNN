import cv2
import csv
import pandas as pd
import os
from random import seed, random
import imutils
from skimage.util import random_noise
import numpy as np

def gcd(a, b):
    while b != 0:
        t = b
        b = a % b
        a = t
    return a

def mydiv(a,b):
	div = 1
	while (a%2 == 0 and b%2 == 0):
		a /= 2
		b /= 2
		div *= 2
	return div

def in_square(points,y0,x0,y,x):
	n = 0
	for point in points:
		x1 = point[0]
		y1 = point[1]
		if ((x1 >= x0 and x1 <= x) and (y1 >= y0 and y1 <= y)):
			n+=1
	return n

# dataset = pd.read_csv('resultados.csv',header=0)

# folder = 'fotos/'

dataset = pd.read_csv('synth_dataset.csv',header=0)

folder = 'sinteticas/'

imagelist = dataset['file']
pointlist = dataset['positions']

outfile = open('dataset.csv','w')
writer = csv.writer(outfile)

writer.writerow(['file','num_ants'])

seed()

i = 0
nimages = 1
for image in imagelist:
	points = eval(pointlist[i])
	try:
		img = cv2.imread(folder+image)

		# shape = (640,480)
		shape = (1200,1200)
		img = cv2.resize(img,(1024,1024))

		for j in range(len(points)):
			points[j] = (int((points[j][0]/shape[0])*1024), int((points[j][1]/shape[1])*1024))

		nimg = img.copy()

		for point in points:
			cv2.circle(nimg, point, 2, (0,0,255),2)

		# cv2.imshow('Imagem', nimg)
		cv2.imwrite('marcadas/'+image,nimg)

		x = 0
		squaresize = 128
		while x < img.shape[0]:
			y = 0
			while y < img.shape[1]:
				subimg = img[x:x+squaresize,y:y+squaresize,:]
				nsubimg = nimg[x:x+squaresize,y:y+squaresize,:]
				numpoints = in_square(points,x,y,x+squaresize,y+squaresize)
				if numpoints>0:
					if numpoints<5:
						numpoints=1
					elif numpoints<10:
						numpoints=2
					elif numpoints<15:
						numpoints=3
					elif numpoints<20:
						numpoints=4
					elif numpoints<25:
						numpoints=5
					elif numpoints<30:
						numpoints=6
					elif numpoints<35:
						numpoints=7
					elif numpoints<40:
						numpoints=8
					else:
						numpoints=9

					cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(subimg,(128,128)))
					writer.writerow(['img_%08d.png'%(nimages),numpoints])
					nimages+=1
					# if numpoints >= 1:
					# 	for u in range(3):
					# 		nsubimg = imutils.rotate_bound(subimg, 45+45*u)
					# 		cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(nsubimg,(128,128)))
					# 		writer.writerow(['img_%08d.png'%(nimages),numpoints])
					# 		nimages+=1
					# 		for v in range(1):
					# 			nsubimg_i = 255*random_noise(nsubimg, mode='s&p',amount=0.1)
					# 			cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(nsubimg_i,(128,128)))
					# 			writer.writerow(['img_%08d.png'%(nimages),numpoints])
					# 			nimages+=1


				else:	
					cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(subimg,(128,128)))
					writer.writerow(['img_%08d.png'%(nimages),0])
					nimages+=1
					# for u in range(3):
					# 	nsubimg = imutils.rotate_bound(subimg, 90+90*u)
					# 	cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(nsubimg,(128,128)))
					# 	writer.writerow(['img_%08d.png'%(nimages),numpoints])
					# 	nimages+=1
				y+=squaresize
			x+=squaresize
		print(image,nimages)
	except KeyboardInterrupt:
		break
	except:
		print(image,'skipped!')
		continue
	i+=1

dataset = pd.read_csv('resultados.csv',header=0)

folder = 'fotos/'

# dataset = pd.read_csv('synth_dataset.csv',header=0)

# folder = 'sinteticas/'

imagelist = dataset['file']
pointlist = dataset['positions']

seed()
i = 0
for image in imagelist:
	points = eval(pointlist[i])
	try:
		img = cv2.imread(folder+image)

		shape = (640,480)
		# shape = (1200,1200)
		img = cv2.resize(img,(1024,1024))

		for j in range(len(points)):
			points[j] = (int((points[j][0]/shape[0])*1024), int((points[j][1]/shape[1])*1024))

		nimg = img.copy()

		for point in points:
			cv2.circle(nimg, point, 2, (0,0,255),2)

		# cv2.imshow('Imagem', nimg)
		cv2.imwrite('marcadas/'+image,nimg)

		x = 0
		squaresize = 128
		while x < img.shape[0]:
			y = 0
			while y < img.shape[1]:
				subimg = img[x:x+squaresize,y:y+squaresize,:]
				nsubimg = nimg[x:x+squaresize,y:y+squaresize,:]
				numpoints = in_square(points,x,y,x+squaresize,y+squaresize)
				if numpoints>0:
					if numpoints<5:
						numpoints=1
					elif numpoints<10:
						numpoints=2
					elif numpoints<15:
						numpoints=3
					elif numpoints<20:
						numpoints=4
					elif numpoints<25:
						numpoints=5
					elif numpoints<30:
						numpoints=6
					elif numpoints<35:
						numpoints=7
					elif numpoints<40:
						numpoints=8
					else:
						numpoints=9

					cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(subimg,(128,128)))
					writer.writerow(['img_%08d.png'%(nimages),numpoints])
					nimages+=1
					# if numpoints >= 1:
					# 	for u in range(3):
					# 		nsubimg = imutils.rotate_bound(subimg, 45+45*u)
					# 		cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(nsubimg,(128,128)))
					# 		writer.writerow(['img_%08d.png'%(nimages),numpoints])
					# 		nimages+=1
					# 		for v in range(1):
					# 			nsubimg_i = 255*random_noise(nsubimg, mode='s&p',amount=0.1)
					# 			cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(nsubimg_i,(128,128)))
					# 			writer.writerow(['img_%08d.png'%(nimages),numpoints])
					# 			nimages+=1


				else:	
					cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(subimg,(128,128)))
					writer.writerow(['img_%08d.png'%(nimages),0])
					nimages+=1
					# for u in range(3):
					# 	nsubimg = imutils.rotate_bound(subimg, 90+90*u)
					# 	cv2.imwrite('dataset/img_%08d.png'%(nimages),cv2.resize(nsubimg,(128,128)))
					# 	writer.writerow(['img_%08d.png'%(nimages),numpoints])
					# 	nimages+=1
				y+=squaresize
			x+=squaresize
		print(image,nimages)
	except KeyboardInterrupt:
		break
	except:
		print(image,'skipped!')
		continue
	i+=1