from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
import time
import csv
import cv2
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


dataset = pd.read_csv('resultados.csv',header=0)
filelist = dataset['file']
numbers = dataset['number_of_ants']

model = tf.keras.models.load_model('cnn_model.h5')
model.summary()

windowlen = 128

k = 0
length = len(filelist)
outfile = open('counter.txt','w')
timefile = open('times.txt','w')
for file in filelist:
	try:
		t0 = time.perf_counter()
		img = cv2.imread('fotos/'+file)
		shape = img.shape[:-1]
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
		img = cv2.resize(img,(1024,1024))
		ants = 0

		x = 0
		buf = []
		while x<1024:
			y = 0
			while y<1024:
				subimg = img[x:x+windowlen,y:y+windowlen,:].astype(np.float32)
				buf.append(subimg)
				y+=windowlen
			x+=windowlen

		buf = np.array(buf)
		output_data = model(buf)
		predictions = np.argmax(output_data,axis=-1)

		index = 0
		for pred in predictions:
			if pred > 1 and pred <= 7:
				num = round(4.1*pred)
			elif pred > 7:
				num = round(5*pred)
			else:
				num=pred
			
			y = int((index%(1024/windowlen))*windowlen)
			x = int((index//(1024/windowlen))*windowlen)

			if pred > 0:
				ants+=num
				img[x:x+windowlen,y:y+windowlen,0] = img[x:x+windowlen,y:y+windowlen,0] * (0.9 - (0.1*pred))
				img[x:x+windowlen,y:y+windowlen,2] = img[x:x+windowlen,y:y+windowlen,2] * (0.9 - (0.1*pred))
			index += 1


		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		img = cv2.resize(img,(shape[1],shape[0]))

		font = cv2.FONT_HERSHEY_SIMPLEX
  
		# org
		org = (50, 250)
		
		# fontScale
		fontScale = 8
		
		# Blue color in BGR
		color = (0, 255, 0)
		
		# Line thickness of 2 px
		thickness = 10

		text = 'Ants: %d'%(ants)
		
		# Using cv2.putText() method
		img = cv2.putText(img, text, org, font, 
						fontScale, (0,0,0), thickness*2, cv2.LINE_AA)
		img = cv2.putText(img, text, org, font, 
						fontScale, color, thickness, cv2.LINE_AA)

		cv2.imwrite('fotos_marcadas/'+file,img)
		outfile.write(file+' %d %d\n'%(numbers[k],ants))
		interval = time.perf_counter() - t0
		timefile.write("%.4f\n"%(interval))
		outfile.flush()
		timefile.flush()
	except KeyboardInterrupt:
		print()
		print()
		break
	except:
		print('File %s skipped!'%(file))

	k+=1
	print("%d/%d"%(k,length),end='\r')
print()
timefile.close()
outfile.close()