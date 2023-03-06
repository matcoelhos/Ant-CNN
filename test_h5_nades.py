from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

model = tf.keras.models.load_model('cnn_model.h5')
model.summary()

test_datagen = ImageDataGenerator()

test_annot = pd.read_csv('test.csv',header=0)
labels = test_annot['num_ants']

test_generator = test_datagen.flow_from_dataframe(dataframe=test_annot,shuffle=False,directory='dataset',class_mode='raw', x_col='file', y_col='num_ants',
													target_size=(128, 128), batch_size=32)


model = tf.keras.models.load_model('cnn_model.h5')
model.summary()

predictions = []
i = 0
# length = len(labels)
# for file in filelist:
# 	img = cv2.imread(path+'/'+file)
# 	t0 = time.perf_counter()
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
# 	#x = image.img_to_array(img)
# 	x = np.expand_dims(img, axis=0)
# 	#print(x.shape)
# 	P = model.predict(x, verbose=False)
# 	pred = np.argmax(P, axis=-1)[0]

# 	predictions.append(pred)
# 	interval = time.perf_counter() - t0
# 	#print(P,pred,labels[i])
# 	i+=1
# 	print("%d/%d"%(i,length),end='\r')
# 	#input()
# 	timefile.write('%.5f\n'%(interval))

# print()

length = len(labels)
predictions = np.argmax(model.predict(test_generator),axis=-1)
print()
print(len(predictions))

# timefile.close()

# VL = ['SEM FORMIGA',
# 	'FORMIGA']

VL = []
for i in range(10):
	VL.append(str(i))

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(labels, predictions, labels=range(len(VL)), target_names=VL))
print(confusion_matrix(labels, predictions))

import matplotlib.pyplot as plt
import seaborn as sns

cfmat = np.array(confusion_matrix(labels, predictions),dtype='float32')
print(cfmat.shape)

for i in range(cfmat.shape[0]):
    div = np.sum(cfmat[i,:])
    print(i+1)
    for j in range(cfmat.shape[1]):
        if div != 0:
            cfmat[i,j] = cfmat[i,j]/div
        else:
            cfmat[i,j] = 0.

ax = sns.heatmap(cfmat, annot=True, cmap='Blues',fmt='.2%')
ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(VL)
ax.yaxis.set_ticklabels(VL)

plt.show()