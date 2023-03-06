import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop
from  keras.layers import Input
import pandas as pd
from math import log
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.regularizers import l1,l2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_annot = pd.read_csv('train.csv',header=0)
y_train = train_annot['num_ants']
test_annot = pd.read_csv('validation.csv',header=0)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_annot,directory='dataset',class_mode='raw', x_col='file', y_col="num_ants",
													target_size=(128, 128), batch_size=32)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_annot,directory='dataset',class_mode='raw', x_col='file', y_col='num_ants',
													target_size=(128, 128), batch_size=32)


inn = layers.Input(shape=(128,128,3))
# x = layers.experimental.preprocessing.Rescaling(1./255.)(inn)

# VGG16 = tf.keras.applications.vgg16.VGG16(
# 	include_top=False, weights='imagenet', input_tensor=inn,
# )
# backbone = VGG16

MOBILENET = tf.keras.applications.mobilenet.MobileNet(
	include_top=False, weights='imagenet', input_tensor=inn,
)
backbone = MOBILENET

# MOBILENETv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
# 	include_top=False, weights='imagenet', input_tensor=inn,
# )
# backbone = MOBILENETv2

# MOBILENETv3 = tf.keras.applications.MobileNetV3Small(
# 	include_top=False, weights='imagenet', input_tensor=inn,
# )
# backbone = MOBILENETv3

# DENSENET = tf.keras.applications.densenet.DenseNet169(
# 	include_top=False, weights='imagenet', input_tensor=inn,
# )
# backbone = DENSENET

# RESNET50 = tf.keras.applications.resnet50.ResNet50(
#     include_top=False, weights='imagenet',input_tensor=inn
# )
# backbone = RESNET50

# RESNET101 = tf.keras.applications.resnet.ResNet101(
#     include_top=False, weights='imagenet',input_tensor=inn
# )
# backbone = RESNET101

# NasNetMobile = tf.keras.applications.nasnet.NASNetMobile(
#     include_top=False, weights='imagenet',input_tensor=inn,
# )
# backbone = NasNetMobile

# EFFICIENTNET = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
#     include_top=False, weights='imagenet',input_tensor=inn,
# )
# backbone = EFFICIENTNET

# backbone.trainable = False

# x = layers.Conv2D(32, (3, 3), activation='relu')(inn)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Flatten()(x)
x = layers.Flatten()(backbone.layers[-1].output)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Dense(1024,activation='linear',kernel_regularizer=l2(0.01))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Dense(1024,activation='linear',kernel_regularizer=l2(0.01))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.1)(x)
x = layers.Dense(32,activation='linear',kernel_regularizer=l1(0.01))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.3)(x)
o = layers.Dense(10,activation='softmax',kernel_regularizer=l1(0.01))(x)


class_weights = compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
for i in range(len(class_weights)):
	class_weights[i] = class_weights[i]**0.5
# 	class_weights[i] = min(class_weights[i],20)
# 	# class_weights[i] = max(class_weights[i],0.6)
# class_weights = np.insert(class_weights,3,0.0)
# class_weights[0] = .4
# class_weights[1] = 1.2
# class_weights[2] = 2.4
# class_weights[3] = 3.6
# class_weights[4] = 4.8
# class_weights[5] = 6.0
# class_weights[6] = 7
# class_weights[7] = 7
# class_weights[8] = 7
# class_weights[9] = 8
print(class_weights)
class_weight_dict = dict(enumerate(class_weights))

model = tf.keras.Model(inputs=inn,outputs=[o])

# model.summary()
# model.load_weights('cnn_model.h5')

num_epochs = 4000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
				loss=tf.keras.losses.SparseCategoricalCrossentropy(),
				metrics = ['accuracy'])

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.1,
        patience = 5,
        verbose  = 1,
        mode     = 'min',
        min_delta  = 0.001,
        cooldown = 0,
        min_lr   = 0
    )

history = model.fit(train_generator,
		validation_data=test_generator,
		steps_per_epoch=128,
		epochs = num_epochs,
		# validation_steps = 64,
		class_weight=class_weight_dict,
		verbose = 1,
		callbacks=[
		# tf.keras.callbacks.LearningRateScheduler(
		# 	lambda epoch: 0.0001/((epoch+1)**0.5)
		# ),
		tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,min_delta=0.001,restore_best_weights=True),
		reduce_on_plateau
	])

model.save('cnn_model.h5')

model.evaluate(test_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
