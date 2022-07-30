#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:06:22 2022

@author: hao
"""



import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
#%%
train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('update/trainface',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')#categorical

val_datagen = ImageDataGenerator(rescale = 1.255)
val_set = val_datagen.flow_from_directory('update/valface',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')
#
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1.255)
test_set = test_datagen.flow_from_directory('update/testface',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')
#%%
# Part 2 - Building the CNN

# Initialising the CNN
cnnface = tf.keras.models.Sequential()

# Step 1 - Convolution
cnnface.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[256, 256, 3]))
cnnface.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
# Step 2 - Pooling
cnnface.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnnface.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnnface.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnnface.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnnface.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnnface.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnnface.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnnface.add(tf.keras.layers.Dense(units=4,activation = 'softmax'))
cnnface.summary()
# Part 3 - Training the CNN
#%%

# Compiling the CNN
# cnn.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.001),
#             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics = ['accuracy'])
cnnface.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Training the CNN on the Training set and evaluating it on the Test set
history =cnnface.fit(x = training_set, validation_data = val_set, epochs =5)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
import  matplotlib.pyplot as plt
plt.plot(epochs,acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation')
plt.legend(loc='lower right')
#plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
#plt.title('Training and validation loss')
plt.legend()
plt.show()
#%%4
import os
path = 'facecustom.h5'

cnnface.save(path)

#%%
Cnnface=tf.keras.models.load_model('facecustom.h5')
import numpy as np
import os


#%%
#matrix

from sklearn.metrics import classification_report, confusion_matrix
print(len(test_set))
Y_pred = Cnnface.predict_generator(test_set)
print(Y_pred.shape)

y_pred = np.argmax(Y_pred, axis=1)
print(y_pred.shape)
print('Confusion Matrix')
print(set(test_set.classes),set(y_pred))
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ["normal","atypical","brachycephalic","siamese_oriental_(dolichocephalic)"]#pain_absent_(0)",
                #'pain_present_but_unsure_how_to_grade_specifically',
                #'unable_to_assess_for_other_reason']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
Y_pred = Cnnface.predict_generator(test_set)#, 1599 // 32+1)
y_pred = np.argmax(Y_pred,axis=1)
labels = ["normal","atypical","brachycephalic","siamese_oriental_(dolichocephalic)"]
#labels = [ "pain_absent_(0)","pain_moderately_present_(1)", "pain_markedly_present_(2)",
 #         'pain_present_but_unsure_how_to_grade_specifically',
  #        'unable_to_assess_for_other_reason'
  #        ]
yest=(test_set.classes)#,1599// 32+1)
#yest=test_set.labels
print(yest.shape,y_pred.shape)
print(yest,y_pred)
cm = confusion_matrix(yest,y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.title("confusion matrix test-face")
plt.show()

#%%
test=os.listdir("update/testface/normal")
#%%

i=14
import cv2
from keras.preprocessing import image
#import cv2
import keras
#import tensorflow as tf
#from keras.preprocessing import image
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
#import tf.keras.preprocessing.image
test_image = keras.utils.load_img('update/testface/normal/'+ test[i], target_size = (256, 256, 3))
#keras.utils
test_image = keras.utils.img_to_array(test_image)
#test_image = image.load_img('update/testhair/short_hair/'+ test[i], target_size = (256, 256, 3))
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = Cnnface.predict(test_image)
image=cv2.imread('update/testface/normal/'+ test[i])
image=cv2.resize(image,(512,512))
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
   
# Using cv2.putText() method

print(training_set.class_indices)

f=np.argmax(result)
#
if f==0:
    print('normal')
    image = cv2.putText(image, 'normal', org, font, 
    fontScale, color, thickness, cv2.LINE_AA)
elif f==1:
    print('atypical')
    image = cv2.putText(image, 'atypical', org, font, 
    fontScale, color, thickness, cv2.LINE_AA)
elif f==2:
    print('brachycephalic')
    image = cv2.putText(image, 'brachycephalic', org, font, 
    fontScale, color, thickness, cv2.LINE_AA)
elif f==3:
    print('siamese_oriental_(dolichocephalic)')
    image = cv2.putText(image, 'siamese_oriental_(dolichocephalic)', org, font, 
    fontScale, color, thickness, cv2.LINE_AA)
#elif f==2:
   # print('feature_present_but_unsure_how_to_grade_specifically')
   # image = cv2.putText(image, 'feature_present_but_unsure_how_to_grade_specifically', org, font, 
    #fontScale, color, thickness, cv2.LINE_AA)
#
#if f==0:
  #  print('pain_absent_(0)')
  #  image = cv2.putText(image, 'pain_absent_(0)', org, font, 
  #                     fontScale, color, thickness, cv2.LINE_AA)
#elif f==1:
 #   print('pain_markedly_present_(2)')
 #   image = cv2.putText(image, 'pain_markedly_present_(2)', org, font, 
 #                      fontScale, color, thickness, cv2.LINE_AA)
#elif f==2:
   # print('pain_moderately_present_(1)')
   # image = cv2.putText(image, 'pain_moderately_present_(1)', org, font, 
   #                    fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("Result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
