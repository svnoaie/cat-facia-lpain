#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 21:28:20 2022

@author: hao
"""


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
#%%
epochs = 100
train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('update/train',
                                                 target_size = (256, 256),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')#categorical

val_datagen = ImageDataGenerator(rescale = 1.255)
val_set = val_datagen.flow_from_directory('update/val',
                                            target_size = (256, 256),
                                            batch_size = 64,
                                            class_mode = 'categorical')
#
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1.255)
test_set = test_datagen.flow_from_directory('update/test',
                                            target_size = (256, 256),
                                            batch_size = 64,
                                            class_mode = 'categorical')
#%%
# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[256, 256, 3]))
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=5,activation = 'softmax'))
cnn.summary()
# Part 3 - Training the CNN

#%%
# Compiling the CNN
# cnn.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.001),
#             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics = ['accuracy'])
cnn.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history =cnn.fit(x = training_set, validation_data = val_set, epochs =epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
import  pandas as pd
data ={'train_acc':acc,
       'val_acc':val_acc,
       'train_loss':loss,
       'val_loss':val_loss}
pd.DataFrame(data).to_csv('./results/cust/cu_h_{}.csv'.format(epochs))

import  matplotlib.pyplot as plt
plt.plot(range(len(acc)),acc, 'b', label='Training accuracy')
plt.plot(range(len(acc)), val_acc, 'r', label='validation accuracy')
plt.title('Training and validation')
plt.legend(loc='lower right')
#plt.plot(epochs, loss, 'r', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.legend()
plt.show()
# Part 4 -c Making a single prediction
#%%
import os
path = 'VGG19 custom.h5'

cnn.save(path)
#%% 
Cnn=tf.keras.models.load_model('VGG19 custom.h5')
import numpy as np
import os
#%% confusion matrix
#五类
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
print(len(test_set))
Y_pred = Cnn.predict_generator(test_set)
print(Y_pred.shape)

y_pred = np.argmax(Y_pred, axis=1)
print(y_pred.shape)
print('Confusion Matrix')
print(set(test_set.classes),set(y_pred))
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ["pain_absent_(0)","pain_moderately_present_(1)", "pain_markedly_present_(2)",#pain_absent_(0)",
                'pain_present_but_unsure_how_to_grade_specifically',
                'unable_to_assess_for_other_reason']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

import numpy as np
Y_pred = Cnn.predict_generator(test_set)#, 1599 // 32+1)
y_pred = np.argmax(Y_pred,axis=1)
labels = [ "pain_absent_(0)","pain_moderately_present_(1)", "pain_markedly_present_(2)",
          'pain_present_but_unsure_how_to_grade_specifically',
          'unable_to_assess_for_other_reason'
          ]
yest=(test_set.classes)#,1599// 32+1)
#yest=test_set.labels
print(yest.shape,y_pred.shape)
print(yest,y_pred)
pd.DataFrame({'label':yest,'pred':y_pred}).to_csv('./results/cust/cu_labels_{}.csv'.format(epochs))
cm = confusion_matrix(yest,y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()

#%%
test=os.listdir("update/test/pain_moderately_present_(1)")
#%% 
i=6
import cv2
from keras.preprocessing import image
import keras
from tensorflow.keras.utils import load_img, img_to_array
test_image = keras.utils.load_img('update/test/pain_moderately_present_(1)/'+ test[i], target_size = (256, 256, 3))
#keras.utils
test_image = keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = Cnn.predict(test_image)
image=cv2.imread('update/test/pain_moderately_present_(1)/'+ test[i])
image=cv2.resize(image,(512,512))
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
print(training_set.class_indices)
f=np.argmax(result)
if f==0:
    print('pain_absent_(0)')
    image = cv2.putText(image, 'pain_absent_(0)', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
elif f==1:
    print('pain_markedly_present_(2)')
    image = cv2.putText(image, 'pain_markedly_present_(2)', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
elif f==2:
    print('pain_moderately_present_(1)')
    image = cv2.putText(image, 'pain_moderately_present_(1)', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("Result",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
# import cv2
# from keras.preprocessing import image
# test_image = image.load_img('pain_markedly_present_(2).png', target_size = (256, 256, 256))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = Cnn.predict(test_image)
# image=cv2.imread("pain_markedly_present_(2).png")
# image=cv2.resize(image,(512,512))
# font = cv2.FONT_HERSHEY_SIMPLEX
# # org
# org = (50, 50)
#
# # fontScale
# fontScale = 1
#
# # Blue color in BGR
# color = (255, 0, 0)
#
# # Line thickness of 2 px
# thickness = 2
#
# # Using cv2.putText() method
#
# print(training_set.class_indices)
# #%%
# f=np.argmax(result)
# if f==0:
#     print('brachycephalic')
#     image = cv2.putText(image, 'brachycephalic', org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
# elif f==1:
#     print('pain_markedly_present_(2)')
#     image = cv2.putText(image, 'pain_markedly_present_(2)', org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
# elif f==2:
#     print('siameseoriental_(dolichocephalic)')
#     image = cv2.putText(image, 'siameseoriental_(dolichocephalic)', org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
# cv2.imshow("Result",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #%%%
# import cv2
# from keras.preprocessing import image
# test_image = image.load_img('pain_moderately_present_(1).png', target_size = (256, 256, 256))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = Cnn.predict(test_image)
# image=cv2.imread("pain_moderately_present_(1).png")
# image=cv2.resize(image,(512,512))
# font = cv2.FONT_HERSHEY_SIMPLEX
# # org
# org = (50, 50)
#
# # fontScale
# fontScale = 1
#
# # Blue color in BGR
# color = (255, 0, 0)
#
# # Line thickness of 2 px
# thickness = 2
#
# # Using cv2.putText() method
#
# print(training_set.class_indices)
# #%%
#
# f=np.argmax(result)
# if f==0:
#     print('brachycephalic')
#     image = cv2.putText(image, 'brachycephalic', org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
# elif f==1:
#     print('normal')
#     image = cv2.putText(image, 'normal', org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
# elif f==2:
#     print('siameseoriental_(dolichocephalic)')
#     image = cv2.putText(image, 'siameseoriental_(dolichocephalic)', org, font,
#                        fontScale, color, thickness, cv2.LINE_AA)
# cv2.imshow("Result",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
