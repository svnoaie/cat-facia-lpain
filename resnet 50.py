#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 21:28:20 2022

@author: haocc
"""
import sys

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
# %%
epochs = 100
batch_size = 40

sys.stdout = open('./results/resnet_{}.txt'.format(epochs),'w')

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('update/train',
                                                 target_size=(256, 256),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')  # categorical
#
val_datagen = ImageDataGenerator(rescale = 1./255)
val_set = val_datagen.flow_from_directory('update/val',
                                          target_size = (256, 256),
                                          batch_size = batch_size,
                                          class_mode = 'categorical')
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('update/test',
                                            target_size=(256, 256),
                                            batch_size=batch_size,
                                            class_mode='categorical')
#class_mode = 'categorical')
resnet_model = tf.keras.models.Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                                                 input_shape=(256,256,3),
                                                 pooling='avg',classes=5,
                                                 weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable=True

resnet_model.add(pretrained_model)
resnet_model.add(tf.keras.layers.Flatten())
resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
resnet_model.add(tf.keras.layers.Dense(units=5, activation='softmax'))
resnet_model.summary()
# Part 3 - Training the CNN
# %%
resnet_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),#学习率为0.001
                     loss = 'categorical_crossentropy',
                     metrics=['accuracy'])
history=resnet_model.fit(x=training_set, validation_data=val_set, epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


import pandas as pd

data ={'train_acc':acc,
       'val_acc':val_acc,
       'train_loss':loss,
       'val_loss':val_loss}
pd.DataFrame(data).to_csv('./results/resnet50_h_{}.csv'.format(epochs))
import  matplotlib.pyplot as plt
plt.plot(range(len(acc)),acc, 'b', label='Training accuracy')
plt.plot(range(len(acc)), val_acc, 'r', label='validation accuracy')
plt.title('Training and validation')
plt.legend(loc='lower right')
#plt.plot(epochs, loss, 'r', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='validation loss')
#plt.title('Training and validation loss')
plt.legend()
#plt.show()cc
# Part 4 - Making a single prediction
# %%
#loss

import  matplotlib.pyplot as plt
#plt.plot(epochs,acc, 'b', label='Training accuracy')
#plt.plot(epochs, val_acc, 'r', label='validation accuracy')
#plt.title('Training and validation loss')
plt.legend(loc='lower right')
plt.plot(range(loss), loss, 'r', label='Training loss')
plt.plot(range(loss), val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
#plt.show()
# #%%
# import os

path = 'resnet50.h5'

#resnet_model.save(path)
# %%
resnet_model = tf.keras.models.load_model(path)
import numpy as np
import os
# %% confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print(len(test_set))
Y_pred = resnet_model.predict_generator(test_set)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(set(test_set.classes), set(y_pred))
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ["pain_absent_(0)", "pain_moderately_present_(1)", "pain_markedly_present_(2)",
                'pain_present_but_unsure_how_to_grade_specifically',
                'unable_to_assess_for_other_reason']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

Y_pred = resnet_model.predict_generator(test_set)  # , 1599 // 32+1)
y_pred = np.argmax(Y_pred, axis=1)
labels = ["pain_absent_(0)", "pain_moderately_present_(1)", "pain_markedly_present_(2)",
          'pain_present_but_unsure_how_to_grade_specifically',
          'unable_to_assess_for_other_reason'
          ]
yest = (test_set.classes)  # ,1599// 32+1)
# yest=test_set.labels
print(yest.shape, y_pred.shape)
print(yest, y_pred)
cm = confusion_matrix(yest, y_pred)
import pandas as pd
pd.DataFrame({'label':yest,'pred':y_pred}).to_csv('./results/resnet50_labels_{}.csv'.format(epochs))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot(cmap=plt.cm.Blues)
#plt.show()

cm = confusion_matrix(yest, y_pred)
print(cm)
# %%
test = os.listdir("update/test/pain_moderately_present_(1)")
# %%
i = 28
import cv2
from keras.preprocessing import image
#
#import cv2
import keras
test_image = keras.utils.load_img('update/test/pain_moderately_present_(1)/'+ test[i], target_size = (256, 256, 3))
#keras.utils
test_image = keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = resnet_model.predict(test_image)
image = cv2.imread('update/test/pain_moderately_present_(1)/' + test[i])
image = cv2.resize(image, (512, 512))
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

f = np.argmax(result)
if f == 0:
    print('pain_absent_(0)')
    image = cv2.putText(image, 'pain_absent_(0)', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
elif f == 1:
    print('pain_markedly_present_(2)')
    image = cv2.putText(image, 'pain_markedly_present_(2)', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
elif f == 2:
    print('pain_moderately_present_(1)')
    image = cv2.putText(image, 'pain_moderately_present_(1)', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # %%
# import cv2
# from keras.preprocessing import image
#
# test_image = image.load_img('pain_markedly_present_(2).png', target_size=(256, 256, 256))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = resnet_model.predict(test_image)
# image = cv2.imread("pain_markedly_present_(2).png")
# image = cv2.resize(image, (512, 512))
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
# # %%
# f = np.argmax(result)
# if f == 0:
#     print('brachycephalic')
#     image = cv2.putText(image, 'brachycephalic', org, font,
#                         fontScale, color, thickness, cv2.LINE_AA)
# elif f == 1:
#     print('pain_markedly_present_(2)')
#    cc image = cv2.putText(image, 'pain_markedly_present_(2)', org, font,
#                         fontScale, color, thickness, cv2.LINE_AA)
# elif f == 2:
#     print('siameseoriental_(dolichocephalic)')
#     image = cv2.putText(image, 'siameseoriental_(dolichocephalic)', org, font,
#                         fontScale, color, thickness, cv2.LINE_AA)
# cv2.imshow("Result", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # %%%
# import cv2
# from keras.preprocessing import image
#
# test_image = image.load_img('pain_moderately_present_(1).png', target_size=(256, 256, 256))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = resnet_model.predict(test_image)
# image = cv2.imread("pain_moderately_present_(1).png")
# image = cv2.resize(image, (512, 512))
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
# # %%
#
# f = np.argmax(result)
# if f == 0:
#     print('brachycephalic')
#     image = cv2.putText(image, 'brachycephalic', org, font,
#                         fontScale, color, thickness, cv2.LINE_AA)
# elif f == 1:
#     print('normal')
#     image = cv2.putText(image, 'normal', org, font,
#                         fontScale, color, thickness, cv2.LINE_AA)
# elif f == 2:
#     print('siameseoriental_(dolichocephalic)')
#     image = cv2.putText(image, 'siameseoriental_(dolichocephalic)', org, font,
#                         fontScale, color, thickness, cv2.LINE_AA)
# cv2.imshow("Result", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
