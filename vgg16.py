# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 10:24:35 2022

@author: USER
"""
# this one
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
#%%1
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('update/train',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_datagen = ImageDataGenerator(rescale = 1.0/255)
val_set = val_datagen.flow_from_directory('update/val',
                                          target_size = (256, 256),
                                          batch_size = 32,
                                          class_mode = 'categorical')
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('update/test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')
#%%
model = Sequential()
model.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=5, activation="softmax"))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#%%
history =model.fit(steps_per_epoch=100,x=training_set, validation_data= val_set,epochs=10)


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

#plt.plot(epochs, loss, 'r', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='validation loss')
#plt.title('Training and validation loss')
plt.legend()
plt.show()
#%% 2
model.save('VGG16.h5')
#%% 3
model=tf.keras.models.load_model('VGG16.h5')
import os
import numpy as np
#%%

from sklearn.metrics import classification_report, confusion_matrix

print(len(test_set))
Y_pred = model.predict_generator(test_set)
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

Y_pred = model.predict_generator(test_set)  # , 1599 // 32+1)
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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot(cmap=plt.cm.Blues)
plt.show()

cm = confusion_matrix(yest, y_pred)
print(cm)
#%%
test=os.listdir("update/test/pain_moderately_present_(1)")
#%%
i=7
import cv2
from keras.preprocessing import image
#
#import cv2
import keras
#import tensorflow as tf
#from keras.preprocessing import image
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
#import tf.keras.preprocessing.image
#tf.keras.utils.load_img
#test_image = image.load_img('update/test/pain_moderately_present_(1)/'+ test[i], target_size = (256, 256, 3))
#test_image = image.load_img('update/test/pain_moderately_present_(1)/'+ test[i], target_size = (256, 256, 3))
test_image = keras.utils.load_img('update/test/pain_moderately_present_(1)/'+ test[i], target_size = (256, 256, 3))
#keras.utils
test_image = keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
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
#%% 4 
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import confusion_matrix
#Y_pred = model.predict_generator(test_set, 263 // 32+1)
#y_pred = np.argmax(Y_pred, axis=1)
#print('Confusion Matrix')
#print(confusion_matrix(test_set.classes, y_pred))
#print('Classification Report')
#target_names = ['pain_absent_(0)', 'pain_markedly_present_(2)', 'pain_moderately_present_(1)']
#print(classification_report(test_set.classes, y_pred, target_names=target_names))
#%%




