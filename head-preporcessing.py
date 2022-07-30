#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:48:24 2022

@author: hao
"""


import cv2
from numpy.lib.function_base import piecewise
import pandas as pd
import random 
random.seed(10)
import os
from tqdm import tqdm
# # 65cff374-1e99-4e48-8ad5-32c909948705.png
# ddd = cv2.imread('D:/ali/v5/v5/images140721/65cff374-1e99-4e48-8ad5-32c909948705.png')

# # f6f0bb6e-d4be-4eb7-8492-cd6c3a750f5d.png


# print(ddd.shape)

# exec()

file_path ="update/catpainlabels210322.csv"
root_image_path ="all of the pic"
image_root_path ="all of the pic"
root_txt_path ="train.txt"
#%%
os.makedirs(root_txt_path,exist_ok=True)



# df = pd.read_excel(file_path)
df = pd.read_csv(file_path)
labels_df = df['head_position']
label_list = []

#%%
image_and_labels   = [] #inlcuding all available pairs （images vs label）

for index, row in tqdm(df.iterrows()):
  # print(row['image_id'])
  #print(row['overall_impression'])
  label = row['head_position']
  #print(label)
  # if label in label_tag_dict.values():
  image_path =root_image_path+str('/')+row['imageid']
  # ddd = None
  if os.path.exists(image_path):
    #print(label)#directory name
    #ddd = cv2.imread(image_path)
    image_and_labels.append([image_path,label])

print(len(image_and_labels))#3948
#%%

# train_list = []
# val_list = []

# 6 :2 : 2
len_of_datasets = len(image_and_labels)
random.seed(0)
random.shuffle(image_and_labels)
train_list = image_and_labels[:int(len_of_datasets*(0.6))]
val_list = image_and_labels[int(len_of_datasets*(0.6)):int(len_of_datasets*(0.8))]
test_list = image_and_labels[int(len_of_datasets*(0.8)):]


print("train:val:test = {}:{}:{}".format(len(train_list),len(val_list),len(test_list)))
labelset = set([e[1] for e in image_and_labels])
print(labelset)# 5 label

for e in labelset:
  print(e)
  if  os.path.exists('update/trainhead/' + e)==False:
    os.mkdir('update/trainhead/'+e)
  if  os.path.exists('update/testhead/' + e) ==False:
    os.mkdir('update/testhead/'+e)
  if  os.path.exists('update/valhead/' + e) ==False:
    os.mkdir('update/valhead/'+e)
#
for item in train_list:
  img_path = item[0]
  print("trainhead:",img_path)
  label = item[1]
  img = cv2.imread(img_path)

  img_name = img_path.split('/')[-1]

  cv2.imwrite('update/trainhead/'+label+'/'+img_name,img)


for item in val_list:
  img_path = item[0]
  print("valhead:",img_path)
  label = item[1]
  img = cv2.imread(img_path)

  img_name = img_path.split('/')[-1]

  cv2.imwrite('update/valhead/'+label+'/'+img_name,img)

for item in test_list:
  img_path = item[0]
  print("testhead:",img_path)
  label = item[1]
  img = cv2.imread(img_path)

  img_name = img_path.split('/')[-1]

  cv2.imwrite('update/testhead/'+label+'/'+img_name,img)