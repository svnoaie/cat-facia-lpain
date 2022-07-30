#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 13:33:37 2022

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
labels_df = df['overall_impression']
label_list = []

# label_tag_dict = {0:['pain_moderately_present_(1)','pain_markedly_present_(2)','pain_present_but_unsure_how_to_grade_specifically'],1:['pain_absent_(0)']}
# label_tag_dict = {1:['pain_moderately_present_(1)'],0:['pain_absent_(0)'],3:['unable_to_assess_for_other_reason'],2:['pain_markedly_present_(2)']}#####4
#label_tag_dict = {0:['pain_absent_(0)'],1:['pain_moderately_present_(1)'],2:['pain_markedly_present_(2)'],3:['unable_to_assess_for_other_reason']}####4
#label_tag_dict = {0:'pain_absent_(0)',1:'pain_moderately_present_(1)',2:'pain_markedly_present_(2)'}####3
# label_tag_dict = {0:['pain_moderately_present_(1)'],1:['pain_absent_(0)']}####2
#label_dict = {}
#%%

image_and_labels   = [] #inlcuding all available pairs （images vs label）

for index, row in tqdm(df.iterrows()):
  # print(row['image_id'])
  #print(row['overall_impression'])
  label = row['overall_impression']
  #print(label)
  # if label in label_tag_dict.values():
  image_path =root_image_path+str('/')+row['imageid']
  # ddd = None
  if os.path.exists(image_path):
    #print(label)#directory name
    #ddd = cv2.imread(image_path)
    image_and_labels.append([image_path,label])

print(len(image_and_labels))#3948
  # if ddd is None:
  #   continue
  #
  # if not label in label_dict.keys():
  #   label_dict[label] =[]
  # label_dict[label].append(row['imageid'])
#%%
# max_value = 0
# for key in label_dict.keys():
#
#   tmp_value = len(label_dict[key])
#
#   if tmp_value > max_value:
#     max_value = tmp_value

#%%
# print(label_dict[label_tag_dict[0][0]])
# #%%
# for key in label_tag_dict.keys():
#
#   print(key)
#   len_ = len(label_dict[label_tag_dict[key][0]])
#   # print(len_)
#   print(key,max_value/len_)
# print(10*'=')
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
  if  os.path.exists('update/train/' + e)==False:
    os.mkdir('update/train/'+e)
  if  os.path.exists('update/test/' + e) ==False:
    os.mkdir('update/test/'+e)
  if  os.path.exists('update/val/' + e) ==False:
    os.mkdir('update/val/'+e)
#
for item in train_list:
  img_path = item[0]
  print("train:",img_path)
  label = item[1]
  img = cv2.imread(img_path)

  img_name = img_path.split('/')[-1]

  cv2.imwrite('update/train/'+label+'/'+img_name,img)


for item in val_list:
  img_path = item[0]
  print("val:",img_path)
  label = item[1]
  img = cv2.imread(img_path)

  img_name = img_path.split('/')[-1]

  cv2.imwrite('update/val/'+label+'/'+img_name,img)

for item in test_list:
  img_path = item[0]
  print("test:",img_path)
  label = item[1]
  img = cv2.imread(img_path)

  img_name = img_path.split('/')[-1]

  cv2.imwrite('update/test/'+label+'/'+img_name,img)
 #%%
 # 分层抽样

#%%

 
# five_data = []
# for label_key in label_dict.keys():
#   tag = None
#   for label_tag in label_tag_dict.keys():
#     if label_key in label_tag_dict[label_tag]:
#       tag = label_tag
#       break
#
#   image_list = label_dict[label_key]
#
#   data={'data':None,'val_splice':[]}
#   if not tag is None:
#     image_list = [ image_root_path +'/'+x+","+str(tag)+"\n" for x in image_list]
#     random.shuffle(image_list)
#     image_list_len = len(image_list)
#
#     diff_num = int(image_list_len/5)
#     data['data'] = image_list
#     start=0
#     for i in range(5):
#       start_num =start*diff_num
#       end_num = (start+1)*diff_num
#       start +=1
#       data['val_splice'].append([start_num,end_num])
      # print(start_num,end_num)
    # print(image_list_len)
    ####################
    # 划分五折
    # train_len = int(image_list_len*split_rate)
    # train_list.extend(image_list[:train_len])

    # val_list.extend(image_list[train_len:])
    # five_data.append(data)

# for i in range(5):
#   train_list=[]
#   val_list = []
#   for tmp_data in five_data:
#     image_list = tmp_data['data']
#     split_num = tmp_data['val_splice'][i]
#     start_num,end_num = split_num
#
#     val_list.extend(image_list[start_num:end_num])
#     if start_num==0:
#       train_list.extend(image_list[end_num:-1])
#     else:
#       train_list.extend(image_list[0:start_num])
#       train_list.extend(image_list[end_num:-1])
#
#     # print(start_num,end_num)
#
#
# # write
#   with open(root_txt_path+'/train'+str(i)+'.txt','w') as f:
#     f.writelines(train_list)
#   with open(root_txt_path+'/val'+str(i)+'.txt','w') as f:
#     f.writelines(val_list)


