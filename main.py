#!/usr/bin/env python
# coding: utf-8

# # Load datasets

# In[2]:


import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('./catpainlabels210322.csv')
df


# In[60]:

#yuanbingtu-7 features
#5 classes

for i,e in enumerate(list(df.columns)[2:]):
    #plt.subplot(4,2,i+1)
    df[e].value_counts().plot.pie(subplots=True, figsize=(12, 10), autopct='%.2f', fontsize=10)
    plt.show()
#[{e:df[e].value_counts()} for e in list(df.columns)[2:]]
#sta = [{e:df[e].value_counts()} for e in list(df.columns)[2:]]

# In[149]:
#五类NB

X_cols = list(df.columns)[2:-1]
X = df[X_cols]
Y = df['overall_impression']
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
X_encoder = OneHotEncoder()
Y_encoder = LabelEncoder()
X_enc = X_encoder.fit_transform(X).toarray()
Y_enc = Y_encoder.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X_enc,Y_enc,test_size=0.2)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
cm = confusion_matrix(y_true=Y_test,y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Y_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Naive Bayes classifier")
plt.show()

print(classification_report(y_true=Y_test,y_pred=y_pred,target_names=Y_encoder.classes_))


# In[150]:
#dt classifier
#五类DT

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
clf.score(X_test,Y_test)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_true=Y_test,y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Y_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Classifier")
plt.show()

print(classification_report(y_true=Y_test,y_pred=y_pred,target_names=Y_encoder.classes_))


# # Drop the un

# In[63]:


labels = df['overall_impression']
labels = sorted(list(set(labels)))[:-2]


# In[64]:


df_drop = pd.concat([df[df['overall_impression']==l] for l in labels])
len(df_drop)


# In[65]:


print("After Drop :",set(df_drop['overall_impression']))# make sure we have drop two classes)


# In[66]:


df_drop.columns[2:-1]


# In[68]:
#圆饼图-3类


# 3 classes
#plt.figure(figsize=(15,15))

for i,e in enumerate(list(df_drop.columns)[2:]):
    #plt.subplot(4,2,i+1)
    df_drop[e].value_counts().plot.pie(subplots=True, figsize=(12, 10), autopct='%.2f', fontsize=10)
    plt.show()


# In[ ]:





# In[151]:
#三类NB

X_cols = list(df_drop.columns)[2:-1]
X = df_drop[X_cols]
Y = df_drop['overall_impression']

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

X_encoder = OneHotEncoder()
Y_encoder = LabelEncoder()

X_enc = X_encoder.fit_transform(X).toarray()
Y_enc = Y_encoder.fit_transform(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X_enc,Y_enc,test_size=0.2)


from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
cm = confusion_matrix(y_true=Y_test,y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Y_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Naive Bayes classifier")
plt.show()

print(classification_report(y_true=Y_test,y_pred=y_pred,target_names=Y_encoder.classes_))


# In[152]:
#三类DT

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
clf.score(X_test,Y_test)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_true=Y_test,y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Y_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Classifier")
plt.show()

print(classification_report(y_true=Y_test,y_pred=y_pred,target_names=Y_encoder.classes_))


# In[83]:
#7featurs-NB

np.random.seed(0)
X_cols = list(df_drop.columns)[2:-1]
for col in X_cols:
    X = df_drop[col]
    Y = df_drop['overall_impression']

    from sklearn.preprocessing import OneHotEncoder,LabelEncoder
    from sklearn.model_selection import train_test_split

    X_encoder = OneHotEncoder()
    Y_encoder = LabelEncoder()
    print(X.shape)
    X_enc = X_encoder.fit_transform(X.values.reshape(-1, 1)).toarray()
    Y_enc = Y_encoder.fit_transform(Y)

    X_train,X_test,Y_train,Y_test=train_test_split(X_enc,Y_enc,test_size=0.2)


    from sklearn.naive_bayes import CategoricalNB
    clf = CategoricalNB()
    clf.fit(X_train,Y_train)
    clf.score(X_test,Y_test)
    y_pred = clf.predict(X_test)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
    cm = confusion_matrix(y_true=Y_test,y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Y_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Naive Bayes classifier ({col})")
    plt.show()

    print(classification_report(y_true=Y_test,y_pred=y_pred,target_names=Y_encoder.classes_))


# In[85]:
#7 featurs-DT

from sklearn import tree
for col in X_cols:
    X = df_drop[col]
    Y = df_drop['overall_impression']

    from sklearn.preprocessing import OneHotEncoder,LabelEncoder
    from sklearn.model_selection import train_test_split

    X_encoder = OneHotEncoder()
    Y_encoder = LabelEncoder()
    print(X.shape)
    X_enc = X_encoder.fit_transform(X.values.reshape(-1, 1)).toarray()
    Y_enc = Y_encoder.fit_transform(Y)

    X_train,X_test,Y_train,Y_test=train_test_split(X_enc,Y_enc,test_size=0.2)



    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    clf.score(X_test,Y_test)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_true=Y_test,y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Y_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Decision Tree Classifier ({col})")
    plt.show()

    print(classification_report(y_true=Y_test,y_pred=y_pred,target_names=Y_encoder.classes_))


# In[ ]:




