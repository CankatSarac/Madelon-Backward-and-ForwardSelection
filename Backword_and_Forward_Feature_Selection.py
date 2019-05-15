#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Datayi yukledim
df = pd.read_csv('madelon_csv.csv', sep=';')

# Train/test ayirdim
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:,:-1],
    df.values[:,-1:],
    test_size=0.25,
    random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)

# RF siniflandirmasi uyguladim
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Forward Feature Selection uyguladim
sfs1 = sfs(clf,
          k_features=4, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=5)

#  SFFS uyguladim
sfs1 = sfs1.fit(X_train, y_train)

# featuralara baktim
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

#modeli kurdum

clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train[:, make_feature_cols()], y_train)

y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[ ]:




