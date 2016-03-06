
# coding: utf-8

# ## Library Import

# In[3]:

## Classical libraries
import boto
import pandas as pd
import numpy as np
import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import pyprind


# ## Connectors

# In[5]:

s3 = boto.connect_s3(aws_access_key_id='', 
                     aws_secret_access_key='')
s3_bucket_p2 = s3.get_bucket('harvard_practicals2')


# ## Support Functions

# ##### Features Engineering

# In[6]:

def XML_get(root, element, attrib_name=''):
    res = []
    if attrib_name != '':
        for t in root.iter(element):
            try: 
                res.append(element + ' ' + t.attrib[attrib_name])
            except:
                continue
                #res.append(None)
    else:
        for t in root.iter(element):
            try:
                res.append(t.attrib)
            except:
                continue
                #res.append(None)
    return res


# In[7]:

def createFeatures(root):
    load_dll_files = XML_get(root, 'load_dll',   'filename')
    load_dll_files = [i.replace('\\', ' ')  for i in load_dll_files]
    
    vm_protect_target = XML_get(root, 'vm_protect', 'target')
    
    vm_protect_protect = XML_get(root, 'vm_protect', 'protect')
    vm_protect_behavior = XML_get(root, 'vm_protect', 'behavior')
    open_key_key = XML_get(root, 'open_key', 'key')
    
    kill_process = XML_get(root, 'kill_process', 'apifunction')
    
    res = np.concatenate([load_dll_files,
                          kill_process,
                          vm_protect_target,
                          vm_protect_protect,
                          vm_protect_behavior,
                          open_key_key
                         ])
    
    return res


# ##### Fitting

# In[8]:

def classify_and_score(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    print 'train accuracy: ' + str(metrics.accuracy_score(y_train, train_preds))
    test_preds = clf.predict(X_test)
    print 'validation accuracy: ' + str(metrics.accuracy_score(y_test, test_preds))
    return X_train, X_test, y_train, y_test


# ## Loop on the S3 train

# ##### Parse the files

# In[9]:

len(list(s3_bucket_p2.list(prefix='test')))


# In[10]:

s3_bucket_p2.list(prefix='train')


# In[11]:

# Results repo
ids_classes = []
trees = []

#### TRAIN

# Size of the operation (to properly update pyprind)
print("Parsing the Training Set")
loop_size = len(list(s3_bucket_p2.list(prefix='train')))
mybar = pyprind.ProgBar(loop_size)

c = 0
for i in s3_bucket_p2.list(prefix='train'):
#     c = c+1
#     if c > 10:
#         break
    if i.key.rfind('.xml')>0:
        # Work : 
        id_str, clazz = i.key.replace('train/','').split('.')[:2]
        ids_classes.append((id_str, clazz))
        tree = ET.fromstring(i.get_contents_as_string())
        trees.append(tree)
    
    mybar.update()


# In[12]:

#### TEST

# Size of the operation (to properly update pyprind)
print("Parsing the Test Set")
loop_size = len(list(s3_bucket_p2.list(prefix='test')))
mybar = pyprind.ProgBar(loop_size)

c = 0
for i in s3_bucket_p2.list(prefix='test'):
#     c = c+1
#     if c > 10:
#         break
    if i.key.rfind('.xml')>0:
        # Work : 
        #print(i),
        id_str, clazz = i.key.replace('test/','').split('.')[:2]
        ids_classes.append((id_str, clazz))
        tree = ET.fromstring(i.get_contents_as_string())
        trees.append(tree)
    
    mybar.update()


# In[13]:


### Get the final DF
df = pd.DataFrame.from_records(ids_classes, columns=['id','class']) 


# ##### Get the tags (Mark Style)

# In[14]:

print("Get the Tag in the features")
mybar = pyprind.ProgBar(len(trees))

not_calls = ['processes','all_section','thread','process']
docs = []
for tree in trees:
    calls = []
    for ele in tree.iter():
        if ele.tag not in not_calls:
            calls.append(ele.tag)   
    docs.append(calls)
    mybar.update()


# ##### Adding the other features (from XMLget)

# In[15]:

print("Extend the features XMLget")
mybar = pyprind.ProgBar(len(trees))

for i in range(0,len(trees)):
    docs[i].extend(createFeatures(trees[i]))
    mybar.update()


# ## Fitting

# ##### Vectorization

# In[16]:

print("Vectorization")
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform([' '.join(doc) for doc in docs])
X = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())


# ##### ix of Fit

# In[17]:

print("Fitting")
test_ix  = (df['class'] == 'X')
train_ix = (df['class'] != 'X')
y = df['class'].values


# ## RF

# In[98]:

# ### Model Fit
# rfc = RFC(n_estimators=50)
# _ = classify_and_score(rfc, X.loc[train_ix], y[train_ix.values])

# print("sum test_ix")
# print(sum(test_ix))
# print("prediction")

# df.loc[test_ix, 'Prediction'] = rfc.predict(X.loc[test_ix])

# ### Get the Prediction
# malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
#            "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
#            "VB", "Virut", "Zbot"]

# malware_classes_dict = pd.DataFrame(malware_classes)
# malware_classes_dict.columns = ['Name']
# malware_classes_dict.loc[:, 'ID'] = malware_classes_dict.index.values 
# malware_classes_dict.index = malware_classes_dict.Name.values

# ### Fill the res
# df.loc[test_ix, 'Prediction'] = df.loc[test_ix, 'Prediction'].apply(lambda x : malware_classes_dict.loc[x, 'ID'])


# ## XG Boost

# In[21]:

malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]


# In[41]:

y_list = []
for clazz in y[train_ix.values]:
    i=0
    for el in malware_classes:
        if clazz == el:
            y_list.append(i)
        i=i+1
        
numerical_y_train = np.array(y_list)

y_list = []
for clazz in y[test_ix.values]:
    i=0
    for el in malware_classes:
        if clazz == el:
            y_list.append(i)
        i=i+1
        
numerical_y_test = np.array(y_list)


# In[50]:

import numpy as np
import xgboost as xgb

train_X = X.loc[train_ix]
test_X  = X.loc[test_ix]
train_Y = numerical_y_train
test_Y = numerical_y_test

#train_X, test_X, train_Y, test_Y = train_test_split(X, numerical_y)

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
#param['nthread'] = 4 #If not defined, it is set to maximum
param['num_class'] = 15
param["booster"] = "gbtree"
#param["lambda"] = 1 #default: 1
#param["alpha"] = 0 #default:0
#param["gamma"] = 0 # the larger the more conservative the model is

num_round = 5 # Given was 5 (Train error decreases with increasing rounds, maybe keep it small to avoid overfitting)
ntree=200 # Given was 6

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]

#Tree
bst = xgb.train(param, xg_train, num_round);
# get prediction
pred = bst.predict( xg_test , ntree_limit=ntree);

df.loc[test_ix, 'Prediction'] = pred


# In[52]:

malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

malware_classes_dict = pd.DataFrame(malware_classes)
malware_classes_dict.columns = ['Name']
malware_classes_dict.loc[:, 'ID'] = malware_classes_dict.index.values 
malware_classes_dict.index = malware_classes_dict.Name.values


# In[58]:

df.loc[test_ix, 'Prediction'] = df.loc[test_ix, 'Prediction'].apply(lambda x : malware_classes[int(x)])


# ## Saving Res in S3

# In[104]:

print("Saving")
df.loc[test_ix, ['id', 'Prediction']].to_csv('res.csv', index = False)
k = s3_bucket_p2.new_key('res.csv')
k.set_contents_from_filename('res.csv')

print('***** END *****')


# ## TEST ZONE

# In[8]:

# import pandas as pd
# import boto

# a = pd.DataFrame([0,1,2])
# print(a)
# a.to_csv('res.csv')

# ### Connection to S3

# s3 = boto.connect_s3(aws_access_key_id='', 
#                      aws_secret_access_key='')

# s3_bucket_p2 = s3.get_bucket('harvard_practicals2')
# k = s3_bucket_p2.new_key('res.csv')
# k.set_contents_from_filename('res.csv')
##sdsdsds

