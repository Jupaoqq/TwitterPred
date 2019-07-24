
# coding: utf-8

# ## Importing Libraries

# In[61]:


import numpy as np
import pandas as pd
import os
import os.path
from os import path
import csv


# # 2. Get the Data

# In[62]:


dt = pd.read_csv('user.csv', header=None)
temp_user = str(dt[1][4])
tweets = 'collection'
dataset_gunviolence = pd.read_csv(os.path.join(tweets,'%s_interaction.csv') % temp_user)


# ## Feature Engineering

# In[63]:


dataset_gunviolence.status = pd.Categorical(dataset_gunviolence.status)
#print (dataset_gunviolence.status)
dataset_gunviolence['code'] = dataset_gunviolence.status.cat.codes
dataset_gunviolence['screen_name'] = dataset_gunviolence['screen_name'].astype('category').cat.codes
dataset_gunviolence['truncated'] = dataset_gunviolence['truncated'].astype('category').cat.codes
dataset_gunviolence['entities.hashtags'] = dataset_gunviolence['entities.hashtags'].astype('category').cat.codes
dataset_gunviolence['entities.user_mentions'] = dataset_gunviolence['entities.user_mentions'].astype('category').cat.codes
dataset_gunviolence['entities.symbols'] = dataset_gunviolence['entities.symbols'].astype('category').cat.codes
dataset_gunviolence['bool(entities.polls) '] = dataset_gunviolence['bool(entities.polls)'].astype('category').cat.codes
dataset_gunviolence['origin_user'] = dataset_gunviolence['origin_user'].astype('category').cat.codes
dataset_gunviolence['origin_entities_hashtags'] = dataset_gunviolence['origin_entities_hashtags'].astype('category').cat.codes
dataset_gunviolence['origin_entities_user_mentions'] = dataset_gunviolence['origin_entities_user_mentions'].astype('category').cat.codes
dataset_gunviolence['origin_entities_symbols'] = dataset_gunviolence['origin_entities_symbols'].astype('category').cat.codes
dataset_gunviolence['origin_entities_polls'] = dataset_gunviolence['origin_entities_polls'].astype('category').cat.codes

dataset_gunviolence['origin_status'] = dataset_gunviolence['origin_status'].astype('category').cat.codes

dataset_gunviolence = dataset_gunviolence.drop(['node','status'], axis=1)
#dataset_gunviolence.dtypes
dataset_gunviolence = dataset_gunviolence.select_dtypes(exclude=['object'])
dataset_gunviolence = dataset_gunviolence.astype(float)
dataset_gunviolence.dtypes


# # 4. Prepare the data for Machine Learning algorithms

# In[64]:


# additional imports
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math

# to make this notebook's output stable across runs
np.random.seed(42)
# You might want to use the following packages
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix # optional
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[65]:


# Data Splitting
data = dataset_gunviolence
# Data Discovery
data = data.fillna(0)

# data['len(user.description)'] = data['len(user.description)'].astype(float)
# data['user.followers_count'] = data['user.followers_count'].astype(float)
# data['user.friends_count'] = data['user.friends_count'].astype(float)
# data['user.listed_count'] = data['user.listed_count'].astype(float)
# data['user.statuses_count'] = data['user.statuses_count'].astype(float)
# data['user.favourites_count'] = data['user.favourites_count'].astype(float)
              
    

data = data[[		'screen_name',
					'len(user.description)',
					'user.followers_count',
					'user.friends_count',
					'user.listed_count',
					'user.statuses_count',
					'user.favourites_count',
					'retweet_count',
					'favorite_count',
					'origin_user',
					'origin_user_len_description',
					'origin_user_followers_count',
					'origin_user_friends_count',
					'origin_user_listed_count',
					'origin_user_statuses_count',
					'origin_user_favourites_count',
					'origin_retweet_count',
					'origin_favorite_count',
					'origin_status',
					'min_interval',

					'degree_centrality',
					'eigenvector_centrality',
					'closeness_centrality',
					'betweenness_centrality',
					'load_centrality',
					'subgraph_centrality',
					'harmonic_centrality',
					'weighted_degree_centrality',
					'weighted_eigenvector_centrality',
					'weighted_closeness_centrality',
					'weighted_betweenness_centrality',
					'weighted_load_centrality',
					'weighted_subgraph_centrality',
					'weighted_harmonic_centrality',
					'code']]
# # corr_matrix["code"].sort_values()


# In[66]:


data


# In[67]:


train_set, test_set = train_test_split(data, test_size= 0.2, train_size= 0.8, random_state= 102)
cleaned_train = train_set.drop(["code"], axis=1)
cleaned_train.head()
train_pred = train_set["code"].copy()
cleaned_test = test_set.drop(["code"], axis=1)
test_pred = test_set["code"].copy()
corr_matrix = data.corr()
test_set['code'].value_counts(100)


# In[68]:



# Data Cleaning
# This dataset contains only integer values, and there are no missing values. Therefore, no Data Cleaning is required.


# In[69]:


# Feature Scaling
scaler = StandardScaler()
scaled_train = scaler.fit_transform(cleaned_train)
mean = scaler.mean_
variance = scaler.var_
scaled_test = (cleaned_test - mean) / variance
X_train = scaled_train;
y_train = train_pred;
X_test = scaled_test;
y_test = test_pred;


# # 5. Select and train a model

# ### Linear SVM 

# In[70]:


from sklearn.svm import LinearSVC

# Training your svm here
lin_svm_clf = LinearSVC(C=1, loss="hinge", random_state=42)
lin_svm_clf.fit(X_train, y_train)

# Testing your svm here
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier

y_train_pred = lin_svm_clf.predict(X_test)
confusion_matrix(y_test, y_train_pred)


# In[71]:


#Metrics for Linear SVM:
y_train_pred = lin_svm_clf.predict(X_test)
confusion_matrix(y_test, y_train_pred)


# ### Kernelized SVM with Gaussian RBF

# In[72]:


from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform


# In[73]:


#First, Randomly try some arbitrary Gamma and C values
#and report the performance in metrics from section 2
rbf_kernel_svm_clf1 = Pipeline([
        ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=0.0001, C=10,probability=True))
    ])
rbf_kernel_svm_clf1.fit(X_train, y_train)
# Run on Test Set
y_train_pred = rbf_kernel_svm_clf1.predict(X_test)
confusion_matrix(y_test, y_train_pred)
# print ("gamma=0.0001, C=10")
# print ("Accuracy")
# print(accuracy_score(y_test, y_train_pred))
# # Precision
# print ("Precision")
# print(precision_score(y_test, y_train_pred,average=None))
# # Recall
# print ("Recall")
# print(recall_score(y_test, y_train_pred,average=None))
# # F-Score
# print ("F-Score")
# print(f1_score(y_test, y_train_pred,average=None))
# print("")

rbf_kernel_svm_clf2 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.0001, C=1000,probability=True))
    ])
rbf_kernel_svm_clf2.fit(X_train, y_train)
# Run on Test Set
y_train_pred = rbf_kernel_svm_clf2.predict(X_test)
confusion_matrix(y_test, y_train_pred)
# Run SVM (RBF)
rbf_kernel_svm_clf3 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.01, C=10,probability=True))
    ])
rbf_kernel_svm_clf3.fit(X_train, y_train)
# Run on Test Set
y_train_pred = rbf_kernel_svm_clf3.predict(X_test)
confusion_matrix(y_test, y_train_pred)
#print ("gamma=0.01, C=10")
rbf_kernel_svm_clf4 = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.01, C=1000,probability=True))
    ])
rbf_kernel_svm_clf4.fit(X_train, y_train)
# Run on Test Set
y_train_pred = rbf_kernel_svm_clf1.predict(X_test)
confusion_matrix(y_test, y_train_pred)


# In[74]:


rbf_kernel_svm_clf3.fit(X_train, y_train)
# Run on Test Set
y_train_pred = rbf_kernel_svm_clf3.predict(X_test)
confusion_matrix(y_test, y_train_pred)


# ## Voting Classfiers

# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


# In[76]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42,probability=True)


# ## Bagging ensembles

# In[77]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=10, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)


# In[78]:


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)


# ## Random Forests

# In[79]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)


# In[80]:


bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)


# In[81]:


rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)


# ## AdaBoost

# In[82]:


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)


# In[83]:


y_pred = ada_clf.predict(X_test)


# ## Gradient Boosting

# In[84]:


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X_train, y_train)


# In[85]:


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rbf', rbf_kernel_svm_clf3), ('svc', svm_clf)],voting='soft')
voting_clf.fit(X_train, y_train)


# In[86]:


test_tweets = pd.read_csv(os.path.join(tweets,'%s_test_interaction.csv') % temp_user)


# In[87]:


test_tweets.status = pd.Categorical(test_tweets.status)
test_tweets['code'] = test_tweets.status.cat.codes
test_tweets['screen_name'] = test_tweets['screen_name'].astype('category').cat.codes
test_tweets['truncated'] = test_tweets['truncated'].astype('category').cat.codes
test_tweets['entities.hashtags'] = test_tweets['entities.hashtags'].astype('category').cat.codes
test_tweets['entities.user_mentions'] = test_tweets['entities.user_mentions'].astype('category').cat.codes
test_tweets['entities.symbols'] = test_tweets['entities.symbols'].astype('category').cat.codes
test_tweets['bool(entities.polls) '] = test_tweets['bool(entities.polls)'].astype('category').cat.codes
test_tweets['origin_user'] = test_tweets['origin_user'].astype('category').cat.codes
test_tweets['origin_entities_hashtags'] = test_tweets['origin_entities_hashtags'].astype('category').cat.codes
test_tweets['origin_entities_user_mentions'] = test_tweets['origin_entities_user_mentions'].astype('category').cat.codes
test_tweets['origin_entities_symbols'] = test_tweets['origin_entities_symbols'].astype('category').cat.codes
test_tweets['origin_entities_polls'] = test_tweets['origin_entities_polls'].astype('category').cat.codes

test_tweets['origin_status'] = test_tweets['origin_status'].astype('category').cat.codes

test_tweets = test_tweets.drop(['node','status'], axis=1)
#test_tweets.dtypes
test_tweets = test_tweets.select_dtypes(exclude=['object'])
test_tweets = test_tweets.astype(float)
test_tweets.dtypes


# In[88]:


# Data Splitting
data2 = test_tweets
# Data Discovery
data2 = data2.fillna(0)

data2 = data2[[		'screen_name',
					'len(user.description)',
					'user.followers_count',
					'user.friends_count',
					'user.listed_count',
					'user.statuses_count',
					'user.favourites_count',
					'retweet_count',
					'favorite_count',
					'origin_user',
					'origin_user_len_description',
					'origin_user_followers_count',
					'origin_user_friends_count',
					'origin_user_listed_count',
					'origin_user_statuses_count',
					'origin_user_favourites_count',
					'origin_retweet_count',
					'origin_favorite_count',
					'origin_status',
					'min_interval',

					'degree_centrality',
					'eigenvector_centrality',
					'closeness_centrality',
					'betweenness_centrality',
					'load_centrality',
					'subgraph_centrality',
					'harmonic_centrality',
					'weighted_degree_centrality',
					'weighted_eigenvector_centrality',
					'weighted_closeness_centrality',
					'weighted_betweenness_centrality',
					'weighted_load_centrality',
					'weighted_subgraph_centrality',
					'weighted_harmonic_centrality',
					'code']]
# data.corr()
# corr_matrix["code"].sort_values()


# In[89]:


cleaned_test2 = data2.drop(["code"], axis=1)
test_pred2 = data2["code"].copy()
data2['code'].value_counts(100)


# In[90]:


# Feature Scaling
scaled_test2 = (cleaned_test2 - mean) / variance
X_test2 = scaled_test2;
y_test2 = test_pred2;


# In[96]:


best_accuracy = 0
best_clf = ''

for clf in (log_clf, rnd_clf, svm_clf, voting_clf,bag_clf,tree_clf,ada_clf,rbf_kernel_svm_clf3):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_accuracy:
        best_clf = (clf.__class__.__name__)
        best_accuracy = acc

# for clf in (log_clf, rnd_clf, svm_clf, voting_clf,bag_clf,tree_clf,ada_clf,rbf_kernel_svm_clf3):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__)
#     # Accuracy
#     print ("Accuracy")
#     acc = accuracy_score(y_test, y_pred)
#     print(acc)
#     if acc > best_accuracy:
#         best_clf = (clf.__class__.__name__)
#         best_accuracy = acc
#     # Precision
#     print ("Precision")
#     print(precision_score(y_test, y_pred,average = None))
#     # Recall
#     print ("Recall")
#     print(recall_score(y_test, y_pred,average = None))
#     # F-Score
#     print ("F-Score")
#     print(f1_score(y_test, y_pred,average = None))
#     print(np.array(test_set['code']))
#     print(y_pred)
#     try:
#         print(clf.predict_proba(X_test))
#     except:
#         pass
    
# print (best_clf)


# In[95]:


ids = pd.read_csv(os.path.join(tweets,'%s_test_interaction.csv') % temp_user)
if not path.exists(os.path.join(tweets,'%s_tweets_probability.csv' % temp_user)):
    with open(('%s_tweets_probability.csv' % temp_user), 'a') as f:
        writer = csv.writer(f)
        writer.writerows([['ID','like','no action','quote','reply','retweet']])

c = [1,2]

dd = []
for clf in (log_clf, rnd_clf, svm_clf, voting_clf,bag_clf,tree_clf,ada_clf,rbf_kernel_svm_clf3):
    if (clf.__class__.__name__ == best_clf):
        clf.fit(X_train, y_train)
        y_pred2 = clf.predict(X_test2)
        dd = clf.predict_proba(X_test2)
        
b=ids['node'].values
dd = np.column_stack([b,dd])

with open(('%s_tweets_probability.csv' % temp_user), 'a') as f:
    writer = csv.writer(f)
    writer.writerows(dd)

