import pandas as pd
import numpy as np
import glob
import time
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing, cross_validation, metrics, linear_model, cluster
from sklearn.cluster import Birch
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import AdaBoostRegressor as ABR
# from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import make_scorer


from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool 

from sklearn.preprocessing import MinMaxScaler 

import datetime
import math

from sklearn.feature_selection import RFECV
import itertools
import random
import string
import os
import networkx as nx
import json

# from lasagne.layers import DenseLayer
# from lasagne.layers import DropoutLayer
# from lasagne.layers import InputLayer
# from lasagne.nonlinearities import softmax
# from nolearn.lasagne import NeuralNet
# from lasagne.nonlinearities import sigmoid, tanh, rectify

import sys
from data_branch import BranchData
from evaluator_branch import BranchEvaluator
from sklearn.preprocessing import StandardScaler


# from matplotlib import pyplot

# from unbalanced_dataset import SMOTE

# np.random.seed(seed=999999) #16)
# random.seed(999)

import locale
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
locale.setlocale(locale.LC_ALL, 'usa')

n_proc = 30

# os.environ["R_USER"]="C:\Program Files\R\R-3.2.2"

data_location = 'D:/Users/u0107065/Documents/data'
# data_location = '/home/sandra/Documents/ChurnPrediction'

X_train = np.array(json.loads(open('X_train_t1.txt','r').read()))
X_test = np.array(json.loads(open('X_test_t1.txt','r').read()),dtype=np.float32)
Y_train = np.array(json.loads(open('Y_train_t1.txt','r').read()))
rev_branch_dict = json.loads(open('rev_branch_dict_t1.txt','r').read())	
userids_test = json.loads(open('userids_test_t1.txt','r').read())

rev_branch_dict = {int(k):int(v) for k,v in rev_branch_dict.items()}

dist_train = np.array(json.loads(open('dist_train_t1.txt','r').read()))
dist_test = np.array(json.loads(open('dist_test_t1.txt','r').read()))

act_branch_dist_train = np.array(json.loads(open('act_branch_dist_train.txt','r').read()))
act_branch_dist_test = np.array(json.loads(open('act_branch_dist_test.txt','r').read()))


for i in range(Y_train.shape[0]):
	selected = np.argsort(Y_train[i,:])[::-1]
	Y_train[i,selected[5:]] = 0

Y_train = normalize(Y_train, norm='l2', axis=1, copy=False)

# mmscaler = MinMaxScaler(feature_range=(-1, 1))
# mmscaler.fit(np.vstack((X_train,X_test)))
# X_train = mmscaler.transform(X_train)
# X_test = mmscaler.transform(X_test)



def convert_date(d, ref_date, default_val):
	rd = datetime.datetime.strptime(ref_date, '%Y-%m-%d').date()
	if d!='0':
		dd = datetime.datetime.strptime(d, '%Y-%m-%d').date()
		return (rd-dd).days+1
	else:
		return default_val
	

def regr(X, Y):
	l = InputLayer(shape=(None, X.shape[1]))
	l = DenseLayer(l, num_units=Y.shape[1]+100, nonlinearity=tanh)
	# l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
	l = DenseLayer(l, num_units=Y.shape[1]+50, nonlinearity=tanh)
	# l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=None)
	net = NeuralNet(l, regression=True, update_learning_rate=0.1, verbose=1)
	net.fit(X, Y)
	print(net.score(X, Y))
	return net


def predict(net, X_test):
	return net.predict(X_test)


def small_conversion(x):
	if x == 'a':
		return 1
	elif x == 'b':
		return 2
	elif x == 'c':
		return 3

def train_and_score(i):
	global X_train
	global X_test 
	global Y_train
	global dist_train
	global dist_test
	
	cl = GradientBoostingRegressor(n_estimators=100, loss='ls', learning_rate=0.1)
	# cl.fit(X_train,Y_train[:,i])
	# return cl.predict(X_test)	
	dist_from_target_branch_train = dist_train[:,i].reshape((len(dist_train[:,i]),1))  # dist from i-th branch
	X_train = np.hstack((X_train, dist_from_target_branch_train))
	ab_dist_train = act_branch_dist_train[:,i].reshape((len(act_branch_dist_train[:,i]),1))  # dist from i-th branch
	X_train = np.hstack((X_train, ab_dist_train))
	# mmscaler_train = StandardScaler()
	# X_train = mmscaler_train.fit_transform(X_train)

	cl.fit(X_train,Y_train[:,i])

	dist_from_target_branch_test = dist_test[:,i].reshape((len(dist_test[:,i]),1))  # dist from i-th branch
	X_test = np.hstack((X_test, dist_from_target_branch_test))
	ab_dist_test = act_branch_dist_test[:,i].reshape((len(act_branch_dist_test[:,i]),1))  # dist from i-th branch
	X_test = np.hstack((X_test, ab_dist_test))

	# mmscaler_test = StandardScaler()
	# X_test = mmscaler_test.fit_transform(X_test)

	return cl.predict(X_test)


# main program
if __name__=='__main__':

	pred = []

	pool = Pool(processes=n_proc)
	params = [i for i in range(323)]
	# params = [Y_train[:,i] for i in range(323)]
	it = pool.map(train_and_score, params)

	for res in it:
		pred.append(res.tolist())
	pool.terminate()

	pred = np.array(pred).T

	# for i in range(323):
	#   print('RF #', i)
	#   # cl = RFR(n_estimators=10, criterion='mse', n_jobs=-1, random_state=1234) # oob_score=False, 
	#   # cl = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=10)
	#   cl = ABR(n_estimators=50)
	#   # cl = LinearRegression(n_jobs=-1) # oob_score=False, 
	#   cl.fit(X_train,Y_train[:,i])
	#   pred[:,i] = cl.predict(X_test)

	#   # RF_list.append(cl)

	# # pred = dim_red.inverse_transform(pred)

   

	# dim_red = PCA(n_components=200) #, copy=True, whiten=False)[source]
	# X_train = dim_red.fit_transform(X_train)
	# X_test = dim_red.fit_transform(X_test)

	# mmscaler = MinMaxScaler(feature_range=(-1, 1))
	# mmscaler.fit(np.vstack((X_train,X_test)))
	# X_train = mmscaler.transform(X_train)
	# X_test = mmscaler.transform(X_test)

	# net = regr(X_train, Y_train)
	# pred = predict(net, X_test)


	res = pd.DataFrame(columns = ['#USER_ID','POI_ID','NUMBER_OF_VISITS'])
	u = []
	b = []
	num = []

	selected = np.argsort(pred,axis=1)[:,pred.shape[1]-5::]

	for i in range(selected.shape[0]):
		for j in selected[i]:
			if pred[i,j]!=0:
				u.append(userids_test[i])
				b.append(rev_branch_dict[j])
				num.append(pred[i,j])

	print('done')

	res['#USER_ID'] = list(map(int,u))
	res['POI_ID'] = b
	res['NUMBER_OF_VISITS'] = num 
	
	res.to_csv('pred.csv',delimiter=',',index=False)

	# ground_truth_train = ground_truth[ground_truth['USER_ID'].isin(u)]
	# ground_truth_train.to_csv('gt2014_train.csv',delimiter=',',index=False)
