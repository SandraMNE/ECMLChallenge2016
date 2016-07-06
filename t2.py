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
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier 
# from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import make_scorer


from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

from sklearn.preprocessing import MinMaxScaler 

import datetime
import math

from sklearn.feature_selection import RFECV
import itertools
import random
import string
import os
import networkx as nx

# from lasagne.layers import DenseLayer
# from lasagne.layers import DropoutLayer
# from lasagne.layers import InputLayer
# from lasagne.nonlinearities import softmax
# from nolearn.lasagne import NeuralNet
# from lasagne.nonlinearities import sigmoid, tanh

# from matplotlib import pyplot

# from unbalanced_dataset import SMOTE

# np.random.seed(seed=999999) #16)
random.seed(999)

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
# locale.setlocale(locale.LC_ALL, 'usa')



# os.environ["R_USER"]="C:\Program Files\R\R-3.2.2"

data_location = 'D:/Users/u0107065/Documents/data'
# data_location = '/home/sandra/Documents/ChurnPrediction'



def convert_date(d, ref_date, default_val):
	# print (d,type(d))
	rd = datetime.datetime.strptime(ref_date, '%Y-%m-%d').date()
	if (d!='0' and d!='f'):
		dd = datetime.datetime.strptime(d, '%Y-%m-%d').date()
		return (rd-dd).days+1
	else:
		return default_val
 
 
def small_conversion(x):
	if x == 'a':
		return 1
	elif x == 'b':
		return 2
	elif x == 'c':
		return 3

# def regr(X, Y):
#     l = InputLayer(shape=(None, X.shape[1]))
#     l = DenseLayer(l, num_units=X.shape[1], nonlinearity=tanh) #tanh, sigmoid
#     l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
#     l = DenseLayer(l, num_units=X.shape[1], nonlinearity=None)
#     l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
#     l = DenseLayer(l, num_units=1, nonlinearity=sigmoid)
#     # l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
#     net = NeuralNet(l, regression=True, update_learning_rate=0.01, verbose=1)
#     net.fit(X, Y)
#     print(net.score(X, Y))
#     return net

def regr(X, Y):
	l = InputLayer(shape=(None, X.shape[1]))
	l = DenseLayer(l, num_units=X.shape[1], nonlinearity=tanh) #tanh, sigmoid
	l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
	l = DenseLayer(l, num_units=1, nonlinearity=sigmoid)
	# l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
	net = NeuralNet(l, regression=True, update_learning_rate=0.01, verbose=1)
	net.fit(X, Y)
	print(net.score(X, Y))
	return net


def predict(net, X_test):
	return net.predict(X_test)


def prepare_for_ict(orig_df):
 
	intercall_dict = {}
 
	for r in orig_df.values:
		# print(r,r[0],r[1])
		# we skip first as it's header
		if r[0] in intercall_dict:
			intercall_dict[r[0]].append(r[1])
		else:
			intercall_dict[r[0]] = [r[1]]
 
	intercall_df = pd.DataFrame(columns=['USER_ID','DATE'])
	intercall_df['USER_ID'] = intercall_dict.keys()
	intercall_df['DATE'] = intercall_dict.values()
 
	return intercall_df
 
 
def calc_ict(val_array):
 
	num_days = 31+28+31+30+31+30 # first half of the year 
	res = []
	if val_array is None or val_array==['0'] or val_array==[170]:
		res.append(1)
	elif len(val_array) == 1:
		res.append(num_days - int(val_array[0]) + 1)
	else:
		start_val = int(val_array[0])
		res.append(num_days-start_val+1)
		for i in range(1,len(val_array)):
			res.append(start_val-int(val_array[i])+1)
			start_val = int(val_array[i])
		res.append(start_val-1)  
	return res
 
 
def calc_cl(val_array):
	H = 1
	num_days = 31+28+31+30+31+30 # first half of the year 
 
	for i in range(0,len(val_array)):
		q = val_array[i]/num_days
		if q!=0:
			H += (q * np.log(q))/np.log(len(val_array)+1)
 
	return H    



# main program
if __name__=='__main__': 

	# one hot ind_encoding
	# standardize
	# create_matrix

	start_time = time.time()

	user2014 = pd.read_csv('users_2014.csv',sep=',')
	user2014 = user2014.replace('-','0')
	user2014.rename(columns={'LOC_CAT':'USER_LOC_CAT'},inplace=True)
	# target for those having bought credit card in 1st half of 2014 will anyways be 0 so we don't care
	data2014 = pd.read_csv('train_2014.csv',sep=',')
	data2014 = data2014.replace('-', '0')
	# removing branch data since it's not available in test set
	data2014 = data2014[data2014['CHANNEL']!='b']
	# we select only first half of the year
	data2014['DATE'] = data2014['DATE'].apply(lambda x: convert_date(x, '2014-07-01', 170))
	data2014 = data2014[data2014['DATE']>0]
	data2014 = pd.merge(user2014, data2014, how='left', on='USER_ID')
	data2014.fillna('f', inplace=True)
	data2014 = data2014[['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']]


	user2015 = pd.read_csv('users_2015.csv',sep=',')
	user2015 = user2015.replace('-','0')
	user2015.rename(columns={'LOC_CAT':'USER_LOC_CAT'},inplace=True)
	data2015 = pd.read_csv('train_2015.csv',sep=',')
	data2015 = data2015.replace('-','0')
	# removing branch data since it's not available in test set
	data2015 = data2015[data2015['CHANNEL']!='b']
	# we select only first half of the year
	data2015['DATE'] = data2015['DATE'].apply(lambda x: convert_date(x, '2015-07-01', 170))
	data2015 = data2015[data2015['DATE']>0]
	data2015 = pd.merge(user2015, data2015, how='left', on='USER_ID')
	data2015.fillna('f', inplace=True)
	data2015 = data2015[['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']]

	data = pd.concat([data2014,data2015])

	for f in ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']:
		data[f] = data[f].apply(str)
		le = preprocessing.LabelEncoder()
		data[f] = le.fit_transform(np.reshape(data[[f]].values,(len(data[[f]]),)))


	ohe = OneHotEncoder(categorical_features='all')
	ohe.fit(data.values)


#######################



	user2014 = pd.read_csv('users_2014.csv',sep=',')
	user2014 = user2014.replace('-','0')
	user2014.rename(columns={'LOC_CAT':'USER_LOC_CAT'},inplace=True)
	user2014.drop(['C201407','C201408','C201409','C201410','C201411','C201412','W201407','W201408','W201409','W201410','W201411','W201412'],1,inplace=True)

	# we define ground truth
	# user2014['TARGET_TASK_2'] = user2014['TARGET_TASK_2'].apply(lambda x: 1 if x in ['2014.07.31','2014.08.31','2014.09.30','2014.10.31','2014.11.30','2014.12.31','2015.01.31','2015.02.28','2015.03.31','2015.04.30','2015.05.31','2015.06.30','2015.07.31','2015.08.31','2015.09.30','2015.10.31','2015.11.30','2015.12.31'] else 0)
	user2014['TARGET_TASK_2'] = user2014['TARGET_TASK_2'].apply(lambda x: 1 if x in ['2014.07.31','2014.08.31','2014.09.30','2014.10.31','2014.11.30','2014.12.31'] else 0)
	# target for those having bought credit card in 1st half of 2014 will anyways be 0 so we don't care
	# user2014 = user2014[~user2014['TARGET_TASK_2'].isin(['2014.01.31', '2014.02.28', '2014.03.31', '2014.04.30','2014.05.31','2014.06.30'])]


	data2014 = pd.read_csv('train_2014.csv',sep=',')
	data2014 = data2014.replace('-', '0')
	# data2014 = data2014.replace(r'-', np.nan, regex=True)
	# data2014.fillna('0', inplace=True)

	# removing branch data since it's not available in test set
	data2014 = data2014[data2014['CHANNEL']!='b']

	# we convert date in number of days till '2014-07-01' and those being null into fixed value 170 
	# now everything that is before 01-07 will be positive and everything after negative
	data2014['DATE'] = data2014['DATE'].apply(lambda x: convert_date(x, '2014-07-01', 170))

	# we select only first half of the year
	data2014 = data2014[data2014['DATE']>0]

	data2014 = pd.merge(user2014, data2014, how='left', on='USER_ID')
	# just temporarily dropping
	data2014.drop(['TARGET_TASK_2'],1,inplace=True)


	data2014.fillna('f', inplace=True)
	data2014['DATE'] = data2014['DATE'].apply(lambda x: 170 if x=='f' else x)
 
	data2014['GEO_X'] = data2014['GEO_X'].apply(lambda x: 0.0 if x=='f' else x)
	data2014['GEO_Y'] = data2014['GEO_Y'].apply(lambda x: 0.0 if x=='f' else x)
 
	# calculating clumpiness wrongly named as ict :)
	ict_calc_df = data2014[['USER_ID','DATE']]
	ict_calc_df = prepare_for_ict(ict_calc_df)
	ict_calc_df['ICT'] = ict_calc_df['DATE'].apply(lambda x: calc_ict(x))
	ict_calc_df['mean_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.mean(x))
	ict_calc_df['std_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.std(x))
	ict_calc_df['C'] = ict_calc_df['ICT'].apply(lambda x: calc_cl(x))
	ict_calc_df.drop(['DATE','ICT'],1,inplace=True)

	add_counters = data2014[['USER_ID','CHANNEL','AMT_CAT','CARD_CAT','DATE']]
	add_counters['pos_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='p' else 0)
	add_counters['webshop_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='n' else 0)
	add_counters['credit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='c' else 0)
	add_counters['debit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='d' else 0)
	add_counters.drop(['CHANNEL','CARD_CAT'],1,inplace=True)
	add_counters['AMT_CAT'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['num_diff_amounts'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['last_act_till_feb'] = add_counters['DATE'].apply(lambda x: 1 if x>122 else 0)
	add_counters['last_act_after_feb'] = add_counters['DATE'].apply(lambda x: 1 if x<=122 else 0)
	add_counters.rename(columns={'AMT_CAT':'max_amount','DATE':'recency'},inplace=True)
	add_counters = add_counters.groupby(['USER_ID']).agg({'pos_cnt': np.sum,'webshop_cnt': np.sum,'credit_card_cnt': np.sum,'debit_card_cnt': np.sum, 'max_amount': np.max, 'num_diff_amounts':pd.Series.nunique,'recency':np.min})
	add_counters = pd.DataFrame(add_counters.reset_index())
	add_counters['total_num_act'] = add_counters['pos_cnt'] + add_counters['webshop_cnt']
	add_counters.fillna('0', inplace=True)
 
	data2014 = pd.merge(data2014, add_counters, how='inner', on='USER_ID')

	# we calc user-activity distance as Euclidean
	data2014['GEO_X'] = data2014['GEO_X'].astype(float)
	data2014['GEO_Y'] = data2014['GEO_Y'].astype(float)
	data2014['LOC_GEO_X'] = data2014['LOC_GEO_X'].astype(float)
	data2014['LOC_GEO_Y'] = data2014['LOC_GEO_Y'].astype(float)

	# data2014['act_dist'] = data2014.apply(lambda row: math.sqrt((row['LOC_GEO_X']-row['GEO_X'])**2+(row['LOC_GEO_Y']-row['GEO_Y'])**2), 1)
	# data2014.drop(['LOC_GEO_X','GEO_X','GEO_Y','LOC_GEO_Y'],1,inplace=True)

	# calc log2 of date
	# data2014['DATE'] = data2014['DATE'].apply(lambda x: 1/np.log2(x))
	data2014['DATE'] = data2014['DATE'].apply(lambda x: 1)

	# getting rid of branch id
	data2014.drop(['POI_ID'],1,inplace=True)

	cat_vars = ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']
	other_vars = [col for col in data2014.columns if col not in cat_vars]

	data2014_cat = data2014[cat_vars]

	for f in cat_vars:
		data2014_cat[f] = data2014_cat[f].apply(str)
		le = preprocessing.LabelEncoder()
		data2014_cat[f] = le.fit_transform(np.reshape(data2014_cat[[f]].values,(len(data2014_cat[[f]]),)))

	X_cat = ohe.transform(data2014_cat.values)
	X_other = data2014[other_vars].values

	X = np.hstack((X_other,X_cat.todense()))

	# multiplying with weights
	X = np.multiply(X,np.reshape(data2014['DATE'].values,(X.shape[0],))[:,np.newaxis])
	
	cols = other_vars + ['c'+str(i) for i in range(X_cat.shape[1])]
	train_df = pd.DataFrame(X,columns=cols)


	# # we do label encoding for categorical vars
	# for f in ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']:
	# 	data2014[f] = data2014[f].apply(str)
	# 	le = preprocessing.LabelEncoder()
	# 	data2014[f] = le.fit_transform(np.reshape(data2014[[f]].values,(len(data2014[[f]]),)))

	# d1 = data2014[['USER_ID','DATE']] 	
	# d2 = data2014.drop(['USER_ID','DATE'],1)

	# X = d2.values
	# # now we apply one hot encoding for all these + channel
	# ind_encoding = [0,1,2,18,19,20,21,22,23]  # corresponding to ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT'] I have checked this on my own ! ATTENTION!!
	# # ohe = OneHotEncoder(categorical_features=ind_encoding)
	# #ohe = OneHotEncoder(categorical_features='all')
	# X = ohe.fit_transform(X)

	# # multiplying with weights
	# X = np.multiply(X.todense(),np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	# d1 = np.hstack((d1.values,X))
	# cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	# train_df = pd.DataFrame(d1,columns=cols)

	train_df = train_df.groupby(['USER_ID']).sum()
	train_df = pd.DataFrame(train_df.reset_index())
	d1 = train_df[['USER_ID','DATE']]
	d2 = train_df.drop(['USER_ID','DATE'],1)

	X = np.array(d2.values,dtype=np.float32)
	# calculating weighted average
	X = np.divide(X,np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	d1 = np.hstack((d1.values,X))
	cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	train_df = pd.DataFrame(d1,columns=cols)

	train_df.drop(['DATE'],1,inplace=True)
	train_df = pd.DataFrame(train_df.reset_index())


	# train_df.drop(['level_0'],1,inplace=True)

	bank_info = pd.read_csv('bank_info.csv',sep=',')
	bank_info.rename(columns={'GEO_X':'GEO_X_bank','GEO_Y':'GEO_Y_bank'},inplace=True)
	bank_info = bank_info[['GEO_X_bank','GEO_Y_bank']]
	bank_geo_matrix = bank_info.values

	user_geo = user2014[['USER_ID','LOC_GEO_X','LOC_GEO_Y']]
	train_df = pd.merge(train_df, user_geo, how='left',on='USER_ID')
	user_geo_matrix = train_df[['LOC_GEO_X','LOC_GEO_Y']].values


	dist_train = cdist(user_geo_matrix, bank_geo_matrix, 'euclidean')
	aux = dist_train.min(axis=1)
	d1 = np.hstack((train_df.values, aux.reshape((len(aux),1))))
	cols = list(train_df.columns) + ['min_branch_geo_dist']
	# d1 = np.hstack((train_df.values,dist_train))
	# cols = list(train_df.columns) + ['geo_dist'+str(i) for i in range(dist_train.shape[1])]
	train_df = pd.DataFrame(d1,columns=cols)
	train_df = pd.merge(train_df, ict_calc_df, how='left', on='USER_ID')
	train_df.fillna(0, inplace=True)

	userids = train_df['USER_ID'] # list(map(int,train_df['USER_ID'].tolist()))

	train_df = pd.merge(train_df, user2014[['USER_ID','TARGET_TASK_2']], how='inner', on='USER_ID')
	Y_train = train_df['TARGET_TASK_2'].values

	train_df.drop(['index','USER_ID','TARGET_TASK_2'],1,inplace=True)

	X_train = train_df.values

	###########################################################

	user2015 = pd.read_csv('users_2015.csv',sep=',')
	user2015 = user2015.replace('-','0')
	user2015.rename(columns={'LOC_CAT':'USER_LOC_CAT'},inplace=True)
	
	data2015 = pd.read_csv('train_2015.csv',sep=',')
	data2015 = data2015.replace('-','0')

	# removing branch data since it's not available in test set
	data2015 = data2015[data2015['CHANNEL']!='b']
 
	# we convert date in number of days till '2015-07-01' and those being null into fixed value 170 
	# now everything that is before 01-07 will be positive and everything after negative
	data2015['DATE'] = data2015['DATE'].apply(lambda x: convert_date(x, '2015-07-01', 170))
 
	# we select only first half of the year
	data2015 = data2015[data2015['DATE']>0]
 
	data2015 = pd.merge(user2015, data2015, how='left', on='USER_ID')
	data2015.fillna('f', inplace=True)
	data2015['DATE'] = data2015['DATE'].apply(lambda x: 170 if x=='f' else x)
 
	data2015['GEO_X'] = data2015['GEO_X'].apply(lambda x: 0.0 if x=='f' else x)
	data2015['GEO_Y'] = data2015['GEO_Y'].apply(lambda x: 0.0 if x=='f' else x)
 
	# calculating clumpiness wrongly named as ict :)
	ict_calc_df = data2015[['USER_ID','DATE']]
	ict_calc_df = prepare_for_ict(ict_calc_df)
	ict_calc_df['ICT'] = ict_calc_df['DATE'].apply(lambda x: calc_ict(x))
	ict_calc_df['mean_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.mean(x))
	ict_calc_df['std_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.std(x))
	ict_calc_df['C'] = ict_calc_df['ICT'].apply(lambda x: calc_cl(x))
	ict_calc_df.drop(['DATE','ICT'],1,inplace=True)


	add_counters = data2015[['USER_ID','CHANNEL','AMT_CAT','CARD_CAT','DATE']]
	add_counters['pos_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='p' else 0)
	add_counters['webshop_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='n' else 0)
	add_counters['credit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='c' else 0)
	add_counters['debit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='d' else 0)
	add_counters.drop(['CHANNEL','CARD_CAT'],1,inplace=True)
	add_counters['AMT_CAT'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['num_diff_amounts'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['last_act_till_feb'] = add_counters['DATE'].apply(lambda x: 1 if x>122 else 0)
	add_counters['last_act_after_feb'] = add_counters['DATE'].apply(lambda x: 1 if x<=122 else 0)
	add_counters.rename(columns={'AMT_CAT':'max_amount','DATE':'recency'},inplace=True)
	add_counters = add_counters.groupby(['USER_ID']).agg({'pos_cnt': np.sum,'webshop_cnt': np.sum,'credit_card_cnt': np.sum,'debit_card_cnt': np.sum, 'max_amount': np.max, 'num_diff_amounts':pd.Series.nunique,'recency':np.min})
	add_counters = pd.DataFrame(add_counters.reset_index())
	add_counters['total_num_act'] = add_counters['pos_cnt'] + add_counters['webshop_cnt']
	add_counters.fillna('0', inplace=True)
 
	data2015 = pd.merge(data2015, add_counters, how='inner', on='USER_ID')

	# we calc user-activity distance as Euclidean
	data2015['GEO_X'] = data2015['GEO_X'].astype(float)
	data2015['GEO_Y'] = data2015['GEO_Y'].astype(float)
	data2015['LOC_GEO_X'] = data2015['LOC_GEO_X'].astype(float)
	data2015['LOC_GEO_Y'] = data2015['LOC_GEO_Y'].astype(float)

	# data2015['act_dist'] = data2015.apply(lambda row: math.sqrt((row['LOC_GEO_X']-row['GEO_X'])**2+(row['LOC_GEO_Y']-row['GEO_Y'])**2), 1)
	# data2015.drop(['LOC_GEO_X','GEO_X','GEO_Y','LOC_GEO_Y'],1,inplace=True)


	data2015['DATE'] = data2015['DATE'].apply(lambda x: 1)

	# getting rid of branch id
	data2015.drop(['POI_ID'],1,inplace=True)

	cat_vars = ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']
	other_vars = [col for col in data2015.columns if col not in cat_vars]

	data2015_cat = data2015[cat_vars]

	for f in cat_vars:
		data2015_cat[f] = data2015_cat[f].apply(str)
		le = preprocessing.LabelEncoder()
		data2015_cat[f] = le.fit_transform(np.reshape(data2015_cat[[f]].values,(len(data2015_cat[[f]]),)))

	X_cat = ohe.transform(data2015_cat.values)
	X_other = data2015[other_vars].values

	X = np.hstack((X_other,X_cat.todense()))

	# multiplying with weights
	X = np.multiply(X,np.reshape(data2015['DATE'].values,(X.shape[0],))[:,np.newaxis])
	
	cols = other_vars + ['c'+str(i) for i in range(X_cat.shape[1])]
	test_df = pd.DataFrame(X,columns=cols)


	# # we do label encoding for categorical vars
	# for f in ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']:
	# 	data2015[f] = data2015[f].apply(str)
	# 	le = preprocessing.LabelEncoder()
	# 	data2015[f] = le.fit_transform(np.reshape(data2015[[f]].values,(len(data2015[[f]]),)))

	# d1 = data2015[['USER_ID','DATE']] 	
	# d2 = data2015.drop(['USER_ID','DATE'],1)

	# X = d2.values
	# # now we apply one hot encoding for all these + channel
	# # ind_encoding = [0,1,2,18,19,20,21,22,23]  # corresponding to ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT'] I have checked this on my own ! ATTENTION!!
	# # ohe = OneHotEncoder(categorical_features=ind_encoding)
	# #ohe = OneHotEncoder(categorical_features='all')
	# X = ohe.fit_transform(X)

	# # multiplying with weights
	# X = np.multiply(X.todense(),np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	# d1 = np.hstack((d1.values,X))
	# cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	# test_df = pd.DataFrame(d1,columns=cols)

	test_df = test_df.groupby(['USER_ID']).sum()
	test_df = pd.DataFrame(test_df.reset_index())
	d1 = test_df[['USER_ID','DATE']]
	d2 = test_df.drop(['USER_ID','DATE'],1)

	# X = d2.values
	X = np.array(d2.values,dtype=np.float32)
	# calculating weighted average
	X = np.divide(X,np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	d1 = np.hstack((d1.values,X))
	cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	test_df = pd.DataFrame(d1,columns=cols)

	test_df.drop(['DATE'],1,inplace=True)
	test_df = pd.DataFrame(test_df.reset_index())
 
	user_geo = user2015[['USER_ID','LOC_GEO_X','LOC_GEO_Y']]
	test_df = pd.merge(test_df, user_geo, how='left', on='USER_ID')

	test_user_geo_matrix = test_df[['LOC_GEO_X','LOC_GEO_Y']].values

	dist_test = cdist(test_user_geo_matrix, bank_geo_matrix, 'euclidean')
	aux = dist_test.min(axis=1)
	d1 = np.hstack((test_df.values, aux.reshape((len(aux),1))))
	cols = list(test_df.columns) + ['min_branch_geo_dist']
	# d1 = np.hstack((test_df.values,dist))
	# cols = list(test_df.columns) + ['geo_dist'+str(i) for i in range(dist.shape[1])]
	test_df = pd.DataFrame(d1,columns=cols)
	# test_df.drop(['geo_dist'+str(i) for i in range(dist.shape[1])], 1, inplace=True) 
	test_df = pd.merge(test_df, ict_calc_df, how='left', on='USER_ID')
	test_df.fillna(0, inplace=True)
 
	userids_test = test_df['USER_ID'] #np.array(map(int,test_df['USER_ID'].tolist()))
	test_df.drop(['index','USER_ID'],1,inplace=True)

	X_test = test_df.values

	###########################################################


	# # dim_red = PCA(n_components=300) #, copy=True, whiten=False)[source]
	# # Y_train = dim_red.fit_transform(Y_train)
	# # print('variance ratio', np.sum(dim_red.explained_variance_ratio_))


	# # RF_list = []
	# pred = np.zeros((X_test.shape[0],323))

	# for i in range(323):
	# 	print('RF #', i)
	# 	# cl = RFR(n_estimators=10, criterion='mse', n_jobs=-1, random_state=1234) # oob_score=False, 
	# 	cl = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=10)
	# 	# cl = LinearRegression(n_jobs=-1) # oob_score=False, 
	# 	cl.fit(X_train,Y_train[:,i])
	# 	pred[:,i] = cl.predict(X_test)

	# 	# RF_list.append(cl)

	# # pred = dim_red.inverse_transform(pred)

	pred = np.zeros(X_test.shape[0])


	print('cl')
	# cl = RFC(n_estimators=100, criterion='gini', n_jobs=-1, random_state=1234) # oob_score=False, 
	# cl.fit(X_train,Y_train)
	# pred += cl.predict_proba(X_test)[:,1]
	# # cl = GradientBoostingRegressor(n_estimators=100, loss='ls', learning_rate=0.1)
	# cl = AdaBoostClassifier(n_estimators=500, learning_rate=1.0, algorithm='SAMME.R', random_state=1234)
	# cl.fit(X_train,Y_train)
	# pred += cl.predict_proba(X_test)[:,1]
	cl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1) #, loss='ls')
	cl.fit(X_train,Y_train)
	pred += cl.predict_proba(X_test)[:,1]
	# 	# cl = ABR(n_estimators=50)
	# 	# cl.fit(X_train,Y_train[:,i])
	# 	# pred[:,i] += cl.predict(X_test)
	# pred /= 2.0



	# mmscaler = MinMaxScaler(feature_range=(-1, 1))
	# mmscaler.fit(np.vstack((X_train,X_test)))
	# X_train = mmscaler.transform(X_train)
	# X_test = mmscaler.transform(X_test)

	# dim_red = PCA(n_components=150) #, copy=True, whiten=False)[source]
 #    X_train = dim_red.fit_transform(X_train)
 #    X_test = dim_red.fit_transform(X_test)


	# net = regr(X_train, Y_train)
	# pred = predict(net, X_test)

	# cl = RF(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=10, oob_score=False, n_jobs=5, random_state=1234) #, class_weight=None)
	# rf = cl.fit(X_train, Y_train)
	# Y_pred = rf.predict_proba(X_test)[:,1]

	res = pd.DataFrame(columns=['#USER_ID','SCORE'])
	u = []
	s = []

	for i in range(X_test.shape[0]):
		u.append(userids_test[i]) #user2014_df[i])
		# s.append(Y_pred[i])
		# s.append(pred[i][0])
		s.append(pred[i])


	print('done')

	res['#USER_ID'] = u  
	res['SCORE'] = s

	res['#USER_ID']=res['#USER_ID'].astype(int)
	
	res.to_csv('pred_t2.csv',delimiter=',',index=False)








