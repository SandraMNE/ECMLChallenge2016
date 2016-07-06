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

import json

# from lasagne.layers import DenseLayer
# from lasagne.layers import DropoutLayer
# from lasagne.layers import InputLayer
# from lasagne.nonlinearities import softmax
# from nolearn.lasagne import NeuralNet
# from lasagne.nonlinearities import sigmoid, tanh, rectify

# from matplotlib import pyplot

# from unbalanced_dataset import SMOTE

# np.random.seed(seed=999999) #16)
# random.seed(999)

import locale
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
locale.setlocale(locale.LC_ALL, 'usa')



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
# 	l = InputLayer(shape=(None, X.shape[1]))
# 	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=sigmoid)
# 	l = DropoutLayer(l, p=0.5, rescale=True)  # previous: p=0.5
# 	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=None)
# 	l = DropoutLayer(l, p=0.5, rescale=True)  # previous: p=0.5
# 	net = NeuralNet(l, regression=True, update_learning_rate=0.01, verbose=1)
# 	net.fit(X, Y)
# 	print(net.score(X, Y))
# 	return net

# def regr(X, Y):
# 	l = InputLayer(shape=(None, X.shape[1]))
# 	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=tanh)
# 	l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
# 	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=rectify)
# 	net = NeuralNet(l, regression=True, update_learning_rate=0.01, verbose=1)
# 	net.fit(X, Y)
# 	print(net.score(X, Y))
# 	return net

def regr(X, Y):
	l = InputLayer(shape=(None, X.shape[1]))
	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=tanh)
	l = DropoutLayer(l, p=0.3, rescale=True)  # previous: p=0.5
	l = DenseLayer(l, num_units=Y.shape[1], nonlinearity=None)
	net = NeuralNet(l, regression=True, update_learning_rate=0.01, verbose=1)
	net.fit(X, Y)
	print(net.score(X, Y))
	return net



def prepare_for_ict(orig_df):

	cols = orig_df.columns
 
	intercall_dict = {}
 
	for r in orig_df.values:
		# print(r,r[0],r[1])
		# we skip first as it's header
		if r[0] in intercall_dict:
			intercall_dict[r[0]].append(r[1])
		else:
			intercall_dict[r[0]] = [r[1]]
 
	intercall_df = pd.DataFrame(columns=cols)
	intercall_df[cols[0]] = intercall_dict.keys()
	intercall_df[cols[1]] = intercall_dict.values()
	# intercall_df = pd.DataFrame(columns=['USER_ID','DATE'])
	# intercall_df['USER_ID'] = intercall_dict.keys()
	# intercall_df['DATE'] = intercall_dict.values()
 
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



def train_and_score(i):
	# global X_train
	# global X_test 

	cl = GradientBoostingRegressor(n_estimators=100, loss='ls', learning_rate=0.1)
	cl.fit(X_train,Y_train)
	return cl.predict(X_test)
	



def predict(net, X_test):
	return net.predict(X_test)


def count_ups(val_array):
	res = 0.0
	for i in range(1,len(val_array)):
		if val_array[i]>val_array[i-1]:
			res+=1
	return res/len(val_array)


def count_downs(val_array):
	res = 0.0
	for i in range(1,len(val_array)):
		if val_array[i]<val_array[i-1]:
			res+=1
	return res/len(val_array)


def calc_act_branch_dist(x,y,branch_loc):
	# print ('1',type(x),type(y),type(branch_loc))

	x = np.float64(x)
	y = np.float64(y)
	GEO_X_bank = branch_loc['GEO_X_bank'].values[0]
	GEO_Y_bank = branch_loc['GEO_Y_bank'].values[0]

	# print('2',type(GEO_X_bank),type(GEO_Y_bank))
	return math.sqrt((x-GEO_X_bank)**2+(y-GEO_Y_bank)**2)



# main program
if __name__=='__main__': 

	# one hot ind_encoding
	# standardize
	# create_matrix

	start_time = time.time()

	bank_info = pd.read_csv('bank_info.csv',sep=',')
	bank_info.rename(columns={'GEO_X':'GEO_X_bank','GEO_Y':'GEO_Y_bank'},inplace=True)
	bank_info = bank_info[['GEO_X_bank','GEO_Y_bank']]
	bank_geo_matrix = bank_info.values


	user2014 = pd.read_csv('users_2014.csv',sep=',')
	user2014['AGE_CAT'] = user2014['AGE_CAT'].apply(lambda x: 'b' if x=='-' else x)
	user2014 = user2014.replace('-','0')
	user2014.rename(columns={'LOC_CAT':'USER_LOC_CAT'},inplace=True)
	user2014.drop(['TARGET_TASK_2'],1,inplace=True)
	user2014.drop(['C201407','C201408','C201409','C201410','C201411','C201412','W201407','W201408','W201409','W201410','W201411','W201412'],1,inplace=True)

	data2014 = pd.read_csv('train_2014.csv',sep=',')
	data2014 = data2014.replace('-','0')

	df_branch = data2014[data2014['CHANNEL']=='b']
	# for evaluation we use just second half of 2014
	df_branch['DATE'] = df_branch['DATE'].apply(lambda x: convert_date(x, '2014-07-01', 170))
	df_branch = df_branch[df_branch['DATE']<=0]

	# removing branch data since it's not available in test set
	data2014 = data2014[data2014['CHANNEL']!='b']

	# we convert date in number of days till '2014-07-01' and those being null into fixed value 170 
	# now everything that is before 01-07 will be positive and everything after negative
	data2014['DATE'] = data2014['DATE'].apply(lambda x: convert_date(x, '2014-07-01', 170))

	# we select only first half of the year
	data2014 = data2014[data2014['DATE']>0]

	data2014 = pd.merge(user2014, data2014, how='left', on='USER_ID')
	data2014['sum_C_cols'] = data2014['C201401']+data2014['C201402']+data2014['C201403']+data2014['C201404']+data2014['C201405']+data2014['C201406']
	data2014['sum_W_cols'] = data2014['W201401']+data2014['W201402']+data2014['W201403']+data2014['W201404']+data2014['W201405']+data2014['W201406']


	data2014.fillna('f', inplace=True)
	data2014['DATE'] = data2014['DATE'].apply(lambda x: 170 if x=='f' else x)

	data2014['GEO_X'] = data2014['GEO_X'].apply(lambda x: 0.0 if x=='f' else x)
	data2014['GEO_Y'] = data2014['GEO_Y'].apply(lambda x: 0.0 if x=='f' else x)


	data2014['sum_C_cols'] = data2014['sum_C_cols'].apply(lambda x: 0 if x=='f' else x)
	data2014['sum_W_cols'] = data2014['sum_W_cols'].apply(lambda x: 0 if x=='f' else x)

	data2014['GEO_X'] = data2014['GEO_X'].astype(float)
	data2014['GEO_Y'] = data2014['GEO_Y'].astype(float)
	act_geo = data2014[['USER_ID','GEO_X','GEO_Y']]
	act_geo = act_geo.groupby(['USER_ID']).mean()
	act_geo = pd.DataFrame(act_geo.reset_index())

	# act_branch_dist2014 = data2014[['USER_ID','GEO_X','GEO_Y']]

	# calculating clumpiness wrongly named as ict :)
	ict_calc_df = data2014[['USER_ID','DATE']]
	ict_calc_df = prepare_for_ict(ict_calc_df)
	ict_calc_df['ICT'] = ict_calc_df['DATE'].apply(lambda x: calc_ict(x))
	ict_calc_df['mean_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.mean(x))
	ict_calc_df['std_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.std(x))
	ict_calc_df['C'] = ict_calc_df['ICT'].apply(lambda x: calc_cl(x))
	ict_calc_df.drop(['DATE','ICT'],1,inplace=True)

	add_counters = data2014[['USER_ID','CHANNEL','AMT_CAT','CARD_CAT','DATE','LOC_CAT','TIME_CAT','MC_CAT']]
	add_counters['pos_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='p' else 0)
	add_counters['webshop_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='n' else 0)
	add_counters['credit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='c' else 0)
	add_counters['debit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='d' else 0)
	add_counters.drop(['CHANNEL','CARD_CAT'],1,inplace=True)
	add_counters['AMT_CAT'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['num_diff_amounts'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['last_act_till_feb'] = add_counters['DATE'].apply(lambda x: 1 if x>122 else 0)
	add_counters['last_act_after_feb'] = add_counters['DATE'].apply(lambda x: 1 if x<=122 else 0)
	add_counters['act_loc1'] = add_counters['LOC_CAT'].apply(lambda x: 1 if x=='a' else 0)
	add_counters['act_loc2'] = add_counters['LOC_CAT'].apply(lambda x: 2 if x=='b' else 0)
	add_counters['act_loc3'] = add_counters['LOC_CAT'].apply(lambda x: 3 if x=='c' else 0)
	add_counters['time_cat1'] = add_counters['TIME_CAT'].apply(lambda x: 1 if x=='a' else 0)
	add_counters['time_cat2'] = add_counters['TIME_CAT'].apply(lambda x: 2 if x=='b' else 0)
	add_counters['time_cat3'] = add_counters['TIME_CAT'].apply(lambda x: 3 if x=='c' else 0)
	add_counters.rename(columns={'AMT_CAT':'max_amount','DATE':'recency','LOC_CAT':'num_diff_loc','TIME_CAT':'num_diff_time_slots','MC_CAT':'num_diff_mark_cat'},inplace=True)
	add_counters = add_counters.groupby(['USER_ID']).agg({'pos_cnt': np.sum,'webshop_cnt': np.sum,'credit_card_cnt': np.sum,'debit_card_cnt': np.sum, 'max_amount': np.max, 'num_diff_amounts':pd.Series.nunique,'recency':np.min, 'num_diff_loc':pd.Series.nunique,'num_diff_time_slots':pd.Series.nunique,'num_diff_mark_cat':pd.Series.nunique,'act_loc1':np.sum,'act_loc2':np.sum,'act_loc3':np.sum,'time_cat1':np.sum,'time_cat2':np.sum,'time_cat3':np.sum})
	add_counters = pd.DataFrame(add_counters.reset_index())
	add_counters['total_num_act'] = add_counters['pos_cnt'] + add_counters['webshop_cnt']
	add_counters['max_freq_time_slot'] = add_counters.apply(lambda row: np.max([row['time_cat1'],row['time_cat2'],row['time_cat3']]), axis=1)
	add_counters['max_freq_act_loc'] = add_counters.apply(lambda row: np.max([row['act_loc1'],row['act_loc2'],row['act_loc3']]), axis=1)
	add_counters.drop(['act_loc1','act_loc2','act_loc3','time_cat1','time_cat2','time_cat3'],1,inplace=True)
	add_counters.fillna('0', inplace=True)

	data2014 = pd.merge(data2014, add_counters, how='inner', on='USER_ID')

	trends = data2014[['USER_ID','DATE','AMT_CAT','MC_CAT']]
	trends = trends.sort(['USER_ID', 'DATE'], ascending=[1, 0])
	for f in ['AMT_CAT','MC_CAT']:
		trends[f] = trends[f].apply(str)
		le = preprocessing.LabelEncoder()
		trends[f] = le.fit_transform(np.reshape(trends[[f]].values,(len(trends[[f]]),)))

	trends.drop(['DATE'],1,inplace=True)
	trends_amt = prepare_for_ict(trends[['USER_ID','AMT_CAT']])
	trends_mc = prepare_for_ict(trends[['USER_ID','MC_CAT']])
	del trends
	trends_amt['amt_ups_ratio'] = trends_amt['AMT_CAT'].apply(count_ups)
	trends_amt['amt_downs_ratio'] = trends_amt['AMT_CAT'].apply(count_downs)
	trends_mc['mccat_ups_ratio'] = trends_mc['MC_CAT'].apply(count_ups)
	trends_mc['mccat_downs_ratio'] = trends_mc['MC_CAT'].apply(count_downs)
	trends_amt.drop(['AMT_CAT'],1,inplace=True)
	trends_mc.drop(['MC_CAT'],1,inplace=True)
	data2014 = pd.merge(data2014,trends_amt,how='inner',on='USER_ID')
	data2014 = pd.merge(data2014,trends_mc,how='inner',on='USER_ID')


	# we calc user-activity distance as Euclidean
	# data2014['GEO_X'] = data2014['GEO_X'].astype(float)
	# data2014['GEO_Y'] = data2014['GEO_Y'].astype(float)
	data2014['LOC_GEO_X'] = data2014['LOC_GEO_X'].astype(float)
	data2014['LOC_GEO_Y'] = data2014['LOC_GEO_Y'].astype(float)

	data2014['act_dist'] = data2014.apply(lambda row: math.sqrt((row['LOC_GEO_X']-row['GEO_X'])**2+(row['LOC_GEO_Y']-row['GEO_Y'])**2), 1)
	data2014['ratio'] = data2014.apply(lambda row: row['act_dist']/row['total_num_act'] if row['total_num_act']!=0 else 0, axis=1)
	# data2014.drop(['LOC_GEO_X','GEO_X','GEO_Y','LOC_GEO_Y'],1,inplace=True)

	# calc log2 of date
	# data2014['DATE'] = data2014['DATE'].apply(lambda x: 1/np.log2(x))
	data2014['DATE'] = data2014['DATE'].apply(lambda x: 1)

	# getting rid of branch id
	data2014.drop(['POI_ID'],1,inplace=True)

	# we do label encoding for categorical vars
	for f in ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']:
		data2014[f] = data2014[f].apply(str)
		le = preprocessing.LabelEncoder()
		data2014[f] = le.fit_transform(np.reshape(data2014[[f]].values,(len(data2014[[f]]),)))

	d1 = data2014[['USER_ID','DATE']] 	
	d2 = data2014.drop(['USER_ID','DATE'],1)

	X = d2.values
	# now we apply one hot encoding for all these + channel
	ind_encoding = [0,1,2,18,19,20,21,22,23] # corresponding to ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT'] I have checked this on my own ! ATTENTION!!
	ohe = OneHotEncoder(categorical_features=ind_encoding)
	#ohe = OneHotEncoder(categorical_features='all')
	ohe.fit(X)
	# X = X[0:X.shape[0]-1,:]
	X = ohe.transform(X)

	# # have to decrease len of d1 as well
	# d1 = d1[0:len(d1)-1]

	# multiplying with weights
	X = np.multiply(X.todense(),np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	d1 = np.hstack((d1.values,X))
	cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	train_df = pd.DataFrame(d1,columns=cols)

	train_df = train_df.groupby(['USER_ID']).sum()
	train_df = pd.DataFrame(train_df.reset_index())
	d1 = train_df[['USER_ID','DATE']]
	d2 = train_df.drop(['USER_ID','DATE'],1)

	X = d2.values
	# calculating weighted average
	X = np.divide(X,np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	d1 = np.hstack((d1.values,X))
	cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	train_df = pd.DataFrame(d1,columns=cols)

	train_df.drop(['DATE'],1,inplace=True)

	# for f in ['AGE_CAT','USER_LOC_CAT','INC_CAT']:
	# 	user2014[f] = user2014[f].apply(str)
	# 	le = preprocessing.LabelEncoder()
	# 	user2014[f] = le.fit_transform(np.reshape(user2014[[f]].values,(len(user2014[[f]]),)))



	# train_df = pd.merge(train_df, user2014, how='inner', on='USER_ID')	
	train_df = pd.DataFrame(train_df.reset_index())
	# userids = train_df['USER_ID'] # list(map(int,train_df['USER_ID'].tolist()))


	df_branch = df_branch[['USER_ID','POI_ID','CHANNEL']]
	df_branch['CHANNEL'] = df_branch['CHANNEL'].apply(lambda x: 1 if x=='b' else 0)
	df_branch = df_branch.groupby(['USER_ID','POI_ID']).sum()
	df_branch = pd.DataFrame(df_branch.reset_index())
	df_branch = df_branch.sort(['USER_ID','CHANNEL'], ascending=[1, 0])
	df_branch = df_branch.groupby('USER_ID').head(5).reset_index(drop=True)
	# df_branch.to_csv('gt2014.csv',delimiter=',',index=False)

	branches = set(df_branch['POI_ID'])
	cust = set(train_df['USER_ID'])

	cust_dict = {}
	rev_cust_dict = {}
	nc = 0
	for c in cust:
		if c not in cust_dict:
			cust_dict[c] = nc
			rev_cust_dict[nc] = c
			nc += 1


	branch_dict = {}
	rev_branch_dict = {}
	nb = 0
	for b in branches:
		if b not in branch_dict:
			branch_dict[b] = nb
			rev_branch_dict[nb] = b
			nb += 1

	mat = np.zeros((nc,nb))

	for r in df_branch.values:
		mat[cust_dict[r[0]],branch_dict[r[1]]] = r[2]


	# train_df = train_df[train_df['USER_ID'].isin(cust_dict)]
	# train_df = pd.DataFrame(train_df.reset_index())
	# userids = train_df['USER_ID'] #np.array(map(int,train_df['USER_ID'].tolist()))

	# train_df.drop(['level_0'],1,inplace=True)

	# train_df = pd.merge(train_df, user_geo, how='inner', on='USER_ID')

	user_geo = user2014[['USER_ID','LOC_GEO_X','LOC_GEO_Y']]
	train_df = pd.merge(train_df, user_geo, how='left',on='USER_ID')
	user_geo_matrix = train_df[['LOC_GEO_X','LOC_GEO_Y']].values

	dist_train = cdist(user_geo_matrix, bank_geo_matrix, 'euclidean')
	aux = dist_train.min(axis=1)
	d1 = np.hstack((train_df.values, aux.reshape((len(aux),1))))
	cols = list(train_df.columns) + ['min_user_branch_geo_dist']
	# d1 = np.hstack((train_df.values,dist_train))
	# cols = list(train_df.columns) + ['geo_dist'+str(i) for i in range(dist_train.shape[1])]
	train_df = pd.DataFrame(d1,columns=cols)

	train_df = pd.merge(train_df, act_geo, how='left',on='USER_ID')
	act_geo_matrix = train_df[['GEO_X','GEO_Y']].values
	act_branch_dist_train = cdist(act_geo_matrix, bank_geo_matrix, 'euclidean')
	aux = act_branch_dist_train.min(axis=1)
	d1 = np.hstack((train_df.values, aux.reshape((len(aux),1))))
	cols = list(train_df.columns) + ['min_act_branch_geo_dist']
	train_df = pd.DataFrame(d1,columns=cols)

	train_df = pd.merge(train_df, ict_calc_df, how='left', on='USER_ID')
	train_df.fillna(0, inplace=True)

	userids = train_df['USER_ID'] # list(map(int,train_df['USER_ID'].tolist()))

	# act_branch_dist2014 = pd.merge(train_df, act_branch_dist2014, how='left',on='USER_ID')[['USER_ID','GEO_X','GEO_Y']]
	# act_branch_dist2014.fillna(9999,inplace=True)
	# for i in range(len(bank_info)):
	# 	act_branch_dist2014['act_branch'+str(i)+'dist'] = act_branch_dist2014.apply(lambda row: calc_act_branch_dist(row['GEO_X'],row['GEO_Y'],bank_info[i:i+1]), 1)

	# act_branch_dist2014.drop(['GEO_X','GEO_Y'],1,inplace=True)
	# act_branch_dist2014 = act_branch_dist2014.groupby(['USER_ID']).min()
	# act_branch_dist2014 = pd.DataFrame(act_branch_dist2014.reset_index())

	# user_geo = user2014[['USER_ID','LOC_GEO_X','LOC_GEO_Y']]
	# user_geo.rename(columns={'LOC_GEO_X':'user_x_coord','LOC_GEO_Y':'user_y_coord'},inplace=True)
	# train_df = pd.merge(train_df, user_geo, how='inner',on='USER_ID')

	train_df.drop(['index','USER_ID'],1,inplace=True)
	# train_df.drop(['geo_dist'+str(i) for i in range(dist_train.shape[1])], 1, inplace=True)

	X_train = train_df.values

	###########################################################

	user2015 = pd.read_csv('users_2015.csv',sep=',')
	user2015['AGE_CAT'] = user2015['AGE_CAT'].apply(lambda x: 'b' if x=='-' else x)
	user2015 = user2015.replace('-','0')
	user2015.rename(columns={'LOC_CAT':'USER_LOC_CAT'},inplace=True)

	data2015 = pd.read_csv('train_2015.csv',sep=',')
	data2015 = data2015.replace('-','0')

	data2015['DATE'] = data2015['DATE'].apply(lambda x: convert_date(x, '2015-07-01', 170))

	# we select only first half of the year
	data2015 = data2015[data2015['DATE']>0]


	data2015 = pd.merge(user2015, data2015, how='left', on='USER_ID')

	data2015['sum_C_cols'] = data2015['C201501']+data2015['C201502']+data2015['C201503']+data2015['C201504']+data2015['C201505']+data2015['C201506']
	data2015['sum_W_cols'] = data2015['W201501']+data2015['W201502']+data2015['W201503']+data2015['W201504']+data2015['W201505']+data2015['W201506']

	data2015.fillna('f', inplace=True)
	data2015['DATE'] = data2015['DATE'].apply(lambda x: 170 if x=='f' else x)
	# data2015['DATE'] = data2015['DATE'].apply(lambda x: 170 if x=='f' else x)
	# data2015['DATE'] = data2015['DATE'].apply(lambda x: 170 if x=='f' else x)


	data2015['GEO_X'] = data2015['GEO_X'].apply(lambda x: 0.0 if x=='f' else x)
	data2015['GEO_Y'] = data2015['GEO_Y'].apply(lambda x: 0.0 if x=='f' else x)

	data2015['sum_C_cols'] = data2015['sum_C_cols'].apply(lambda x: 0 if x=='f' else x)
	data2015['sum_W_cols'] = data2015['sum_W_cols'].apply(lambda x: 0 if x=='f' else x)

	data2015['GEO_X'] = data2015['GEO_X'].astype(float)
	data2015['GEO_Y'] = data2015['GEO_Y'].astype(float)
	act_geo = data2015[['USER_ID','GEO_X','GEO_Y']]
	act_geo = act_geo.groupby(['USER_ID']).mean()
	act_geo = pd.DataFrame(act_geo.reset_index())

	# act_branch_dist2015 = data2015[['USER_ID','GEO_X','GEO_Y']]


	# calculating clumpiness wrongly named as ict :)
	ict_calc_df = data2015[['USER_ID','DATE']]
	ict_calc_df = prepare_for_ict(ict_calc_df)
	ict_calc_df['ICT'] = ict_calc_df['DATE'].apply(lambda x: calc_ict(x))
	ict_calc_df['mean_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.mean(x))
	ict_calc_df['std_ICT'] = ict_calc_df['ICT'].apply(lambda x: np.std(x))
	ict_calc_df['C'] = ict_calc_df['ICT'].apply(lambda x: calc_cl(x))
	ict_calc_df.drop(['DATE','ICT'],1,inplace=True)


	add_counters = data2015[['USER_ID','CHANNEL','AMT_CAT','CARD_CAT','DATE','LOC_CAT','TIME_CAT','MC_CAT']]
	add_counters['pos_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='p' else 0)
	add_counters['webshop_cnt'] = add_counters['CHANNEL'].apply(lambda x: 1 if x=='n' else 0)
	add_counters['credit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='c' else 0)
	add_counters['debit_card_cnt'] = add_counters['CARD_CAT'].apply(lambda x: 1 if x=='d' else 0)
	add_counters.drop(['CHANNEL','CARD_CAT'],1,inplace=True)
	add_counters['AMT_CAT'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['num_diff_amounts'] = add_counters['AMT_CAT'].apply(lambda x: small_conversion(x))
	add_counters['last_act_till_feb'] = add_counters['DATE'].apply(lambda x: 1 if x>122 else 0)
	add_counters['last_act_after_feb'] = add_counters['DATE'].apply(lambda x: 1 if x<=122 else 0)
	add_counters['act_loc1'] = add_counters['LOC_CAT'].apply(lambda x: 1 if x=='a' else 0)
	add_counters['act_loc2'] = add_counters['LOC_CAT'].apply(lambda x: 2 if x=='b' else 0)
	add_counters['act_loc3'] = add_counters['LOC_CAT'].apply(lambda x: 3 if x=='c' else 0)
	add_counters['time_cat1'] = add_counters['TIME_CAT'].apply(lambda x: 1 if x=='a' else 0)
	add_counters['time_cat2'] = add_counters['TIME_CAT'].apply(lambda x: 2 if x=='b' else 0)
	add_counters['time_cat3'] = add_counters['TIME_CAT'].apply(lambda x: 3 if x=='c' else 0)
	add_counters.rename(columns={'AMT_CAT':'max_amount','DATE':'recency','LOC_CAT':'num_diff_loc','TIME_CAT':'num_diff_time_slots','MC_CAT':'num_diff_mark_cat'},inplace=True)
	add_counters = add_counters.groupby(['USER_ID']).agg({'pos_cnt': np.sum,'webshop_cnt': np.sum,'credit_card_cnt': np.sum,'debit_card_cnt': np.sum, 'max_amount': np.max, 'num_diff_amounts':pd.Series.nunique,'recency':np.min, 'num_diff_loc':pd.Series.nunique,'num_diff_time_slots':pd.Series.nunique,'num_diff_mark_cat':pd.Series.nunique,'act_loc1':np.sum,'act_loc2':np.sum,'act_loc3':np.sum,'time_cat1':np.sum,'time_cat2':np.sum,'time_cat3':np.sum})
	add_counters = pd.DataFrame(add_counters.reset_index())
	add_counters['total_num_act'] = add_counters['pos_cnt'] + add_counters['webshop_cnt']
	add_counters['max_freq_time_slot'] = add_counters.apply(lambda row: np.max([row['time_cat1'],row['time_cat2'],row['time_cat3']]), axis=1)
	add_counters['max_freq_act_loc'] = add_counters.apply(lambda row: np.max([row['act_loc1'],row['act_loc2'],row['act_loc3']]), axis=1)
	add_counters.drop(['act_loc1','act_loc2','act_loc3','time_cat1','time_cat2','time_cat3'],1,inplace=True)
	add_counters.fillna('0', inplace=True)

	data2015 = pd.merge(data2015, add_counters, how='inner', on='USER_ID')

	trends = data2015[['USER_ID','DATE','AMT_CAT','MC_CAT']]
	trends = trends.sort(['USER_ID', 'DATE'], ascending=[1, 0])
	for f in ['AMT_CAT','MC_CAT']:
		trends[f] = trends[f].apply(str)
		le = preprocessing.LabelEncoder()
		trends[f] = le.fit_transform(np.reshape(trends[[f]].values,(len(trends[[f]]),)))

	trends.drop(['DATE'],1,inplace=True)
	trends_amt = prepare_for_ict(trends[['USER_ID','AMT_CAT']])
	trends_mc = prepare_for_ict(trends[['USER_ID','MC_CAT']])
	del trends
	trends_amt['amt_ups_ratio'] = trends_amt['AMT_CAT'].apply(count_ups)
	trends_amt['amt_downs_ratio'] = trends_amt['AMT_CAT'].apply(count_downs)
	trends_mc['mccat_ups_ratio'] = trends_mc['MC_CAT'].apply(count_ups)
	trends_mc['mccat_downs_ratio'] = trends_mc['MC_CAT'].apply(count_downs)
	trends_amt.drop(['AMT_CAT'],1,inplace=True)
	trends_mc.drop(['MC_CAT'],1,inplace=True)
	data2015 = pd.merge(data2015,trends_amt,how='inner',on='USER_ID')
	data2015 = pd.merge(data2015,trends_mc,how='inner',on='USER_ID')

	# we calc user-activity distance as Euclidean
	# data2015['GEO_X'] = data2015['GEO_X'].astype(float)
	# data2015['GEO_Y'] = data2015['GEO_Y'].astype(float)
	data2015['LOC_GEO_X'] = data2015['LOC_GEO_X'].astype(float)
	data2015['LOC_GEO_Y'] = data2015['LOC_GEO_Y'].astype(float)

	data2015['act_dist'] = data2015.apply(lambda row: math.sqrt((row['LOC_GEO_X']-row['GEO_X'])**2+(row['LOC_GEO_Y']-row['GEO_Y'])**2), 1)
	data2015['ratio'] = data2015.apply(lambda row: row['act_dist']/row['total_num_act'] if row['total_num_act']!=0 else 0, axis=1)
	# data2015.drop(['LOC_GEO_X','GEO_X','GEO_Y','LOC_GEO_Y'],1,inplace=True)


	data2015['DATE'] = data2015['DATE'].apply(lambda x: 1)

	# getting rid of branch id
	data2015.drop(['POI_ID'],1,inplace=True)

	# we do label encoding for categorical vars
	for f in ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT','AGE_CAT','USER_LOC_CAT','INC_CAT']:
		data2015[f] = data2015[f].apply(str)
		le = preprocessing.LabelEncoder()
		data2015[f] = le.fit_transform(np.reshape(data2015[[f]].values,(len(data2015[[f]]),)))

	d1 = data2015[['USER_ID','DATE']] 	
	d2 = data2015.drop(['USER_ID','DATE'],1)

	X = d2.values
	# now we apply one hot encoding for all these + channel
	ind_encoding = [0,1,2,18,19,20,21,22,23] # corresponding to ['CHANNEL','TIME_CAT','LOC_CAT','MC_CAT','CARD_CAT','AMT_CAT'] I have checked this on my own ! ATTENTION!!
	# ohe = OneHotEncoder(categorical_features=ind_encoding)
	#ohe = OneHotEncoder(categorical_features='all')
	X = ohe.transform(X)

	# multiplying with weights
	X = np.multiply(X.todense(),np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	d1 = np.hstack((d1.values,X))
	cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	test_df = pd.DataFrame(d1,columns=cols)

	test_df = test_df.groupby(['USER_ID']).sum()
	test_df = pd.DataFrame(test_df.reset_index())
	d1 = test_df[['USER_ID','DATE']]
	d2 = test_df.drop(['USER_ID','DATE'],1)

	X = d2.values
	# calculating weighted average
	X = np.divide(X,np.reshape(d1['DATE'].values,(X.shape[0],))[:,np.newaxis])

	d1 = np.hstack((d1.values,X))
	cols = ['USER_ID','DATE'] + ['c'+str(i) for i in range(X.shape[1])]
	test_df = pd.DataFrame(d1,columns=cols)

	test_df.drop(['DATE'],1,inplace=True)

	# test_df = pd.merge(test_df, user2015, how='inner', on='USER_ID')
	test_df = pd.DataFrame(test_df.reset_index())

	# test_df = pd.merge(test_df, user_geo, how='inner', on='USER_ID')
	user_geo = user2015[['USER_ID','LOC_GEO_X','LOC_GEO_Y']]
	test_df = pd.merge(test_df, user_geo, how='left',on='USER_ID')
	test_user_geo_matrix = test_df[['LOC_GEO_X','LOC_GEO_Y']].values

	# test_act_geo_matrix = test_df[['GEO_X','GEO_Y']].values

	dist_test = cdist(test_user_geo_matrix, bank_geo_matrix, 'euclidean')
	aux = dist_test.min(axis=1)
	d1 = np.hstack((test_df.values,aux.reshape((len(aux),1))))
	cols = list(test_df.columns) + ['min_user_branch_geo_dist']
	# d1 = np.hstack((test_df.values,dist_test))
	# cols = list(test_df.columns) + ['geo_dist'+str(i) for i in range(dist_test.shape[1])]
	test_df = pd.DataFrame(d1,columns=cols)

	test_df = pd.merge(test_df, act_geo, how='left',on='USER_ID')
	test_act_geo_matrix = test_df[['GEO_X','GEO_Y']].values
	act_branch_dist_test = cdist(test_act_geo_matrix, bank_geo_matrix, 'euclidean')
	aux = act_branch_dist_test.min(axis=1)
	d1 = np.hstack((test_df.values, aux.reshape((len(aux),1))))
	cols = list(test_df.columns) + ['min_act_branch_geo_dist']
	test_df = pd.DataFrame(d1,columns=cols)

	test_df = pd.merge(test_df, ict_calc_df, how='left', on='USER_ID')
	test_df.fillna(0, inplace=True)

	userids_test = test_df['USER_ID'] #np.array(map(int,test_df['USER_ID'].tolist()))

	# act_branch_dist2015 = pd.merge(test_df, act_branch_dist2015, how='left',on='USER_ID')[['USER_ID','GEO_X','GEO_Y']]
	# act_branch_dist2015.fillna(9999,inplace=True)
	# for i in range(len(bank_info)):
	# 	act_branch_dist2015['act_branch'+str(i)+'dist'] = act_branch_dist2015.apply(lambda row: calc_act_branch_dist(row['GEO_X'],row['GEO_Y'],bank_info[i:i+1]), 1)

	# act_branch_dist2015.drop(['GEO_X','GEO_Y'],1,inplace=True)
	# act_branch_dist2015 = act_branch_dist2015.groupby(['USER_ID'].min())
	# act_branch_dist2015 = pd.DataFrame(act_branch_dist2015.reset_index())



	# user_geo = user2015[['USER_ID','LOC_GEO_X','LOC_GEO_Y']]
	# user_geo.rename(columns={'LOC_GEO_X':'user_x_coord','LOC_GEO_Y':'user_y_coord'},inplace=True)
	# test_df = pd.merge(test_df, user_geo, how='inner',on='USER_ID')

	test_df.drop(['index','USER_ID'],1,inplace=True)
	# test_df.drop(['geo_dist'+str(i) for i in range(dist_test.shape[1])], 1, inplace=True)


	X_test = test_df.values

	###########################################################

	Y_train = mat[[cust_dict[m] for m in userids],:]

	# for i in range(Y_train.shape[0]):
	# 	selected = np.argsort(Y_train[i,:])[::-1]
	# 	Y_train[i,selected[5:]] = 0

	# Y_train = normalize(Y_train, norm='l2', axis=1, copy=False)

	# mmscaler = MinMaxScaler(feature_range=(-1, 1))
	# mmscaler.fit(np.vstack((X_train,X_test)))
	# X_train = mmscaler.transform(X_train)
	# X_test = mmscaler.transform(X_test)


	X_train_geo = train_df[['LOC_GEO_X','LOC_GEO_Y']].values
	X_test_geo = test_df[['LOC_GEO_X','LOC_GEO_Y']].values

	cl = cluster.KMeans(n_clusters=200) #, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=49)
	# cl = Birch(n_clusters=num_clusters, compute_labels=True)
	cl.fit(X_train_geo) 
	train_cl = cl.labels_
	train_user_clusters_assigned = cl.predict(X_train_geo)
	test_user_clusters_assigned = cl.predict(X_test_geo)

	X_train = np.hstack((X_train,train_user_clusters_assigned.reshape((-1,1))))
	X_test = np.hstack((X_test,test_user_clusters_assigned.reshape((-1,1))))

	# ohe = OneHotEncoder(categorical_features=[X_train.shape[1]-1])
	# ohe.fit(np.vstack((X_train,X_test)))
	# X_train = ohe.transform(X_train).todense()
	# X_test = ohe.transform(X_test).todense()
	
	# X_train = X_train.tolist()
	# X_test = X_test.tolist()
	# Y_train = Y_train.tolist()
	# dist_train = dist_train.tolist()
	# dist_test = dist_test.tolist()

	with open('X_train_t1.txt', 'w') as outfile:
		json.dump(X_train.tolist(), outfile)

	with open('X_test_t1.txt', 'w') as outfile:
		json.dump(X_test.tolist(), outfile)

	with open('Y_train_t1.txt', 'w') as outfile:
		json.dump(Y_train.tolist(), outfile)

	with open('userids_test_t1.txt', 'w') as outfile:
		json.dump(userids_test.tolist(), outfile)

	for key in rev_branch_dict:
		rev_branch_dict[key] = int(rev_branch_dict[key])

	with open('rev_branch_dict_t1.txt', 'w') as outfile:
		json.dump(rev_branch_dict, outfile, ensure_ascii=False)

	with open('dist_train_t1.txt', 'w') as outfile:
		json.dump(dist_train.tolist(), outfile)

	with open('dist_test_t1.txt', 'w') as outfile:
		json.dump(dist_test.tolist(), outfile)

	with open('act_branch_dist_train.txt', 'w') as outfile:
		json.dump(act_branch_dist_train.tolist(), outfile)

	with open('act_branch_dist_test.txt', 'w') as outfile:
		json.dump(act_branch_dist_test.tolist(), outfile)


	# # dim_red = PCA(n_components=300) #, copy=True, whiten=False)[source]
	# # Y_train = dim_red.fit_transform(Y_train)
	# # print('variance ratio', np.sum(dim_red.explained_variance_ratio_))

