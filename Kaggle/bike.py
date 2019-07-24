#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 19:48
#@Author: czj
#@File  : bike.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
path = 'D:\\MyInstallData\\PyCharm\PythonLearn\\Kaggle\\kaggle_bike_competition_train.csv'
bike = pd.read_csv(path,sep=',')
bike['datetime'] = pd.to_datetime(bike['datetime'])
tmp = pd.DatetimeIndex(bike['datetime'])
#获取  date 和 hour 字段
bike['date'] = tmp.date
bike['hour'] = tmp.hour


#因为一周中 周末和周日的人出去的比较多，设定dayofweek
#dateDays 设立离该人第一次租车有多久了，认为第一次租车后
#这种方式会蔓延
bike['dayofweek'] = tmp.dayofweek
bike['dateDays'] = (bike.date-bike.date[0]).astype('timedelta64[D]')

#统计一个星期的各个天中，租车的人数，分为casual 和 register
bike.groupby(['dayofweek'])[['casual','registered']].sum().reset_index()

#单独拿出一个列判断是不是是星期六或者星期日
bike['Saturday'] = 0
bike['Sunday'] = 0
a = np.where(bike.dayofweek==5)[0]
bike.ix[a,'Saturday']= 1
a = np.where(bike.dayofweek==6)[0]
bike.ix[a,'Sunday']= 1

#从原始数据中删除关于原始的时间字段，将剩下的数据组成特征向量
bike_rel = bike.drop(['datetime','date','count','dayofweek'],axis=1)

#将bike_rel的数据中 连续值属性和离散值属性分开
from sklearn.feature_extraction import DictVectorizer
featureConCols = ['temp','atemp','humidity','windspeed','dateDays','hour']
bikeFeatureCon = bike_rel[featureConCols]
bikeFeatureCon = bikeFeatureCon.fillna('NA')
x_dictCon = bikeFeatureCon.T.to_dict().values()

#吧离散的特征放到另外一个向量中去
featureDisCols = ['season','holiday','workingday','weather','Saturday','Sunday']
bikeFeatureDis = bike_rel[featureDisCols]
bikeFeatureDis = bikeFeatureDis.fillna('NA')
x_dictDis = bikeFeatureDis.T.to_dict().values()

#特征向量化
vec = DictVectorizer(sparse=False)
x_vec_con = vec.fit_transform(x_dictCon)
x_vec_dis = vec.fit_transform(x_dictDis)

#标准化连续值 0mean 1variance
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_vec_con = s.fit_transform(x_vec_con)
#离散值类别特征编码
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
x_vec_dis = encoder.fit_transform(x_vec_dis).toarray()

#把离散特征和连续特征拼接起来
x_vec = np.concatenate((x_vec_con,x_vec_dis),axis=1)

#对于目标进行预测
y_registered = bike_rel['registered'].values.astype(float)
y_casual = bike_rel['casual'].values.astype(float)

y = np.stack((y_registered,y_casual),axis=1)


#建立模型进行预测
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskElasticNetCV
x1,x2,y1,y2 = train_test_split(x_vec,y,test_size=0.2,random_state=20)

############ Lasso
mtl = MultiTaskLassoCV(alphas=np.logspace(-3,-1,3),cv=8,verbose=3)
mtl.fit(x1,y1)
mtl.score(x1,y1)
mtl.score(x2,y2)

############ ElasticNetCV
mte = MultiTaskElasticNetCV(l1_ratio=np.logspace(-3,-1,3),alphas=np.logspace(-3,-1,3),cv=8,verbose=3)
mte.fit(x1,y1)
mtl.score(x1,y1)
mtl.score(x2,y2)
