#!/bin/python3
# -*- coding: utf-8 -*-

from Utils import Utils

class DatasetOld:
	class FeatureLabels:
		def __init__(self,features,labels):
			self.features=features
			self.labels=labels
	class TrainTest:
		def __init__(self,train,test):
			self.train=train
			self.test=test
	class EvalData:
		def __init__(self,features,predicted=None,real=None,index=None):
			self.features=features
			self.predicted=predicted
			self.real=real
			self.index=index
	
	def __init__(self,name,scalers,train_features,train_labels,val_features,val_labels,test_features,test_labels,train_full_features,train_full_labels,index_train,index_test):
		self.name=name
		self.scalers=scalers
		self.train=DatasetOld.FeatureLabels(train_features,train_labels)
		self.val=DatasetOld.FeatureLabels(val_features,val_labels)
		self.test=DatasetOld.FeatureLabels(test_features,test_labels)
		self.train_val=DatasetOld.FeatureLabels(train_full_features,train_full_labels)
		self.indexes=DatasetOld.TrainTest(index_train,index_test)