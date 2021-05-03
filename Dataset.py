#!/bin/python3
# -*- coding: utf-8 -*-

class Dataset:
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
		self.train=Dataset.FeatureLabels(train_features,train_labels)
		self.val=Dataset.FeatureLabels(val_features,val_labels)
		self.test=Dataset.FeatureLabels(test_features,test_labels)
		self.train_val=Dataset.FeatureLabels(train_full_features,train_full_labels)
		self.indexes=Dataset.TrainTest(index_train,index_test)