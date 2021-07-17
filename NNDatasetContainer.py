#!/bin/python3
# -*- coding: utf-8 -*-

from Utils import Utils
from Dataset import Dataset

class NNDatasetContainer:

	def deployScaler(self):
		self.dataset.normalization_method=Dataset.Normalization.NORMALIZE_WITH_EXTERNAL_MAXES
		self.dataset.normalization_params=self.scaler+tuple()

	def importScaler(self):
		self.scaler=self.dataset.normalization_params+tuple()

	def getNormalizationMethod(self):
		norm_method=Dataset.Normalization.DONT_NORMALIZE
		norm_param=None
		if self.normalize:
			if len(self.scaler)>0:
				norm_param=self.scaler
				norm_method=Dataset.Normalization.NORMALIZE_WITH_EXTERNAL_MAXES
			else:
				norm_method=Dataset.Normalization.NORMALIZE_WITH_GAP
		return norm_method,norm_param

	def generateNNArrays(self):
		if not self.dataset.converted:
			self.dataset.convertToTemporalValues(self.back_samples,self.forward_samples)
		
		norm_method,norm_param=self.getNormalizationMethod()

		start_index,dataset_x,dataset_y=self.dataset.getNeuralNetworkArrays(include_test_data=True,normalization=norm_method,external_maxes=norm_param)
		if len(self.scaler)==0:
			self.importScaler()

		if self.train_percent==0:
			self.train_x=None
			self.train_y=None
			self.val_x=None
			self.val_y=None
			self.test_x=dataset_x
			self.test_y=dataset_y
			self.train_start_idx=None
			self.val_start_idx=None
			self.test_start_idx=start_index
			print()
			print('test_x',self.test_x.shape)
			print('test_y',self.test_y.shape)
			print()
		else:
			test_index,train_x,test_x=Dataset.splitNeuralNetworkArray(dataset_x,self.train_percent)
			_,train_y,test_y=Dataset.splitNeuralNetworkArray(dataset_y,part2_index=test_index)
			val_index,train_x,val_x=Dataset.splitNeuralNetworkArray(train_x,1-self.val_percent)
			_,train_y,val_y=Dataset.splitNeuralNetworkArray(train_y,part2_index=val_index)
			self.train_x=train_x
			self.train_y=train_y
			self.val_x=val_x
			self.val_y=val_y
			self.test_x=test_x
			self.test_y=test_y
			self.train_start_idx=start_index
			self.val_start_idx=start_index+val_index
			self.test_start_idx=start_index+test_index
			print()
			print('train_x',self.train_x.shape)
			print('train_y',self.train_y.shape)
			print()
			print('val_x',self.val_x.shape)
			print('val_y',self.val_y.shape)
			print()
			print('test_x',self.test_x.shape)
			print('test_y',self.test_y.shape)
			print()

	def getValuesSplittedByFeature(self):
		norm_method,norm_param=self.getNormalizationMethod()
		normalize=self.dataset.setNormalizationMethod(normalization=norm_method,external_maxes=norm_param)
		return self.dataset.getValuesSplittedByFeature(normalize=normalize)
		

	def __init__(self,dataset,scaler,train_percent,val_percent,back_samples,forward_samples,normalize):
		self.dataset=dataset
		self.scaler=scaler
		self.train_percent=train_percent
		self.val_percent=val_percent
		self.back_samples=back_samples
		self.forward_samples=forward_samples
		self.normalize=normalize
		self.train_x=None
		self.train_y=None
		self.val_x=None
		self.val_y=None
		self.test_x=None
		self.test_y=None
		self.train_start_idx=None
		self.val_start_idx=None
		self.test_start_idx=None