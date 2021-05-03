#!/bin/python3
# -*- coding: utf-8 -*-

import hashlib
import math
import json
from Utils import Utils

class Hyperparameters:
	def __init__(self,input_features=['Close'],output_feature='Close',index_feature='Date',backwards_samples=20,forward_samples=7,lstm_layers=2,max_epochs=200,patience_epochs=10,batch_size=5,stateful=False,dropout_values=[0,0],layer_sizes=[25,15],normalize=True,optimizer='adam',model_metrics=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity'],loss='mean_squared_error',train_percent=.8,val_percent=.2,amount_companies=1,shuffle=True):
		self.backwards_samples=backwards_samples
		self.forward_samples=forward_samples
		self.lstm_layers=lstm_layers
		self.max_epochs=max_epochs
		self.patience_epochs=patience_epochs
		self.batch_size=batch_size
		self.stateful=stateful
		self.dropout_values=dropout_values
		self.layer_sizes=layer_sizes
		self.normalize=normalize
		self.optimizer=optimizer
		self.model_metrics=model_metrics
		self.loss=loss
		self.train_percent=train_percent
		self.val_percent=val_percent
		self.amount_companies=amount_companies
		self.input_features=input_features
		self.output_feature=output_feature
		self.index_feature=index_feature
		self.shuffle=shuffle
		if type(self.dropout_values)==int:
			self.dropout_values=[self.dropout_values]*self.lstm_layers
		if type(self.layer_sizes)==int:
			self.layer_sizes=[self.layer_sizes]*self.lstm_layers
		if len(self.dropout_values)!=self.lstm_layers:
			raise Exception('Wrong dropout_values array size, should be {}'.format(self.lstm_layers))
		if len(self.layer_sizes)!=self.lstm_layers:
			raise Exception('Wrong layer_sizes array size, should be {}'.format(self.lstm_layers))
		if len(self.input_features)>1 and self.amount_companies>1:
			raise Exception('Only input_features or amount_companies must be greater than 1')
		if self.val_percent>1 or self.train_percent>1 or self.val_percent<0 or self.train_percent<0:
			raise Exception('Train + validation percent must be smaller than 1 and bigger than 0')
		if self.stateful:
			self.batch_size=1 # batch size must be one for stateful
		self.layer_sizes.insert(0,self.backwards_samples)
		self.uuid=self.genUuid()

	def toString(self):
		string=''
		string+='backwards_samples: {}'.format(self.backwards_samples)+', '
		string+='forward_samples: {}'.format(self.forward_samples)+', '
		string+='lstm_layers: {}'.format(self.lstm_layers)+', '
		string+='max_epochs: {}'.format(self.max_epochs)+', '
		string+='patience_epochs: {}'.format(self.patience_epochs)+', '
		string+='batch_size: {}'.format(self.batch_size)+', '
		string+='stateful: {}'.format(self.stateful)+', '
		string+='dropout_values: {}'.format(self.dropout_values)+', '
		string+='layer_sizes: {}'.format(self.layer_sizes)+', '
		string+='normalize: {}'.format(self.normalize)+', '
		string+='optimizer: {}'.format(self.optimizer)+', '
		string+='model_metrics: {}'.format(self.model_metrics)+', '
		string+='loss: {}'.format(self.loss)+', '
		string+='train_percent: {}'.format(self.train_percent)+', '
		string+='val_percent: {}'.format(self.val_percent)+', '
		string+='amount_companies: {}'.format(self.amount_companies)+', '
		string+='input_features: {}'.format(self.input_features)+', '
		string+='output_feature: {}'.format(self.output_feature)+', '
		string+='index_feature: {}'.format(self.index_feature)+', '
		string+='shuffle: {}'.format(self.shuffle)
		return string

	def genUuid(self,low_resolution=False):
		to_hash=self.toString().encode('utf-8')
		if low_resolution:
			hash_object=hashlib.md5(to_hash)
		else:
			hash_object=hashlib.sha256(to_hash)
		return hash_object.hexdigest()

	@staticmethod
	def jsonDecoder(obj):
		if '__type__' in obj and obj['__type__'] == 'Hyperparameters':
			return Hyperparameters(obj['input_features'],obj['output_feature'],obj['index_feature'],obj['backwards_samples'],obj['forward_samples'],obj['lstm_layers'],obj['max_epochs'],obj['patience_epochs'],obj['batch_size'],obj['stateful'],obj['dropout_values'],obj['layer_sizes'],obj['normalize'],obj['optimizer'],obj['model_metrics'],obj['loss'],obj['train_percent'],obj['val_percent'],obj['amount_companies'],obj['shuffle'])
		return obj

	@staticmethod
	def loadJson(path):
		with open(path, 'r') as fp :
			return json.load(fp,object_hook=Hyperparameters.jsonDecoder)

	def saveJson(self,path):
		hyperparameters_dict=self.__dict__
		hyperparameters_dict['__type__']='Hyperparameters'
		Utils.saveJson(hyperparameters_dict,path)

	@staticmethod
	def estimateLayerOutputSize(layer_input_size,network_output_size,train_data_size=0,a=2,second_formula=False):
		if not second_formula:
			return int(math.ceil(train_data_size/(a*(layer_input_size+network_output_size))))
		else:
			return int(math.ceil(2/3*(layer_input_size+network_output_size)))
