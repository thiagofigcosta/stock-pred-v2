#!/bin/python3
# -*- coding: utf-8 -*-

import hashlib
import math
import json
from Enums import Features
from Utils import Utils

class Hyperparameters:

	REGRESSION_OUTPUT_ACTIVATION_FUNCTION='linear'
	BINARY_OUTPUT_ACTIVATION_FUNCTION='sigmoid'

	def __init__(self,name='',input_features=['Close'],output_feature='Close',index_feature='Date',backwards_samples=20,forward_samples=7,lstm_layers=2,max_epochs=200,patience_epochs_stop=10,patience_epochs_reduce=10,reduce_factor=.1,batch_size=5,stateful=False,dropout_values=[0,0],layer_sizes=[25,15],normalize=True,optimizer='rmsprop',model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error'],loss='mean_squared_error',train_percent=.8,val_percent=.2,amount_companies=1,shuffle=True,activation_functions='tanh',recurrent_activation_functions='sigmoid',bias=True,use_dense_on_output=False,unit_forget_bias=True,go_backwards=False,recurrent_dropout_values=0,binary_classifier=False):
		input_features=Utils.parseInputFeaturesForHyperparameters(input_features)
		output_feature=Utils.parseOutputFeatureForHyperparameters(output_feature)
		index_feature=Utils.parseIndexFeatureForHyperparameters(index_feature)
		model_metrics=Utils.parseMetricsForHyperparameters(model_metrics)
		loss=Utils.parseLossForHyperparameters(loss)
		activation_functions=Utils.parseNodeTypeForHyperparameters(activation_functions)
		recurrent_activation_functions=Utils.parseNodeTypeForHyperparameters(recurrent_activation_functions)
		optimizer=Utils.parseOptimizerForHyperparameters(optimizer)
		self.uuid=None
		self.name=name
		self.backwards_samples=backwards_samples # [5, 60]
		self.forward_samples=forward_samples # [5, 14]
		self.lstm_layers=lstm_layers # [1, 4]
		self.max_epochs=max_epochs	# [10, 200]
		self.patience_epochs_stop=patience_epochs_stop # [10, 20]
		self.patience_epochs_reduce=patience_epochs_reduce # [10, 20]
		self.reduce_factor=reduce_factor # [0.0, 0.5]
		self.batch_size=batch_size # [1, 22]
		self.stateful=stateful # [False, True] or [0, 1]
		self.dropout_values=dropout_values # [0.0, 1.0] * lstm_layers
		self.layer_sizes=layer_sizes # [10, 80] * lstm_layers (two formulas to approximate that)
		self.normalize=normalize # [False, True] or [0, 1] - or always True
		self.optimizer=optimizer # 'adam' or 'sgd' or 'rmsprop'
		self.model_metrics=model_metrics # always ['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		self.loss=loss # always 'mean_squared_error'
		self.activation_functions=activation_functions # ['tanh','relu','sigmoid'] * lstm_layers
		self.recurrent_activation_functions=recurrent_activation_functions # ['tanh','relu','sigmoid'] * lstm_layers
		self.train_percent=train_percent # [0.6, 0.9] - depends on dataset size
		self.val_percent=val_percent # [0.1, 0.3] - depends on dataset size
		self.amount_companies=amount_companies # depends on the problem to be solved
		self.input_features=list(dict.fromkeys(input_features)) # depends on the features, but great chance of being ['Close']
		self.output_feature=output_feature # depends on the features, but super huge great hyper chance of being 'Close'
		self.index_feature=index_feature  # depends on the features, but super huge great hyper chance of being 'Date'
		self.shuffle=shuffle # [False, True] or [0, 1]
		self.bias=bias  # [False, True] * lstm_layers
		self.use_dense_on_output=use_dense_on_output  # [False, True]
		self.unit_forget_bias=unit_forget_bias  # [False, True] * lstm_layers
		self.go_backwards=go_backwards  # [False, True] * lstm_layers
		self.recurrent_dropout_values=recurrent_dropout_values  # [0.0, 1.0] * lstm_layers
		self.binary_classifier=binary_classifier
		if type(self.bias)==bool:
			self.bias=[self.bias]*self.lstm_layers
		if type(self.unit_forget_bias)==bool:
			self.unit_forget_bias=[self.unit_forget_bias]*self.lstm_layers
		if type(self.go_backwards)==bool:
			self.go_backwards=[self.go_backwards]*self.lstm_layers
		if type(self.dropout_values) in (int,float):
			self.dropout_values=[self.dropout_values]*self.lstm_layers
		if type(self.recurrent_dropout_values) in (int,float):
			self.recurrent_dropout_values=[self.recurrent_dropout_values]*self.lstm_layers
		if type(self.layer_sizes)==int:
			self.layer_sizes=[self.layer_sizes]*self.lstm_layers
		if type(self.activation_functions)==str:
			self.activation_functions=[self.activation_functions]*self.lstm_layers
		if type(self.recurrent_activation_functions)==str:
			self.recurrent_activation_functions=[self.recurrent_activation_functions]*self.lstm_layers
		if len(self.dropout_values)!=self.lstm_layers:
			raise Exception('Wrong dropout_values array size, should be {} instead of {}'.format(self.lstm_layers,len(self.dropout_values)))
		if len(self.layer_sizes)!=self.lstm_layers and not (self.layer_sizes[0]==backwards_samples and len(self.layer_sizes)==self.lstm_layers+1):
			raise Exception('Wrong layer_sizes array size, should be {}'.format(self.lstm_layers))
		if len(self.activation_functions)!=self.lstm_layers:
			raise Exception('Wrong activation_functions array size, should be {}'.format(self.lstm_layers))
		if len(self.recurrent_activation_functions)!=self.lstm_layers:
			raise Exception('Wrong recurrent_activation_functions array size, should be {}'.format(self.lstm_layers))
		if len(self.bias)!=self.lstm_layers:
			raise Exception('Wrong bias array size, should be {}'.format(self.lstm_layers))
		if len(self.unit_forget_bias)!=self.lstm_layers:
			raise Exception('Wrong unit_forget_bias array size, should be {}'.format(self.lstm_layers))
		if len(self.go_backwards)!=self.lstm_layers:
			raise Exception('Wrong go_backwards array size, should be {}'.format(self.lstm_layers))
		if len(self.recurrent_dropout_values)!=self.lstm_layers:
			raise Exception('Wrong recurrent_dropout_values array size, should be {}'.format(self.lstm_layers))
		if len(self.input_features)>1 and self.amount_companies>1:
			raise Exception('Only input_features or amount_companies must be greater than 1')
		if self.val_percent>1 or self.train_percent>1 or self.val_percent<0 or self.train_percent<0:
			raise Exception('Train + validation percent must be smaller than 1 and bigger than 0')
		# constraints
		if self.batch_size%2!=0:
			self.batch_size+=1 # only even batch_sizes
		# patiences cannot be higher than epochs
		self.patience_epochs_stop=min(self.patience_epochs_stop,self.max_epochs)
		self.patience_epochs_reduce=min(self.patience_epochs_reduce,self.max_epochs)
		if self.stateful and (self.batch_size is None or self.batch_size == 0):
			self.batch_size=1 # batch size must be one for stateful
		if len(self.layer_sizes)==self.lstm_layers:
			self.layer_sizes.insert(0,self.backwards_samples)
		self.genAndSetUuid()

	def copy(self):
		name=self.name
		backwards_samples=self.backwards_samples
		forward_samples=self.forward_samples
		lstm_layers=self.lstm_layers
		max_epochs=self.max_epochs
		patience_epochs_stop=self.patience_epochs_stop
		patience_epochs_reduce=self.patience_epochs_reduce
		reduce_factor=self.reduce_factor
		batch_size=self.batch_size
		stateful=self.stateful
		dropout_values=self.dropout_values.copy()
		layer_sizes=self.layer_sizes.copy()
		activation_functions=self.activation_functions.copy()
		recurrent_activation_functions=self.recurrent_activation_functions.copy()
		bias=self.bias.copy()
		normalize=self.normalize
		optimizer=self.optimizer
		model_metrics=self.model_metrics.copy()
		loss=self.loss
		train_percent=self.train_percent
		val_percent=self.val_percent
		amount_companies=self.amount_companies
		input_features=self.input_features.copy()
		output_feature=self.output_feature
		index_feature=self.index_feature
		shuffle=self.shuffle
		use_dense_on_output=self.use_dense_on_output
		unit_forget_bias=self.unit_forget_bias.copy()
		go_backwards=self.go_backwards.copy()
		binary_classifier=self.binary_classifier
		recurrent_dropout_values=self.recurrent_dropout_values.copy()
		new_hyperparams=Hyperparameters(name=name,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values,binary_classifier=binary_classifier)
		return new_hyperparams

	def toString(self):
		string=''
		string+='name: {}'.format(self.name)+', '
		string+='use_dense_on_output: {}'.format(self.use_dense_on_output)+', '
		string+='backwards_samples: {}'.format(self.backwards_samples)+', '
		string+='forward_samples: {}'.format(self.forward_samples)+', '
		string+='lstm_layers: {}'.format(self.lstm_layers)+', '
		string+='max_epochs: {}'.format(self.max_epochs)+', '
		string+='patience_epochs_stop: {}'.format(self.patience_epochs_stop)+', '
		string+='patience_epochs_reduce: {}'.format(self.patience_epochs_reduce)+', '
		string+='reduce_factor: {}'.format(self.reduce_factor)+', '
		string+='batch_size: {}'.format(self.batch_size)+', '
		string+='stateful: {}'.format(self.stateful)+', '
		string+='dropout_values: {}'.format(self.dropout_values)+', '
		string+='layer_sizes: {}'.format(self.layer_sizes)+', '
		string+='activation_functions: {}'.format(self.activation_functions)+', '
		string+='recurrent_activation_functions: {}'.format(self.recurrent_activation_functions)+', '
		string+='unit_forget_bias: {}'.format(self.unit_forget_bias)+', '
		string+='go_backwards: {}'.format(self.go_backwards)+', '
		string+='recurrent_dropout_values: {}'.format(self.recurrent_dropout_values)+', '
		string+='bias: {}'.format(self.bias)+', '
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

	def genAndSetUuid(self,low_resolution=False):
		self.uuid=self.genAndGetUuid(low_resolution=low_resolution)

	def genAndGetUuid(self,low_resolution=False):
		to_hash=self.toString().encode('utf-8')
		if low_resolution:
			hash_object=hashlib.md5(to_hash)
		else:
			hash_object=hashlib.sha256(to_hash)
		return hash_object.hexdigest()

	@staticmethod
	def jsonDecoder(obj):
		if '__type__' in obj and obj['__type__'] == 'Hyperparameters':
			return Hyperparameters(name=obj['name'],input_features=obj['input_features'],output_feature=obj['output_feature'],index_feature=obj['index_feature'],backwards_samples=obj['backwards_samples'],forward_samples=obj['forward_samples'],lstm_layers=obj['lstm_layers'],max_epochs=obj['max_epochs'],patience_epochs_stop=obj['patience_epochs_stop'],patience_epochs_reduce=obj['patience_epochs_reduce'],reduce_factor=obj['reduce_factor'],batch_size=obj['batch_size'],stateful=obj['stateful'],dropout_values=obj['dropout_values'],layer_sizes=obj['layer_sizes'],normalize=obj['normalize'],optimizer=obj['optimizer'],model_metrics=obj['model_metrics'],loss=obj['loss'],train_percent=obj['train_percent'],val_percent=obj['val_percent'],amount_companies=obj['amount_companies'],shuffle=obj['shuffle'],activation_functions=obj['activation_functions'],recurrent_activation_functions=obj['recurrent_activation_functions'],bias=obj['bias'],use_dense_on_output=obj['use_dense_on_output'],unit_forget_bias=obj['unit_forget_bias'],go_backwards=obj['go_backwards'],recurrent_dropout_values=obj['recurrent_dropout_values'],binary_classifier=obj['binary_classifier'])
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

	def setName(self,name):
		self.name=name
		self.genAndSetUuid()

	@staticmethod
	def valueToClass(value):
		threshold=.5
		return 1 if float(value)>threshold else 0

	@staticmethod
	def getFeatureGroups():
		return [
				[Features.OC,Features.OH,Features.OL,Features.CH,Features.CL,Features.LH,Features.VOLUME,Features.LOG_RETURN,Features.FAST_EXP_MOVING_AVG,Features.SLOW_EXP_MOVING_AVG],
				[Features.OC,Features.OH,Features.OL,Features.CH,Features.CL,Features.LH,Features.VOLUME,Features.LOG_RETURN,Features.FAST_EXP_MOVING_AVG,Features.SLOW_EXP_MOVING_AVG,Features.ADJ_CLOSE,Features.FAST_MOVING_AVG,Features.SLOW_MOVING_AVG],
				[],
				[Features.OPEN,Features.HIGH,Features.LOW,Features.ADJ_CLOSE,Features.VOLUME],
				[Features.OPEN,Features.HIGH,Features.LOW,Features.ADJ_CLOSE,Features.VOLUME,Features.OC,Features.OH,Features.OL,Features.CH,Features.CL,Features.LH,Features.FAST_EXP_MOVING_AVG,Features.SLOW_EXP_MOVING_AVG,Features.LOG_RETURN,Features.FAST_MOVING_AVG,Features.SLOW_MOVING_AVG]
				]



	
