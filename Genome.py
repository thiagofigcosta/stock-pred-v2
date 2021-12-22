#!/bin/python
# -*- coding: utf-8 -*-

from SearchSpace import SearchSpace
from Enums import Metric,NodeType,Loss,Optimizers,Features
from Hyperparameters import Hyperparameters
from Utils import Utils

class Genome(object):

	COMPRESS_WEIGHTS=True
	ENCODE_B64_WEIGHTS=True
	ENCODE_B65_WEIGHTS=False
	CACHE_WEIGHTS=True
	CACHE_FOLDER='neural_genome_cache'
	
	def __init__(self, search_space, eval_callback, is_neural=False, has_age=False):
		self.limits=search_space
		self.dna=[]
		for limit in search_space:
			if limit.data_type==SearchSpace.Type.INT:
				self.dna.append(Utils.randomInt(limit.min_value,limit.max_value))
			elif limit.data_type==SearchSpace.Type.FLOAT:
				self.dna.append(Utils.randomFloat(limit.min_value,limit.max_value))
			elif limit.data_type==SearchSpace.Type.BOOLEAN:
				self.dna.append(limit.max_value if Utils.random()>.5 else limit.min_value)
			else:
				raise Exception('Unkown search space data type {}'.format(limit.data_type))
		self.eval_callback=eval_callback
		self.is_neural=is_neural
		self.mt_dna=''
		self.fitness=0
		self.output=0
		self.gen=-1
		if has_age:
			self.age=0
		else:
			self.age=None
		self.id=Utils.randomUUID()
		self.resetMtDna()
		if self.is_neural:
			self._weights=None
			self.cached=False
			self.cache_file=self.genCacheFilename()

	def __del__(self):
		if self.is_neural and self.cached and Genome.CACHE_WEIGHTS:
			Utils.deleteFile(self.cache_file)
		self._weights=None

	def __lt__(self, other):
		return self.fitness < other.fitness or (self.fitness == other.fitness and self.age is not None and other.age is not None and self.age < other.age)

	def __str__(self):
		return self.toString()

	def makeChild(self, dna):
		mother=self
		child=mother.copy()
		child.id=Utils.randomUUID()
		child.dna=dna+[] # deep copy
		child.fitness=0
		child.output=0
		if child.age is not None:
			child.age=0
		return child

	def evaluate(self):
		self.output=self.eval_callback(self) 

	def fixlimits(self):
		for i in range(len(self.dna)):
			self.dna[i]=self.limits[i].fixValue(self.dna[i])
			if self.limits[i].data_type==SearchSpace.Type.INT:
				self.dna[i]=int(self.dna[i])
			elif self.limits[i].data_type==SearchSpace.Type.FLOAT:
				self.dna[i]=float(self.dna[i])
			elif self.limits[i].data_type==SearchSpace.Type.BOOLEAN:
				self.dna[i]=bool(self.dna[i])
			else:
				raise Exception('Unkown search space data type {}'.format(self.limits[i].data_type))

	def toString(self):
		out='Output: {} Fitness: {}'.format(self.output,self.fitness)
		if self.gen > -1:
			out+=' gen: {}'.format(self.gen)
		if self.age is not None:
			out+=' age: {}'.format(self.age)
		out+=' DNA: ['
		for i in range(len(self.dna)):
			out+=' '
			if self.limits[i].name is not None:
				out+=self.limits[i].name+': '
			out+=str(self.dna[i])
			if i+1<len(self.dna):
				out+=','
			else:
				out+=' '
		out+=']'
		return out


	def resetMtDna(self):
		self.mt_dna=Utils.randomUUID()

	def hasWeights(self):
		return (self._weights is not None and len(self._weights)>0) or (self.cached and Genome.CACHE_WEIGHTS)


	def forceCache(self):
		if (self.cached and Genome.CACHE_WEIGHTS):
			Utils.deleteFile(self.cache_file)
			self.cached=False 
		if (not self.cached and Genome.CACHE_WEIGHTS):
			success=False
			tries=0
			max_tries=5
			while (not success and tries<max_tries):
				tries+=1
				try:
					Utils.saveObj(self._weights,self.cache_file)
					success=True
				except Exception as e:
					print(e)
					if (tries==max_tries-1):
						self.cache_file=self.genCacheFilename()
			if success:
				self.cached=True
				self._weights=None
	
	def getWeights(self,raw=False):
		if (self.cached and Genome.CACHE_WEIGHTS):
			self._weights=Utils.loadObj(self.cache_file)
		if raw:
			return self._weights
		else:
			return Genome.decodeWeights(self._weights)

	def setWeights(self,weights,raw=False):
		if raw:
			self._weights=weights
		else:
			self._weights=Genome.encodeWeights(weights)
		self.forceCache()

	@staticmethod
	def encodeWeights(weights):
		if weights is None:
			return weights
		wei_str=Utils.objToJsonStr(weights,compress=Genome.COMPRESS_WEIGHTS,b64=Genome.ENCODE_B64_WEIGHTS)
		if Genome.ENCODE_B64_WEIGHTS and Genome.ENCODE_B65_WEIGHTS:
			wei_str=Utils.base64ToBase65(wei_str)
		return wei_str

	@staticmethod
	def decodeWeights(weights):
		if weights is None:
			return weights
		wei_str=weights
		if Genome.ENCODE_B64_WEIGHTS and Genome.ENCODE_B65_WEIGHTS:
			wei_str=Utils.base65ToBase64(wei_str)
		return Utils.jsonStrToObj(wei_str,compress=Genome.COMPRESS_WEIGHTS,b64=Genome.ENCODE_B64_WEIGHTS)

	def clearMemoryWeightsIfCached(self):
		if (self.cached and Genome.CACHE_WEIGHTS):
			self._weights=None

	def clearWeights(self):
		self._weights=None
		if self.is_neural and self.cached and Genome.CACHE_WEIGHTS:
			Utils.deleteFile(self.cache_file)
			self.cached=False

	def genCacheFilename(self):
		filename='{}-{}.weights_cache'.format(self.id,Utils.randomUUID())
		Utils.createFolderIfNotExists(Genome.CACHE_FOLDER)
		return Utils.joinPath(Genome.CACHE_FOLDER,filename)

	def copy(self):
		that=Genome([], self.eval_callback, is_neural=self.is_neural)
		that.limits=self.limits.copy()
		that.dna=self.dna+[] # deep copy
		that.mt_dna=self.mt_dna
		that.fitness=self.fitness
		that.output=self.output
		that.age=self.age
		that.id=self.id
		that.gen=self.gen
		if self.is_neural:
			that._weights=self._weights # string copy
			that.cached=self.cached
			that.cache_file=self.genCacheFilename()
			if self.is_neural and self.cached and Genome.CACHE_WEIGHTS:
				Utils.copyFile(self.cache_file,that.cache_file)

		return that

	@staticmethod
	def enrichSearchSpace(search_space):
		# mandatory
		backwards_samples=search_space['backwards_samples']
		forward_samples=search_space['forward_samples']
		max_epochs=search_space['max_epochs']
		stateful=search_space['stateful']
		batch_size=search_space['batch_size']
		use_dense_on_output=search_space['use_dense_on_output']
		#optional
		patience_epochs_stop=search_space['patience_epochs_stop']
		patience_epochs_reduce=search_space['patience_epochs_reduce']
		reduce_factor=search_space['reduce_factor']
		normalize=search_space['normalize']
		optimizer=search_space['optimizer']

		shuffle=search_space['shuffle']
		# layer dependent
		lstm_layers=search_space['lstm_layers']
		layer_sizes=search_space['layer_sizes']
		activation_functions=search_space['activation_functions']
		recurrent_activation_functions=search_space['recurrent_activation_functions']
		dropout_values=search_space['dropout_values']
		recurrent_dropout_values=search_space['recurrent_dropout_values']
		bias=search_space['bias']
		unit_forget_bias=search_space['unit_forget_bias']
		go_backwards=search_space['go_backwards']

		if patience_epochs_stop is None:
			patience_epochs_stop=SearchSpace.Dimension(SearchSpace.Type.INT,0,0,name='patience_epochs_stop')
		if patience_epochs_reduce is None:
			patience_epochs_reduce=SearchSpace.Dimension(SearchSpace.Type.INT,0,0,name='patience_epochs_reduce')
		if reduce_factor is None:
			reduce_factor=SearchSpace.Dimension(SearchSpace.Type.FLOAT,0.0001,0.2,name='reduce_factor')
		if normalize is None:
			normalize=SearchSpace.Dimension(SearchSpace.Type.BOOLEAN,True,True,name='normalize')
		if optimizer is None:
			optimizer=SearchSpace.Dimension(SearchSpace.Type.INT,Optimizers.RMSPROP,Optimizers.RMSPROP,name='optimizer')
		if activation_functions is None:
			activation_functions=SearchSpace.Dimension(SearchSpace.Type.INT,NodeType.TANH,NodeType.TANH,name='activation_functions')
		if recurrent_activation_functions is None:
			recurrent_activation_functions=SearchSpace.Dimension(SearchSpace.Type.INT,NodeType.SIGMOID,NodeType.SIGMOID,name='recurrent_activation_functions')
		if shuffle is None:
			shuffle=SearchSpace.Dimension(SearchSpace.Type.BOOLEAN,False,False,name='shuffle')


		enriched_search_space=SearchSpace()
		enriched_search_space.add(backwards_samples.min_value,backwards_samples.max_value,backwards_samples.data_type,backwards_samples.name)
		enriched_search_space.add(forward_samples.min_value,forward_samples.max_value,forward_samples.data_type,forward_samples.name)
		enriched_search_space.add(max_epochs.min_value,max_epochs.max_value,max_epochs.data_type,max_epochs.name)
		enriched_search_space.add(stateful.min_value,stateful.max_value,stateful.data_type,stateful.name)
		enriched_search_space.add(batch_size.min_value,batch_size.max_value,batch_size.data_type,batch_size.name)
		enriched_search_space.add(use_dense_on_output.min_value,use_dense_on_output.max_value,use_dense_on_output.data_type,use_dense_on_output.name)

		enriched_search_space.add(patience_epochs_stop.min_value,patience_epochs_stop.max_value,patience_epochs_stop.data_type,patience_epochs_stop.name)
		enriched_search_space.add(patience_epochs_reduce.min_value,patience_epochs_reduce.max_value,patience_epochs_reduce.data_type,patience_epochs_reduce.name)
		enriched_search_space.add(reduce_factor.min_value,reduce_factor.max_value,reduce_factor.data_type,reduce_factor.name)
		enriched_search_space.add(normalize.min_value,normalize.max_value,normalize.data_type,normalize.name)
		enriched_search_space.add(optimizer.min_value,optimizer.max_value,optimizer.data_type,optimizer.name)
		enriched_search_space.add(shuffle.min_value,shuffle.max_value,shuffle.data_type,shuffle.name)
		
		enriched_search_space.add(lstm_layers.min_value,lstm_layers.max_value,lstm_layers.data_type,lstm_layers.name)
		for l in range(lstm_layers.max_value):
			enriched_search_space.add(layer_sizes.min_value,layer_sizes.max_value,layer_sizes.data_type,layer_sizes.name+'_{}'.format(l))
			enriched_search_space.add(activation_functions.min_value,activation_functions.max_value,activation_functions.data_type,activation_functions.name+'_{}'.format(l))
			enriched_search_space.add(recurrent_activation_functions.min_value,recurrent_activation_functions.max_value,recurrent_activation_functions.data_type,recurrent_activation_functions.name+'_{}'.format(l))
			enriched_search_space.add(dropout_values.min_value,dropout_values.max_value,dropout_values.data_type,dropout_values.name+'_{}'.format(l))
			enriched_search_space.add(recurrent_dropout_values.min_value,recurrent_dropout_values.max_value,recurrent_dropout_values.data_type,recurrent_dropout_values.name+'_{}'.format(l))
			enriched_search_space.add(bias.min_value,bias.max_value,bias.data_type,bias.name+'_{}'.format(l))
			enriched_search_space.add(unit_forget_bias.min_value,unit_forget_bias.max_value,unit_forget_bias.data_type,unit_forget_bias.name+'_{}'.format(l))
			enriched_search_space.add(go_backwards.min_value,go_backwards.max_value,go_backwards.data_type,go_backwards.name+'_{}'.format(l))
		return enriched_search_space

	def toHyperparameters(self,input_features,output_feature,index_feature,metrics,loss,train_percent,val_percent,amount_companies,binary_classifier):
		return Genome.dnaToHyperparameters(self.dna,self.id,input_features,output_feature,index_feature,metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)

	@staticmethod
	def dnaToHyperparameters(dna,ind_id,input_features,output_feature,index_feature,metrics,loss,train_percent,val_percent,amount_companies,binary_classifier):
		if type(input_features) is not list:
			input_features=[input_features]
		for i in range(len(input_features)):
			if isinstance(input_features[i],Features):
				input_features[i]=input_features[i].toDatasetName()
		if isinstance(output_feature,Features):
			output_feature=output_feature.toDatasetName()
		if isinstance(index_feature,Features):
			index_feature=index_feature.toDatasetName()
		if type(metrics) is not list:
			metrics=[metrics]
		for i in range(len(metrics)):
			if isinstance(metrics[i],Metric):
				metrics[i]=metrics[i].toKerasName()
		if isinstance(loss,Loss):
			loss=loss.toKerasName()

		backwards_samples=int(dna[0])
		forward_samples=int(dna[1])
		max_epochs=int(dna[2])
		stateful=bool(dna[3])
		batch_size=int(dna[4])
		use_dense_on_output=bool(dna[5])

		patience_epochs_stop=int(dna[6])
		patience_epochs_reduce=int(dna[7])
		reduce_factor=float(dna[8])
		normalize=bool(dna[9])
		optimizer=Optimizers(dna[10]).toKerasName()
		shuffle=bool(dna[11])

		lstm_layers=int(dna[12])
		first_layer_dependent=13
		layer_sizes=[]
		activation_funcs=[]
		recurrent_activation_funcs=[]
		dropouts=[]
		recurrent_dropouts=[]
		bias=[]
		unit_forget_bias=[]
		go_backwards=[]
		amount_of_dependent=8
		for l in range(lstm_layers):
			layer_sizes.append(int(dna[(first_layer_dependent+0)+amount_of_dependent*l]))
			activation_funcs.append(NodeType(dna[(first_layer_dependent+1)+amount_of_dependent*l]).toKerasName())
			recurrent_activation_funcs.append(NodeType(dna[(first_layer_dependent+2)+amount_of_dependent*l]).toKerasName())
			dropouts.append(float(dna[(first_layer_dependent+3)+amount_of_dependent*l]))
			recurrent_dropouts.append(float(dna[(first_layer_dependent+4)+amount_of_dependent*l]))
			bias.append(bool(dna[(first_layer_dependent+5)+amount_of_dependent*l]))
			unit_forget_bias.append(bool(dna[(first_layer_dependent+6)+amount_of_dependent*l]))
			go_backwards.append(bool(dna[(first_layer_dependent+7)+amount_of_dependent*l]))
		
		hyperparameters=Hyperparameters(name=ind_id,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropouts,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_funcs,recurrent_activation_functions=recurrent_activation_funcs,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropouts,binary_classifier=binary_classifier)
		return hyperparameters
