#!/bin/python3
# -*- coding: utf-8 -*-

import os
from pickle import NONE
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils
from Dataset import Dataset
from Enums import NodeType,Loss,Metric,Optimizers,Features
from SearchSpace import SearchSpace
from Genome import Genome
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
import numpy as np
import pathos.pools as pp



class StockPredNAS(Problem):

	WORST_VALUE=2147483647
	BEST_VALUE=-2147483647

	def __init__(self,search_space,input_features,output_feature,index_feature,metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,pop_size=50,children_per_gen=50,eliminate_duplicates=False):
		objectives=2
		
		x_lower_limit = []
		x_upper_limit = []
		x_types = []
		for limit in search_space:
			x_lower_limit.append(limit.min_value)
			x_upper_limit.append(limit.max_value)
			if limit.data_type==SearchSpace.Type.INT:
				x_types.append('int')
			elif limit.data_type==SearchSpace.Type.FLOAT:
				x_types.append('real')
			elif limit.data_type==SearchSpace.Type.BOOLEAN:
				x_types.append('bin')
			else:
				raise Exception('Unkown search space data type {}'.format(limit.data_type))
		
		x_lower_limit=np.array(x_lower_limit)
		x_upper_limit=np.array(x_upper_limit)

		self.input_features=input_features
		self.output_feature=output_feature
		self.index_feature=index_feature
		self.metrics=metrics
		self.loss=loss
		self.train_percent=train_percent
		self.val_percent=val_percent
		self.amount_companies=amount_companies
		self.binary_classifier=binary_classifier

		self.sampling = MixedVariableSampling(x_types, {
			'real': get_sampling("real_random"),  #'real_random’, ‘real_lhs’, ‘perm_random’
			'int': get_sampling("int_random"),
			'bin': get_sampling('bin_random')
		})

		self.crossover = MixedVariableCrossover(x_types, {
			'real': get_crossover("real_sbx", prob=1.0, eta=3.0), # 0.1 <= prob <= 1.0, 1.0 <= eta <= 30.0
			'int': get_crossover("int_sbx", prob=1.0, eta=3.0),
			'bin': get_crossover('bin_hux')
		})

		self.mutation = MixedVariableMutation(x_types, {
			'real': get_mutation("real_pm", eta=3.0), # 0.01 <= prob <= 1.0, 10 <= eta <= 40
			'int': get_mutation("int_pm", eta=3.0),
			'bin': get_mutation('bin_bitflip')
		})

		self.algorithm = NSGA2(
			pop_size=pop_size,
			n_offsprings=children_per_gen,
			sampling=self.sampling,
			crossover=self.crossover,
			mutation=self.mutation,
			eliminate_duplicates=eliminate_duplicates
		)
		
		super().__init__(n_var=len(x_types), n_obj=objectives, n_constr=0, xl=x_lower_limit, xu=x_upper_limit)

	@staticmethod
	def _trainCallback(i_and_individual,gen,stock,filepath,test_date,input_features,output_feature,index_feature,metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,train=True,test=True):
		i,individual=i_and_individual
		mse=None
		ok_rate=None
		just_train=train and not test
		try:
			ind_id=StockPredNAS.getIndId(i,gen=gen)
			hyperparameters=Genome.dnaToHyperparameters(individual,ind_id,input_features,output_feature,index_feature,metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)
			neuralNetwork=NeuralNetwork(hyperparameters,stock_name=stock,verbose=False)
			if train:
				neuralNetwork.loadDataset(filepath,plot=False,blocking_plots=False,save_plots=True)
				neuralNetwork.buildModel(plot_model_to_file=True)
				neuralNetwork.train()
				neuralNetwork.restoreCheckpointWeights(delete_after=False)
				neuralNetwork.save()
			if test:
				if not train:
					neuralNetwork.load()
				neuralNetwork.loadTestDataset(filepath,from_date=test_date,blocking_plots=False,save_plots=True)
				neuralNetwork.eval(plot=True,print_prediction=False,blocking_plots=False,save_plots=True)
				try:
					mse=neuralNetwork.metrics['test']['Model Metrics']['mean_squared_error']
				except Exception as e:
					print(str(e))
				try:
					ok_rate=neuralNetwork.metrics['test']['Class Metrics']['OK_Rate']/100.0
				except Exception as e:
					print(str(e))

			neuralNetwork.destroy()
		except Exception as e:
			print(str(e))
			if just_train:
				return False

		if just_train:
			return True

		if mse is None or mse!=mse:
			mse=StockPredNAS.WORST_VALUE

		if ok_rate is None or ok_rate!=ok_rate:
			ok_rate=StockPredNAS.WORST_VALUE
		else:
			# since the it is a minimization problem
			ok_rate*=-1

		return {'mse':mse,'ok_rate':ok_rate}

	def _evaluate(self, x, out, *args, **kwargs):
		generation_mse=[]	
		generation_ok_rate=[]	

		if self.parallelism==1:
			for i,individual in enumerate(x):
				metrics=StockPredNAS._trainCallback((i,individual),self.gen,self.stock,self.filepath,self.test_date,self.input_features,self.output_feature,self.index_feature,self.metrics,self.loss,self.train_percent,self.val_percent,self.amount_companies,self.binary_classifier)
				generation_mse.append(metrics['mse'])
				generation_ok_rate.append(metrics['ok_rate'])
		else:
			a=[self.gen]*len(x)
			b=[self.stock]*len(x)
			c=[self.filepath]*len(x)
			d=[self.test_date]*len(x)
			e=[self.input_features]*len(x)
			f=[self.output_feature]*len(x)
			g=[self.index_feature]*len(x)
			h=[self.metrics]*len(x)
			i=[self.loss]*len(x)
			j=[self.train_percent]*len(x)
			k=[self.val_percent]*len(x)
			l=[self.amount_companies]*len(x)
			m=[self.binary_classifier]*len(x)
			n=[True]*len(x) # train
			o=[False]*len(x) # test
			with pp.ThreadPool(self.parallelism,maxtasksperchild=None) as pool:
				# outputs=pool.map(StockPredNAS._trainCallback, enumerate(x),a,b,c,d,e,f,g,h,i,j,k,l,m) # cannot plot because matplot is not thread safe
				success=pool.imap(StockPredNAS._trainCallback, enumerate(x),a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) # imap is non blocking
				error=0
				for ind in success:
					if not ind:
						error+=1
				if error>0:
					print('Got {} errors while evaluating...'.format(error))

			# testing and plotting
			for i,individual in enumerate(x):
				metrics=StockPredNAS._trainCallback((i,individual),self.gen,self.stock,self.filepath,self.test_date,self.input_features,self.output_feature,self.index_feature,self.metrics,self.loss,self.train_percent,self.val_percent,self.amount_companies,self.binary_classifier,False,True)
				generation_mse.append(metrics['mse'])
				generation_ok_rate.append(metrics['ok_rate'])

		self.gen+=1
		out["F"] = np.column_stack([generation_mse, generation_ok_rate])


	@staticmethod
	def getIndId(i,gen=None,individual=None):
		if gen is None:
			gen=individual.data['n_gen']
		return 'gen: {} - ind: {}'.format(gen,i)

	def optmize(self,stock,filepath,test_date,max_evals=1000,parallelism=1,verbose=True,store_metrics=True):
		if parallelism==0:
			parallelism=os.cpu_count()
		elif parallelism < 0:
			parallelism=1
		if parallelism != 1:
			print('Parallelism: {}'.format(parallelism))
			NeuralNetwork.setFigureManagerFromMainThread()
		self.parallelism=parallelism
		self.gen=0
		self.stock=stock
		self.filepath=filepath
		self.test_date=test_date
		res = minimize(
			self,
			self.algorithm,
			termination=('n_eval', max_evals),
			save_history=store_metrics,
			verbose=verbose
		)
		# parse history
		history=[]
		if store_metrics:
			for gen in res.history:
				individuals=[]
				results=[]
				for ind in gen.pop:
					individuals.append(ind.X.tolist())
					results.append(ind.F.tolist())
				gen_vanilla={'individual':individuals,'result':results}
				history.append(gen_vanilla)
		self.history=history

		solution=[]
		# parse solutions
		for i,ind in enumerate(res.opt):
			solution.append({'individual':ind.X.tolist(),'result':ind.F.tolist(),'id':StockPredNAS.getIndId(-i,individual=ind)}) # negative index for solution
		return solution


feature_group=0 #0-6
binary_classifier=False
stock=os.getenv('NAS_STOCK', default='T')
print('Running for `{}` stock'.format(stock))

input_features=Hyperparameters.getFeatureGroups()

never_crawl=os.getenv('NEVER_CRAWL',default='False')
never_crawl=never_crawl.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')

search_space=SearchSpace()
search_space.add(27,34,SearchSpace.Type.INT,'backwards_samples')
search_space.add(7,7,SearchSpace.Type.INT,'forward_samples')
search_space.add(0,2,SearchSpace.Type.INT,'lstm_layers')
search_space.add(2900,3400,SearchSpace.Type.INT,'max_epochs')
search_space.add(1700,1900,SearchSpace.Type.INT,'patience_epochs_stop')
search_space.add(200,250,SearchSpace.Type.INT,'patience_epochs_reduce')
search_space.add(0.045,0.055,SearchSpace.Type.FLOAT,'reduce_factor')
search_space.add(40,50,SearchSpace.Type.INT,'batch_size')
search_space.add(False,False,SearchSpace.Type.BOOLEAN,'stateful')
search_space.add(True,True,SearchSpace.Type.BOOLEAN,'normalize')
search_space.add(Optimizers.ADAM,Optimizers.RMSPROP,SearchSpace.Type.INT,'optimizer')
search_space.add(NodeType.RELU,NodeType.TANH,SearchSpace.Type.INT,'activation_functions')
search_space.add(NodeType.RELU,NodeType.TANH,SearchSpace.Type.INT,'recurrent_activation_functions')
search_space.add(False,False,SearchSpace.Type.BOOLEAN,'shuffle')
search_space.add(False,False,SearchSpace.Type.BOOLEAN,'use_dense_on_output')
search_space.add(60,74,SearchSpace.Type.INT,'layer_sizes')
search_space.add(0.20,0.28,SearchSpace.Type.FLOAT,'dropout_values')
search_space.add(0.12,0.17,SearchSpace.Type.FLOAT,'recurrent_dropout_values')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'bias')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'unit_forget_bias')
search_space.add(False,False,SearchSpace.Type.BOOLEAN,'go_backwards')
search_space=Genome.enrichSearchSpace(search_space)
amount_companies=1
train_percent=.8
val_percent=.3
if binary_classifier:
	input_features=[Features.UP]+input_features[feature_group]
	output_feature=Features.UP
	index_feature='Date'
	model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
	loss='categorical_crossentropy'
else:
	input_features=[Features.CLOSE]+input_features[feature_group]
	output_feature=Features.CLOSE
	index_feature='Date'
	model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
	loss='mean_squared_error'

start_date='01/01/2016' #Utils.FIRST_DATE
end_date='07/05/2021'
test_date='07/01/2021'
start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))
filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
crawler=Crawler()
filepath=crawler.getDatasetPath(filename)
if not Utils.checkIfPathExists(filepath) and not never_crawl:
	crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
	NeuralNetwork.enrichDataset(filepath)
	


parallelism=0
verbose_genetic=True
population_size=30
offspring_size=30
eliminate_duplicates=True
max_evals=500
store_metrics=False
notables=5


print('Instantiating the problem...')
stock_pred_nas=StockPredNAS(search_space,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,pop_size=population_size,children_per_gen=offspring_size,eliminate_duplicates=eliminate_duplicates)
print('Instantiating the problem...OK')
print('Optmizing...')
solutions=stock_pred_nas.optmize(stock,filepath,test_date,max_evals=max_evals,parallelism=parallelism,verbose=verbose_genetic,store_metrics=store_metrics)
print('Optmizing...OK')
print()
print()
print()
print('Solutions')
for i,sol in enumerate(solutions[:notables]):
	print('{} -> {}: {}'.format(i,sol['individual'],sol['result']))
print()
print()
print()

print('Evaluating solutions...')
# TODO use on ind_id the hash of individuals dna, avoiding to train again during the test callback
def test_callback(dna,ind_id):
	global stock,filepath,test_date,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,start_date_formated_for_file,end_date_formated_for_file
	hyperparameters=Genome.dnaToHyperparameters(dna,ind_id,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)
	
	neuralNetwork=NeuralNetwork(hyperparameters,stock_name=stock,verbose=False)
	neuralNetwork.loadDataset(filepath,plot=False,blocking_plots=False,save_plots=False)
	neuralNetwork.buildModel(plot_model_to_file=True)
	neuralNetwork.train()
	neuralNetwork.restoreCheckpointWeights(delete_after=False)
	neuralNetwork.save()
	neuralNetwork.loadTestDataset(filepath,from_date=test_date,blocking_plots=False,save_plots=True)
	neuralNetwork.eval(plot=True,print_prediction=False,blocking_plots=False,save_plots=True)
	neuralNetwork.destroy()
	

for sol in solutions[:notables]:
	test_callback(sol['individual'],sol['id'])


NeuralNetwork.runPareto(use_ok_instead_of_f1=True,plot=True,blocking_plots=False,save_plots=True,label='{}-{}'.format(start_date_formated_for_file,end_date_formated_for_file))


print('Evaluating solutions...OK!')
