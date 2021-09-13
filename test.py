#!/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils
from Dataset import Dataset
from Enums import NodeType,Loss,Metric,Optimizers,Features
from StandardGeneticAlgorithm import StandardGeneticAlgorithm
from EnhancedGeneticAlgorithm import EnhancedGeneticAlgorithm
from PopulationManager import PopulationManager
from HallOfFame import HallOfFame
from Genome import Genome
from SearchSpace import SearchSpace

def dataset_test():
	stock_value=[1,2,3,4,5,6,7,8,9,10]
	features_values=[[10],[20],[30],[40],[50],[60],[70],[80],[90],100]
	dates=Utils.getStrNextNWorkDays('17/06/2021',len(stock_value))
	# -------------------
	print('-------------------')
	dataset=Dataset(name='OriginalGE')
	dataset.addCompany(stock_value)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Values:',dataset.getValues())
	print('Values Main:',dataset.getValues(only_main_value=True))
	# -------------------
	print('-------------------')
	dataset=Dataset(name='OriginalGE')
	dataset.addCompany(stock_value,dates)
	print('Len:',dataset.getSize())
	dataset.print()
	# -------------------
	print('-------------------')
	dataset=Dataset(name='OriginalGE')
	dataset.addCompany(stock_value,dates,features_values)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Indexes:',dataset.getIndexes())
	print('Values:',dataset.getValues())
	print('Values Main:',dataset.getValues(only_main_value=True))
	# -------------------
	print('-------------------')
	dataset.convertToTemporalValues(3,2)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Indexes:',dataset.getIndexes())
	print('Values:',dataset.getValues())
	print('Values Main:',dataset.getValues(only_main_value=True))
	# -------------------
	print('-------------------')
	dataset_reverted=dataset.copy()
	dataset_reverted.name+=' copy'
	dataset_reverted.revertFromTemporalValues()
	print('Len:',dataset_reverted.getSize())
	dataset_reverted.print()
	print('Indexes:',dataset_reverted.getIndexes())
	print('Values:',dataset_reverted.getValues())
	# -------------------
	print('-------------------')
	print('Train data:')
	start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays()
	print('Neural Network Start Index:',start_index)
	print('Neural Network X Shape:',dataset_x.shape)
	print('Neural Network Y Shape:',dataset_y.shape)
	print()
	print('Full data:')
	start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(include_test_data=True)
	print('Neural Network Start Index:',start_index)
	print('Neural Network X Shape:',dataset_x.shape)
	print('Neural Network Y Shape:',dataset_y.shape)
	print()
	print('To predict data:')
	start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(only_test_data=True)
	print('Neural Network Start Index:',start_index)
	print('Neural Network X Shape:',dataset_x.shape)
	print('Neural Network Y Shape:',dataset_y.shape)
	# -------------------
	print('-------------------')
	stock_value_2=[-el for el in stock_value]
	dataset=Dataset(name='OriginalGE and OriginalGE*-1')
	dataset.addCompany(stock_value,dates,features_values)
	dataset.addCompany(stock_value_2,dates,features_values)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Indexes:',dataset.getIndexes())
	print('Values:',dataset.getValues())
	print('Values Main:',dataset.getValues(only_main_value=True))
	print('Values Splitted by Feature:',dataset.getValuesSplittedByFeature())
	# -------------------
	print('-------------------')
	dataset.convertToTemporalValues(3,2)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Indexes:',dataset.getIndexes())
	print('Values:',dataset.getValues())
	print('Values Main:',dataset.getValues(only_main_value=True))
	# -------------------
	print('-------------------')
	print('Full data:')
	start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(include_test_data=True)
	print('Neural Network Start Index:',start_index)
	print('Neural Network X Shape:',dataset_x.shape)
	print('Neural Network Y Shape:',dataset_y.shape)
	print()
	print('To predict data:')
	start_index_to_pred,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(only_test_data=True)
	print('Neural Network Start Index:',start_index_to_pred)
	print('Neural Network X Shape:',dataset_x.shape)
	print('Neural Network Y Shape:',dataset_y.shape)
	print()
	print('Train data:')
	start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays()
	print('Neural Network Start Index:',start_index)
	print('Neural Network X Shape:',dataset_x.shape)
	print('Neural Network Y Shape:',dataset_y.shape)
	start_index_part_2,dataset_x_p1,dataset_x_p2=Dataset.splitNeuralNetworkArray(dataset_x,.7)
	print('Train Data Splitted Start Index:',start_index_part_2)
	print('Neural Network X Splitted P1 Shape:',dataset_x_p1.shape)
	print('Neural Network X Splitted P1 Shape:',dataset_x_p2.shape)
	print()
	# -------------------
	print('-------------------')
	dataset_reverted=dataset.copy()
	dataset_reverted.name+=' copy'
	dataset_reverted.revertFromTemporalValues()
	print('Len:',dataset_reverted.getSize())
	dataset_reverted.print()
	print('Indexes:',dataset_reverted.getIndexes())
	print('Values:',dataset_reverted.getValues())
	# -------------------
	print('-------------------')
	correct_predictions=np.array([[[10, -10], [11, -11]],[[11, -11], [12, -12]]])
	print('Correct preds shape',correct_predictions.shape)
	dataset.setNeuralNetworkResultArray(start_index_to_pred,correct_predictions)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Indexes:',dataset.getIndexes())
	print('Values:',dataset.getValues())
	print('Indexes degree 1:',dataset.getIndexes(degree=1))
	print('Values degree 1:',dataset.getValues(degree=1))
	print('Indexes degree 2:',dataset.getIndexes(degree=2))
	print('Values degree 2:',dataset.getValues(degree=2))
	# -------------------
	print('-------------------')
	correct_predictions=np.array([[10, -10,  11, -11],[11, -11,12, -12]])
	print('Correct preds shape',correct_predictions.shape)
	dataset.setNeuralNetworkResultArray(start_index_to_pred,correct_predictions)
	print('Len:',dataset.getSize())
	dataset.print()
	print('Indexes:',dataset.getIndexes())
	print('Values:',dataset.getValues())
	print('Indexes degree 1:',dataset.getIndexes(degree=1))
	print('Values degree 1:',dataset.getValues(degree=1))
	print('Indexes degree 2:',dataset.getIndexes(degree=2))
	print('Values degree 2:',dataset.getValues(degree=2))
	# -------------------
	print('-------------------')
	dataset_reverted=dataset.copy()
	dataset_reverted.name+=' copy'
	dataset_reverted.revertFromTemporalValues()
	print('Len:',dataset_reverted.getSize())
	dataset_reverted.print()
	print('Indexes:',dataset_reverted.getIndexes())
	print('Values:',dataset_reverted.getValues())
	print()
	indexes,preds=dataset_reverted.getDatesAndPredictions()
	print('Pred Indexes:',indexes)
	print('Pred Values:',preds)
	print('Indexes degree 1:',dataset.getIndexes(degree=1))
	print('Values degree 1:',dataset.getValues(degree=1))
	print('Indexes degree 2:',dataset.getIndexes(degree=2))
	print('Values degree 2:',dataset.getValues(degree=2))
	print('\t*Inner dimmension = companies | After inner dimmension = multiple predictions')
	# -------------------
	print('-------------------')
	dataset=Dataset(name='OriginalGE')
	dataset.addCompany(stock_value,dates,features_values)
	print('Values:',dataset.getValues())
	print('Max:',dataset.getAbsMaxes())
	# -------------------
	print('-------------------')
	dataset.convertToTemporalValues(3,2)
	print('Max:',dataset.getAbsMaxes())
	# -------------------
	print('-------------------')
	start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(include_test_data=True,normalization=Dataset.Normalization.NORMALIZE)
	print('Neural Network X 0:',dataset_x[0].tolist())
	print('Neural Network Y 0:',dataset_y[0].tolist())
	# -------------------
	print('-------------------')
	correct_predictions=np.array([[[1, -1], [1.1, -1.1]],[[1.1, -1.1], [1.2, -1.2]]])
	print('Correct preds shape',correct_predictions.shape)
	dataset.setNeuralNetworkResultArray(start_index_to_pred,correct_predictions)
	print('Len:',dataset.getSize())
	dataset.print()
	# -------------------
	print('-------------------')
	dataset=Dataset(name='OriginalGE and OriginalGE*-1')
	dataset.addCompany(stock_value,dates)
	dataset.addCompany(stock_value_2,dates)
	print('Values Main:',dataset.getValues(only_main_value=True))
	print('Values Splitted by Feature:',dataset.getValuesSplittedByFeature())
	# -------------------
	print('-------------------')
	dataset=Dataset(name='OriginalGE')
	dataset.addCompany(stock_value,dates)
	dataset.convertToTemporalValues(4,3)
	dataset.print()
	correct_predictions=np.array([[[9.1], [10.1], [11.1]],[[10.2], [11.2], [12.2]],[[11.3], [12.3], [13.3]]])
	dataset.setNeuralNetworkResultArray(start_index_to_pred-1,correct_predictions)
	dataset.print()
	dataset.revertFromTemporalValues()
	dataset.print()
	indexes,preds=dataset.getDatesAndPredictions()
	print('Pred Indexes:',indexes)
	print('Pred Values:',preds)
	tmp_pred_values=[[[] for _ in range(3)] for _ in range(1)]
	for i,day_samples in enumerate(preds):
		for j,a_prediction in enumerate(day_samples): 
			for k,company in enumerate(a_prediction):
				print(i,j,k,'-',company)
				tmp_pred_values[k][j].append(company)
	preds=tmp_pred_values
	print('Pred Values:',preds)


def network_architecture_search_genetic_test():
	feature_group=2 #0-6
	binary_classifier=False
	use_ok_instead_of_f1=True
	stock='T'



	Genome.CACHE_WEIGHTS=False
	search_maximum=True	
	input_features=[[],
					[Features.OC,Features.OH,Features.OL,Features.CH,Features.CL,Features.LH],
					[Features.OC,Features.OH,Features.OL,Features.CH,Features.CL,Features.LH,Features.FAST_MOVING_AVG,Features.SLOW_MOVING_AVG,Features.LOG_RETURN],
					[Features.FAST_MOVING_AVG,Features.SLOW_MOVING_AVG,Features.LOG_RETURN],
					[Features.OPEN,Features.HIGH,Features.LOW,Features.ADJ_CLOSE,Features.VOLUME],
					[Features.OPEN,Features.HIGH,Features.LOW,Features.ADJ_CLOSE,Features.VOLUME,Features.FAST_MOVING_AVG,Features.SLOW_MOVING_AVG,Features.LOG_RETURN],
					[Features.OPEN,Features.HIGH,Features.LOW,Features.ADJ_CLOSE,Features.VOLUME,Features.OC,Features.OH,Features.OL,Features.CH,Features.CL,Features.LH,Features.FAST_MOVING_AVG,Features.SLOW_MOVING_AVG,Features.LOG_RETURN]]

	never_crawl=os.getenv('NEVER_CRAWL',default='False')
	never_crawl=never_crawl.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')

	search_space=SearchSpace()
	search_space.add(5,60,SearchSpace.Type.INT,'backwards_samples')
	search_space.add(1,14,SearchSpace.Type.INT,'forward_samples')
	search_space.add(1,5,SearchSpace.Type.INT,'lstm_layers')
	search_space.add(500,5000,SearchSpace.Type.INT,'max_epochs')
	search_space.add(100,5000,SearchSpace.Type.INT,'patience_epochs_stop')
	search_space.add(0,1000,SearchSpace.Type.INT,'patience_epochs_reduce')
	search_space.add(0.0,0.2,SearchSpace.Type.FLOAT,'reduce_factor')
	search_space.add(0,128,SearchSpace.Type.INT,'batch_size')
	search_space.add(False,True,SearchSpace.Type.BOOLEAN,'stateful')
	search_space.add(True,True,SearchSpace.Type.BOOLEAN,'normalize')
	search_space.add(Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),SearchSpace.Type.INT,'optimizer')
	search_space.add(NodeType.RELU,NodeType.TANH,SearchSpace.Type.INT,'activation_functions')
	search_space.add(NodeType.RELU,NodeType.HARD_SIGMOID,SearchSpace.Type.INT,'recurrent_activation_functions')
	search_space.add(False,False,SearchSpace.Type.BOOLEAN,'shuffle')
	search_space.add(False,True,SearchSpace.Type.BOOLEAN,'use_dense_on_output')
	search_space.add(10,200,SearchSpace.Type.INT,'layer_sizes')
	search_space.add(0,0.3,SearchSpace.Type.FLOAT,'dropout_values')
	search_space.add(0,0.3,SearchSpace.Type.FLOAT,'recurrent_dropout_values')
	search_space.add(False,True,SearchSpace.Type.BOOLEAN,'bias')
	search_space.add(False,True,SearchSpace.Type.BOOLEAN,'unit_forget_bias')
	search_space.add(False,True,SearchSpace.Type.BOOLEAN,'go_backwards')
	search_space=Genome.enrichSearchSpace(search_space)
	amount_companies=1
	train_percent=.8
	val_percent=.3
	if binary_classifier:
		input_features=[Features.UP]+input_features[feature_group]
		output_feature=Features.UP
		index_feature='Date'
		model_metrics=['accuracy','mean_squared_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+input_features[feature_group]
		output_feature=Features.CLOSE
		index_feature='Date'
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
		loss='mean_squared_error'

	start_date='01/01/2016' #Utils.FIRST_DATE
	end_date='07/05/2021'
	test_date='07/02/2021'
	start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
	end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))
	filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
	crawler=Crawler()
	filepath=crawler.getDatasetPath(filename)
	if not Utils.checkIfPathExists(filepath) and not never_crawl:
		crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
		NeuralNetwork.enrichDataset(filepath)
		
	def train_callback(genome):
		nonlocal stock,filepath,test_date,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,use_ok_instead_of_f1
		preserve_weights=False # TODO not implemented yet!
		hyperparameters=genome.toHyperparameters(input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)
		neuralNetwork=NeuralNetwork(hyperparameters,stock_name=stock,verbose=False)
		neuralNetwork.loadDataset(filepath,plot=False,blocking_plots=False,save_plots=True)
		neuralNetwork.buildModel(plot_model_to_file=True)
		neuralNetwork.train()
		neuralNetwork.restoreCheckpointWeights(delete_after=False)
		neuralNetwork.save()
		neuralNetwork.loadTestDataset(filepath,from_date=test_date,blocking_plots=False,save_plots=True)
		neuralNetwork.eval(plot=True,print_prediction=False,blocking_plots=False,save_plots=True)
		if use_ok_instead_of_f1:
			output=neuralNetwork.metrics['test']['Class Metrics']['OK_Rate']
		else:
			output=neuralNetwork.metrics['test']['Class Metrics']['f1_monark']
		neuralNetwork.destroy()
		return output

	verbose_natural_selection=True
	verbose_population_details=True
	population_start_size_enh=1
	max_gens=1
	max_age=2
	max_children=3
	mutation_rate=0.1
	recycle_rate=0.13
	sex_rate=0.7
	max_notables=5
	enh_elite=HallOfFame(max_notables, search_maximum)
	en_ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
	enh_population=PopulationManager(en_ga,search_space,train_callback,population_start_size_enh,neural_genome=True,print_deltas=verbose_population_details)
	enh_population.hall_of_fame=enh_elite
	enh_population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)
	
	for individual in enh_elite.notables:
		print(str(individual))
	Utils.printDict(enh_elite.best,'Elite')
	print('Evaluating best')

	def test_callback(genome):
		nonlocal stock,filepath,test_date,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,start_date_formated_for_file,end_date_formated_for_file
		hyperparameters=genome.toHyperparameters(input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)
		neuralNetwork=NeuralNetwork(hyperparameters,stock_name=stock,verbose=True)
		neuralNetwork.load()
		neuralNetwork.restoreCheckpointWeights(delete_after=False)
		neuralNetwork.loadTestDataset(filepath,from_date=test_date,blocking_plots=False,save_plots=True)
		neuralNetwork.eval(plot=True,print_prediction=True,blocking_plots=False,save_plots=True)
		neuralNetwork.destroy()
		NeuralNetwork.runPareto(use_ok_instead_of_f1=True,plot=True,blocking_plots=False,save_plots=True,label='{}-{}'.format(start_date_formated_for_file,end_date_formated_for_file))

	test_callback(enh_elite.getBestGenome())


# dataset_test()
network_architecture_search_genetic_test()