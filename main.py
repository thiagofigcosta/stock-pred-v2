#!/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import re
import getopt
import matplotlib
from Enums import *
from matplotlib import pyplot as plt
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils

COMPUTE_STOCK_MAGNITUDE=False

def getPredefHyperparams():
	hyperparameters=[]

	# Forecast
	name='Subject 1'
	feature_group=0
	binary_classifier=False
	train_percent=.8
	val_percent=.3
	amount_companies=1
	index_feature='Date'
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.UP
		model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.CLOSE
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		loss='mean_squared_error'
	# fixed
	shuffle=False
	# irace
	backwards_samples=16
	forward_samples=11
	lstm_layers=1
	max_epochs=4889
	patience_epochs_stop=1525
	patience_epochs_reduce=539
	reduce_factor=0.01259
	batch_size=122
	stateful=False
	optimizer=Optimizers.RMSPROP # Optimizers.ADAM
	use_dense_on_output=False
	activation_functions=[NodeType.SIGMOID]
	recurrent_activation_functions=[NodeType.TANH]
	layer_sizes=[89]
	dropout_values=[0.03741]
	recurrent_dropout_values=[0.24559]
	bias=[False]
	unit_forget_bias=[True]
	go_backwards=[True]
	hyperparameters.append(Hyperparameters(name=name,binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))


	name='Subject 2'
	feature_group=0
	binary_classifier=False
	train_percent=.8
	val_percent=.3
	amount_companies=1
	index_feature='Date'
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.UP
		model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.CLOSE
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		loss='mean_squared_error'
	# fixed
	shuffle=False
	# irace
	backwards_samples=31
	forward_samples=7
	lstm_layers=1
	max_epochs=3183
	patience_epochs_stop=1827
	patience_epochs_reduce=228
	reduce_factor=0.05265
	batch_size=46
	stateful=False
	optimizer=Optimizers.ADAM # Optimizers.RMSPROP
	use_dense_on_output=False
	activation_functions=[NodeType.SIGMOID]
	recurrent_activation_functions=[NodeType.TANH]
	layer_sizes=[67]
	dropout_values=[0.24864]
	recurrent_dropout_values=[0.15223]
	bias=[True]
	unit_forget_bias=[True]
	go_backwards=[False]
	hyperparameters.append(Hyperparameters(name=name,binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))
	
	name='Subject 3'
	feature_group=0
	binary_classifier=False
	train_percent=.8
	val_percent=.3
	amount_companies=1
	index_feature='Date'
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.UP
		model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.CLOSE
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		loss='mean_squared_error'
	# fixed
	shuffle=False
	# irace
	backwards_samples=38
	forward_samples=10
	lstm_layers=1
	max_epochs=3002
	patience_epochs_stop=853
	patience_epochs_reduce=73
	reduce_factor=0.06082
	batch_size=112
	stateful=False
	optimizer=Optimizers.ADAM # Optimizers.RMSPROP
	use_dense_on_output=False
	activation_functions=[NodeType.TANH]
	recurrent_activation_functions=[NodeType.RELU]
	layer_sizes=[164]
	dropout_values=[0.23404]
	recurrent_dropout_values=[0.00155]
	bias=[True]
	unit_forget_bias=[True]
	go_backwards=[False]
	hyperparameters.append(Hyperparameters(name=name,binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))


	# Binary
	name='Subject 1'
	feature_group=0
	binary_classifier=True
	train_percent=.8
	val_percent=.3
	amount_companies=1
	index_feature='Date'
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.UP
		model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.CLOSE
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		loss='mean_squared_error'
	# fixed
	shuffle=False
	# irace
	backwards_samples=37
	forward_samples=7
	lstm_layers=1
	max_epochs=4346
	patience_epochs_stop=3574
	patience_epochs_reduce=606
	reduce_factor=0.09476
	batch_size=10
	stateful=True
	optimizer=Optimizers.ADAM # Optimizers.RMSPROP
	use_dense_on_output=True
	activation_functions=[NodeType.SIGMOID]
	recurrent_activation_functions=[NodeType.TANH]
	layer_sizes=[125]
	dropout_values=[0.19655]
	recurrent_dropout_values=[0.29698]
	bias=[False]
	unit_forget_bias=[False]
	go_backwards=[False]
	hyperparameters.append(Hyperparameters(name=name,binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))

	name='Subject 2'
	feature_group=0
	binary_classifier=True
	train_percent=.8
	val_percent=.3
	amount_companies=1
	index_feature='Date'
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.UP
		model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.CLOSE
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		loss='mean_squared_error'
	# fixed
	shuffle=False
	# irace
	backwards_samples=5
	forward_samples=7
	lstm_layers=2
	max_epochs=2800
	patience_epochs_stop=692
	patience_epochs_reduce=634
	reduce_factor=0.12503
	batch_size=102
	stateful=True
	optimizer=Optimizers.ADAM # Optimizers.RMSPROP
	use_dense_on_output=True
	activation_functions=[NodeType.LINEAR,NodeType.LINEAR]
	recurrent_activation_functions=[NodeType.LINEAR,NodeType.HARD_SIGMOID]
	layer_sizes=[95, 77]
	dropout_values=[0.16891, 0.01028]
	recurrent_dropout_values=[0.24156, 0.03735]
	bias=[True,False]
	unit_forget_bias=[False,True]
	go_backwards=[True,False]
	hyperparameters.append(Hyperparameters(name=name,binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))


	name='Subject 3'
	feature_group=0
	binary_classifier=True
	train_percent=.8
	val_percent=.3
	amount_companies=1
	index_feature='Date'
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.UP
		model_metrics=['accuracy','mean_squared_error','mean_absolute_percentage_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
		output_feature=Features.CLOSE
		model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity','mean_absolute_percentage_error']
		loss='mean_squared_error'
	# fixed
	shuffle=False
	# irace
	backwards_samples=38
	forward_samples=7
	lstm_layers=4
	max_epochs=4116
	patience_epochs_stop=2418
	patience_epochs_reduce=869
	reduce_factor=0.09956
	batch_size=58
	stateful=True
	optimizer=Optimizers.RMSPROP # Optimizers.ADAM
	use_dense_on_output=True
	activation_functions=[NodeType.RELU,NodeType.EXPONENTIAL,NodeType.SIGMOID,NodeType.RELU]
	recurrent_activation_functions=[NodeType.SIGMOID,NodeType.TANH,NodeType.LINEAR,NodeType.SIGMOID]
	layer_sizes=[48, 30, 194, 72]
	dropout_values=[0.06482, 0.18694, 0.19006, 0.27753]
	recurrent_dropout_values=[0.13098, 0.25548, 0.24709, 0.14463]
	bias=[False,True,False,False]
	unit_forget_bias=[False,False,False,True]
	go_backwards=[False,False,False,False]
	hyperparameters.append(Hyperparameters(name=name,binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))

	return hyperparameters


def run_available_networks():
	models={}
	for file_str in os.listdir(NeuralNetwork.MODELS_PATH):
		re_result=re.search(r'([a-z0-9]*_[a-zA-Z0-9\-%\.=]*)_.*\.(h5|json|bin)', file_str)
		if re_result:
			model_id=re_result.group(1)
			if model_id not in models:
				models[model_id]=[file_str]
			else:
				models[model_id].append(file_str)
	for i,(base,files) in enumerate(models.items()):
		checkpoint_filename=None
		model_filename=None
		metrics_filename=None
		last_patience_filename=None
		scaler_filename=None
		history_filename=None
		hyperparam_filename=None
		hyper_hash= base.split('_')[0]
		stock= base.split('_')[1]
		for file in files:
			if re.search(r'.*_scaler\.bin', file):
				scaler_filename=file
			elif re.search(r'.*_hyperparams\.json', file):
				hyperparam_filename=file
			elif re.search(r'.*_history\.json', file):
				history_filename=file
			elif re.search(r'.*_cp\.h5', file):
				checkpoint_filename=file
			elif re.search(r'.*(?<![_cp|_last_patience])\.h5', file):
				model_filename=file
			elif re.search(r'.*(?<!_last_patience)_metrics\.json', file):
				metrics_filename=file
			elif re.search(r'.*_last_patience\.h5', file):
				last_patience_filename=file

		if hyperparam_filename is not None:
			train = False
			if model_filename is None:
				if checkpoint_filename is not None:
					model_filename = base+'.h5'
					Utils.copyFile(Utils.joinPath(NeuralNetwork.MODELS_PATH,checkpoint_filename), Utils.joinPath(NeuralNetwork.MODELS_PATH,model_filename))
				elif last_patience_filename is not None:
					model_filename = base+'.h5'
					Utils.copyFile(Utils.joinPath(NeuralNetwork.MODELS_PATH,last_patience_filename), Utils.joinPath(NeuralNetwork.MODELS_PATH,model_filename))
				else:
					train = True

			hyperparameter=Hyperparameters.loadJson(Utils.joinPath(NeuralNetwork.MODELS_PATH,hyperparam_filename))
			hyperparameter.uuid = hyper_hash # To ensure that the uuid will remain the same
			if 'mean_absolute_percentage_error' not in hyperparameter.model_metrics:
				hyperparameter.model_metrics.append('mean_absolute_percentage_error') # TODO temporary
			eval_model=True
			plot=True
			plot_eval=True
			plot_dataset=True
			blocking_plots=False
			save_plots=True
			restore_checkpoints=False
			download_if_needed=True
			stocks=[stock]
			start_date='01/01/2016' #Utils.FIRST_DATE
			end_date='07/05/2021'
			test_date='07/01/2021'
			enrich_dataset=True
			analyze_metrics=True if i+1 == len(models) else False
			move_models=False
			all_hyper_for_all_stocks=None
			only_first_hyperparam=None
			add_more_fields_to_hyper=None
			hyperparams_per_stock={
				stock: [hyperparameter]
			}

			print(hyperparameter.uuid)
			print('Found model {}'.format(hyper_hash))
			print('\tstock: {}'.format(stock))
			print('\thyperparam_filename: {}'.format(hyperparam_filename))
			print('\tcheckpoint_filename: {}'.format(checkpoint_filename))
			print('\tmodel_filename: {}'.format(model_filename))
			print('\tmetrics_filename: {}'.format(metrics_filename))
			print('\tlast_patience_filename: {}'.format(last_patience_filename))
			print('\tscaler_filename: {}'.format(scaler_filename))
			print('\thistory_filename: {}'.format(history_filename))
			print()

			run(train,train,eval_model,plot,plot_eval,plot_dataset,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks,only_first_hyperparam,add_more_fields_to_hyper,test_date,hyperparams_per_stock=hyperparams_per_stock)


def run(train_model,force_train,eval_model,plot,plot_eval,plot_dataset,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks,only_first_hyperparam,add_more_fields_to_hyper,test_date,hyperparams_per_stock=None):
	never_crawl=os.getenv('NEVER_CRAWL',default='False')
	never_crawl=never_crawl.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
	
	crawler=Crawler()

	if save_plots:
		matplotlib.use('Agg')
		print('Using plot id: ',NeuralNetwork.SAVED_PLOTS_ID)
		plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

	print('Running for stocks: {}'.format(','.join(stocks)))

	if start_date is None:
		start_date=Utils.FIRST_DATE
	else:
		Utils.assertDateFormat(start_date)

	if end_date is None:
		end_date='07/05/2021'
	else:
		Utils.assertDateFormat(end_date)

	if test_date is None:
		test_date='10/03/2021'
	else:
		Utils.assertDateFormat(test_date)

	start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
	end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))

	filepaths={}
	for stock in stocks:
		filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
		filepath=crawler.getDatasetPath(filename)
		filepaths[stock]=filepath
		if not Utils.checkIfPathExists(filepath) and download_if_needed and not never_crawl:
			crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
			# crawler.downloadStockDataCustomInterval(stock,filename,data_range='max') # just example
	
	if hyperparams_per_stock is None:
		hyperparameters_tmp=getPredefHyperparams()
		if only_first_hyperparam:
			hyperparameters_tmp=[hyperparameters_tmp[0]]
		if not all_hyper_for_all_stocks: # then create a circular"ish" list, only works when running all stocks together, otherwise it will always use the first
			if len(stocks) > len(hyperparameters_tmp):
				for i in range(len(stocks)-len(hyperparameters_tmp)):
					hyperparameters_tmp.append(hyperparameters_tmp[i%len(hyperparameters_tmp)].copy())
		
		hyperparameters={}
		for i,stock in enumerate(stocks):
			new_input_fields=('fast_moving_avg','slow_moving_avg','Volume','Open','High','Low','Adj Close')
			if all_hyper_for_all_stocks:
				hyperparameters[stock]=[]
				for hyper in hyperparameters_tmp:
					if hyper.name != '':
						hyper.setName('{} - from: {} to: {}'.format(hyper.name,start_date,end_date))
					else:
						hyper.setName('manual tunning - from: {} to: {}'.format(start_date,end_date))
					hyperparameters[stock].append(hyper.copy())
					if add_more_fields_to_hyper:
						for new_input_field in new_input_fields:
							new_hyperparameters=hyper.copy()
							new_hyperparameters.input_features.append(new_input_field)
							new_hyperparameters.genAndSetUuid()
							hyperparameters[stock].append(new_hyperparameters)
			else:
				if hyperparameters_tmp[i].name != '':
					hyperparameters_tmp[i].setName('{} - from: {} to: {}'.format(hyperparameters_tmp[i].name,start_date,end_date))
				else:
					hyperparameters_tmp[i].setName('manual tunning - from: {} to: {}'.format(start_date,end_date))
				hyperparameters[stock]=[hyperparameters_tmp[i]]
				if add_more_fields_to_hyper:
					for new_input_field in new_input_fields:
						new_hyperparameters=hyperparameters[stock][-1].copy()
						new_hyperparameters.input_features.append(new_input_field)
						new_hyperparameters.genAndSetUuid()
						hyperparameters[stock].append(new_hyperparameters)
		hyperparameters_tmp=[]
	else:
		hyperparameters=hyperparams_per_stock

	if enrich_dataset:
		for stock in stocks:
			NeuralNetwork.enrichDataset(filepaths[stock])

	if COMPUTE_STOCK_MAGNITUDE:
		print('Magninute calc train')
		for stock in stocks:
			for hyperparameter in hyperparameters[stock]:
				try:
					neuralNetwork=NeuralNetwork(hyperparameter,stock_name=stock,verbose=True)
					neuralNetwork.loadDataset(filepaths[stock],plot=plot_dataset,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.datasetMagnitudeCalc()
					neuralNetwork.destroy()
				except:
					pass
	
	if train_model or force_train:
		for stock in stocks:
			# build and train
			for hyperparameter in hyperparameters[stock]:
				neuralNetwork=NeuralNetwork(hyperparameter,stock_name=stock,verbose=True)
				if not neuralNetwork.checkTrainedModelExists() or force_train:
					neuralNetwork.loadDataset(filepaths[stock],plot=plot_dataset,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.buildModel(plot_model_to_file=plot)
					neuralNetwork.train()
					neuralNetwork.eval(plot=plot,plot_training=plot,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.save()
					neuralNetwork.destroy()
	
	if restore_checkpoints:
		NeuralNetwork.restoreAllBestModelsCPs() # restore the best models

	if COMPUTE_STOCK_MAGNITUDE:
		print('Magninute calc test')
		for stock in stocks:
			for hyperparameter in hyperparameters[stock]:
				try:
					neuralNetwork=NeuralNetwork(hyperparameter,stock_name=stock,verbose=True)
					neuralNetwork.loadTestDataset(filepaths[stock],from_date=test_date,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.datasetMagnitudeCalc()
					neuralNetwork.destroy()
				except:
					pass

	if eval_model:
		for stock in stocks:
			# load
			for hyperparameter in hyperparameters[stock]:
				neuralNetwork=NeuralNetwork(hyperparameter,stock_name=stock,verbose=True)
				neuralNetwork.load()
				neuralNetwork.loadTestDataset(filepaths[stock],from_date=test_date,blocking_plots=blocking_plots,save_plots=save_plots)
				neuralNetwork.eval(plot=(plot or plot_eval),print_prediction=True,blocking_plots=blocking_plots,save_plots=save_plots)
				neuralNetwork.destroy()

	if analyze_metrics:
		NeuralNetwork.runPareto(use_ok_instead_of_f1=True,plot=plot,blocking_plots=blocking_plots,save_plots=save_plots,label='{}-{}'.format(start_date_formated_for_file,end_date_formated_for_file))

	if move_models:
		print('Backing up models...',end='')
		paths_to_backup=Utils.getFolderPathsThatMatchesPattern(NeuralNetwork.MODELS_PATH,r'[a-zA-Z0-9]*_.*\.(h5|json|bin)')
		for path_to_backup in paths_to_backup:
			Utils.moveFile(path_to_backup,Utils.joinPath(NeuralNetwork.BACKUP_MODELS_PATH, Utils.filenameFromPath(path_to_backup,get_extension=True)))
		print('OK!')

	if not blocking_plots or save_plots:
		plt.clf()
		plt.cla()
		plt.close() # delete the last and empty figure
	if not blocking_plots and not save_plots:
		plt.show()

	
def main(argv):

	all_known_stocks=[
						'GOOG','AMD','CSCO','TSLA','AAPL', # international companies
						'T','IBM', # dividend aristocrats
						'BTC-USD',	# crypto currencies
						'BRL=X', # currency exchange rate 
						r'%5EDJI',r'%5EBVSP', # stock market indexes
						'CESP3.SA','CPLE6.SA','CSMG3.SA','ENBR3.SA','TRPL4.SA' # brazilian stable companies
					]

	python_exec_name=Utils.getPythonExecName()
	help_str='main.py\n\t[-h | --help]\n\t[-t | --train]\n\t[--force-train]\n\t[-e | --eval]\n\t[-p | --plot]\n\t[--plot-eval]\n\t[--plot-dataset]\n\t[--blocking-plots]\n\t[--save-plots]\n\t[--force-no-plots]\n\t[--do-not-restore-checkpoints]\n\t[--do-not-download]\n\t[--stock <stock-name>]\n\t\t*default: all\n\t[--start-date <dd/MM/yyyy>]\n\t[--end-date <dd/MM/yyyy>]\n\t[--test-date <dd/MM/yyyy>]\n\t[--enrich-dataset]\n\t[--clear-plots-models-and-datasets]\n\t[--analyze-metrics]\n\t[--move-models-to-backup]\n\t[--restore-backups]\n\t[--dummy]\n\t[--run-all-stocks-together]\n\t[--use-all-hyper-on-all-stocks] *warning: heavy\n\t[--only-first-hyperparam]\n\t[--do-not-test-hyperparams-with-more-fields]'
	help_str+='\n\n\t\t Example for testing datasets: '
	help_str+=r"""
{python} main.py --dummy --clear-plots-models-and-datasets \
echo -e "2018\n\n" >> log.txt; \
{python} main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2018 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt; \
echo -e "\n\n\n\n2015\n\n" >> log.txt; \
{python} main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2015 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt; \
echo -e "\n\n\n\nALL\n\n" >> log.txt; \
{python} main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt \
{python} main.py --dummy --restore-backups >> log.txt; \
echo -e "\n\n\nDONE\n" >> log.txt
	""".format(python=python_exec_name) # FAST RUN: --force-train -e -p --plot-eval --enrich-dataset --start-date 01/01/2018 --stock GOOG --clear-plots-models-and-datasets --analyze-metrics --only-first-hyperparam --do-not-test-hyperparams-with-more-fields
	used_args=[]
	# args vars
	train_model=False
	force_train=False
	eval_model=False
	plot=False
	plot_eval=False
	plot_dataset=False
	blocking_plots=False
	save_plots=False
	force_no_plots=False
	restore_checkpoints=True
	download_if_needed=True
	start_date=None
	end_date=None
	enrich_dataset=False
	analyze_metrics=False
	move_models=False
	dummy=False
	run_stocks_together=False
	all_hyper_for_all_stocks=False
	only_first_hyperparam=False
	add_more_fields_to_hyper=True
	test_date=None
	run_available_hyperparams=False
	stocks=[]
	try:
		opts, _ = getopt.getopt(argv,'htep',['help','run-available-hyperparams','train','force-train','eval','plot','plot-eval','plot-dataset','blocking-plots','save-plots','force-no-plots','do-not-restore-checkpoints','do-not-download','stock=','start-date=','end-date=','test-date=','enrich-dataset','clear-plots-models-and-datasets','analyze-metrics','move-models-to-backup','restore-backups','dummy','run-all-stocks-together','use-all-hyper-on-all-stocks','only-first-hyperparam','do-not-test-hyperparams-with-more-fields'])
	except getopt.GetoptError:
		print ('ERROR PARSING ARGUMENTS, try to use the following:\n\n')
		print (help_str)
		sys.exit(2)
	for opt, arg in opts:
		opt=Utils.removeStrPrefix(Utils.removeStrPrefix(opt,'--'),'-')
		used_args.append(opt)
		if opt in ('h','help'):
			print (help_str)
			sys.exit()
		elif opt in ('t','train'):
			train_model=True
		elif opt == 'force-train':
			force_train=True
		elif opt in ('e','eval'):
			eval_model=True
		elif opt in ('p','plot'):
			plot=True
		elif opt == 'run-available-hyperparams':
			run_available_hyperparams=True
		elif opt == 'use-all-hyper-on-all-stocks':
			all_hyper_for_all_stocks=True
		elif opt == 'run-all-stocks-together':
			run_stocks_together=True
		elif opt == 'plot-eval':
			plot_eval=True
		elif opt == 'plot-dataset':
			plot_dataset=True
		elif opt == 'blocking-plots':
			blocking_plots=True
		elif opt == 'save-plots':
			save_plots=True
		elif opt == 'force-no-plots':
			force_no_plots=True
		elif opt == 'do-not-test-hyperparams-with-more-fields':
			add_more_fields_to_hyper=False
		elif opt == 'do-not-restore-checkpoints':
			restore_checkpoints=False
		elif opt == 'do-not-download':
			download_if_needed=False
		elif opt == 'stock':
			stocks.append(arg.strip())
		elif opt == 'start-date':
			start_date=arg.strip()
		elif opt == 'end-date':
			end_date=arg.strip()
		elif opt == 'test-date':
			test_date=arg.strip()
		elif opt == 'enrich-dataset':
			enrich_dataset=True
		elif opt == 'only-first-hyperparam':
			only_first_hyperparam=True
		elif opt == 'clear-plots-models-and-datasets':
			Utils.deleteFile('log.txt')
			print('Clearing contents of: {}'.format(NeuralNetwork.MODELS_PATH))
			Utils.deleteFolderContents(NeuralNetwork.MODELS_PATH)
			print('Clearing contents of: {}'.format(NeuralNetwork.SAVED_PLOTS_PATH))
			Utils.deleteFolderContents(NeuralNetwork.SAVED_PLOTS_PATH)
			print('Clearing contents of: {}'.format(Crawler.DATASET_PATH))
			Utils.deleteFolderContents(Crawler.DATASET_PATH,['shampoo_example_dataset.csv'])
		elif opt == 'analyze-metrics':
			analyze_metrics=True
		elif opt == 'move-models-to-backup':
			move_models=True
		elif opt == 'restore-backups':
			print('Restoring backups...',end='')
			paths_to_restore=Utils.getFolderPathsThatMatchesPattern(NeuralNetwork.BACKUP_MODELS_PATH,r'[a-zA-Z0-9]*_.*\.(h5|json|bin)')
			for path_to_restore in paths_to_restore:
				Utils.moveFile(path_to_restore,Utils.joinPath(NeuralNetwork.MODELS_PATH, Utils.filenameFromPath(path_to_restore,get_extension=True)))
			print('OK!')
		elif opt == 'dummy':
			dummy=True

	if dummy:
		if analyze_metrics:
			run(False,False,False,True,False,False,False,True,False,False,[],None,None,False,analyze_metrics,False,None,None,None,None,hyperparams_per_stock={})
		sys.exit(0)

	if run_available_hyperparams:
		run_available_networks()
		sys.exit(0)

	if len(stocks)==0:	
		for stock in all_known_stocks:
			stocks.append(stock)

	functional_args=('analyze-metrics','train','force-train','eval')
	if 'analyze-metrics' in used_args and not any(i in used_args for i in functional_args[1:]):
		print('Running only analyze metrics')
		run_stocks_together=True

	if len(opts) == 0 or not any(i in used_args for i in functional_args):
		train_model=True
		force_train=False
		eval_model=True
		analyze_metrics=False
		move_models=False

		if 'plot' not in used_args:
			plot=True
		if 'plot-eval' not in used_args:
			plot_eval=False
		if 'plot-dataset' not in used_args:
			plot_dataset=False
		if 'blocking-plots' not in used_args:
			blocking_plots=False
		if 'do-not-restore-checkpoints' not in used_args:
			restore_checkpoints=True
		if 'do-not-download' not in used_args:
			download_if_needed=True
		if 'enrich-dataset' not in used_args:
			enrich_dataset=True
		if 'save-plots' not in used_args:
			save_plots=False
		print('No functional arguments were found, using defaults:')
		print('\tcmd: python3 main.py --train --eval --plot')
		print('\ttrain_model:',train_model)
		print('\tforce_train:',force_train)
		print('\teval_model:',eval_model)
		print('\tplot:',plot)
		print('\tplot_eval:',plot_eval)
		print('\tplot_dataset:',plot_dataset)
		print('\tblocking_plots:',blocking_plots)
		print('\tforce_no_plots:',force_no_plots)
		print('\trestore_checkpoints:',restore_checkpoints)
		print('\tdownload_if_needed:',download_if_needed)
		print('\tenrich_dataset:',enrich_dataset)
		print('\tsave_plots:',save_plots)
		print('\tstocks:',stocks)
		print('\tstart_date:',start_date)
		print('\tend_date:',end_date)
		print('\ttest_date:',test_date)
		print('\tanalyze_metrics:',analyze_metrics)
		print('\tmove_models:',move_models)
		print('\trun_stocks_together:',run_stocks_together)
		print('\tall_hyper_for_all_stocks:',all_hyper_for_all_stocks)
		print('\tonly_first_hyperparam:',only_first_hyperparam)
		print('\tadd_more_fields_to_hyper:',add_more_fields_to_hyper)

	if run_stocks_together:
		stocks=[stocks]

	for stock in stocks:
		if type(stock) is not list:
			stocks_to_run=[stock]
		else:
			stocks_to_run=stock
		run(train_model,force_train,eval_model,plot and not force_no_plots,plot_eval and not force_no_plots,plot_dataset and not force_no_plots,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks_to_run,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks,only_first_hyperparam,add_more_fields_to_hyper,test_date)

if __name__ == '__main__':
	delta=-time.time()
	main(sys.argv[1:])
	delta+=time.time()
	print('\n\nTotal run time is {}'.format(Utils.timestampByExtensive(delta)))
