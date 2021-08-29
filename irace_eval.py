#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import getopt
import matplotlib
from matplotlib import pyplot as plt
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters



import os
import sys
import argparse
from Utils import Utils
from Enums import NodeType,Loss,Metric,Optimizers,Features

def run(train_model,force_train,eval_model,plot,plot_eval,plot_dataset,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks,only_first_hyperparam,add_more_fields_to_hyper,test_date):
	

	if save_plots:
		matplotlib.use('Agg')
		print('Using plot id: ',NeuralNetwork.SAVED_PLOTS_ID)

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
				hyper.setName('manual tunning - from: {} to: {}'.format(start_date,end_date))
				hyperparameters[stock].append(hyper.copy())
				if add_more_fields_to_hyper:
					for new_input_field in new_input_fields:
						new_hyperparameters=hyper.copy()
						new_hyperparameters.input_features.append(new_input_field)
						new_hyperparameters.genAndSetUuid()
						hyperparameters[stock].append(new_hyperparameters)
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

	if enrich_dataset:
		for stock in stocks:
			NeuralNetwork.enrichDataset(filepaths[stock])
	
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
		plt.close() # delete the last and empty figure
	if not blocking_plots and not save_plots:
		plt.show()

	
def main(stock_name,start_date,end_date,test_date,use_ok_instead_of_f1,binary_classifier,input_features,output_feature,index_feature,model_metrics,loss,normalize,backwards_samples,forward_samples,max_epochs,stateful,batch_size,use_dense_on_output,patience_epochs_stop,patience_epochs_reduce,reduce_factor,optimizer,shuffle,lstm_layers,layer_sizes,activation_funcs,recurrent_activation_funcs,dropouts,recurrent_dropouts,bias,unit_forget_bias,go_backwards,datfile,confid=None):
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
	if confid is None:
		hyper_id=Utils.randomUUID()
	else:
		hyper_id=confid

	backwards_samples=int(backwards_samples)
	forward_samples=int(forward_samples)
	max_epochs=int(max_epochs)
	stateful=stateful.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
	batch_size=int(batch_size)
	use_dense_on_output=use_dense_on_output.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')

	patience_epochs_stop=int(patience_epochs_stop)
	patience_epochs_reduce=int(patience_epochs_reduce)
	reduce_factor=float(reduce_factor)
	normalize=normalize.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
	optimizer=Optimizers(optimizer).toKerasName()
	shuffle=shuffle.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')

	lstm_layers=int(lstm_layers)
	for l in range(lstm_layers):
		layer_sizes[l]=int(layer_sizes[l])
		activation_funcs[l]=NodeType(activation_funcs[l]).toKerasName()
		recurrent_activation_funcs[l]=NodeType(recurrent_activation_funcs[l]).toKerasName()
		dropouts[l]=float(dropouts[l])
		recurrent_dropouts[l]=float(recurrent_dropouts[l])
		bias[l]=bias[l].lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
		unit_forget_bias[l]=unit_forget_bias[l].lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
		go_backwards[l]=go_backwards[l].lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')


	hyperparameters=Hyperparameters(name=hyper_id,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropouts,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_funcs,recurrent_activation_functions=recurrent_activation_funcs,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropouts,binary_classifier=binary_classifier)

	never_crawl=os.getenv('NEVER_CRAWL',default='False')
	never_crawl=never_crawl.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
	
	start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
	end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))
	filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
	crawler=Crawler()
	filepath=crawler.getDatasetPath(filename)
	if not Utils.checkIfPathExists(filepath) and not never_crawl:
		crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
		NeuralNetwork.enrichDataset(filepath)


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
		output=neuralNetwork.metrics['test']['Class Metrics']['f1_monark']*100
	neuralNetwork.destroy()
	
	with open(datfile, 'w') as f:
		f.write(str(output))

	
if __name__ == '__main__':
	with open('args.txt', 'w') as f:
		f.write(str(sys.argv))
	stock_name='T'
	binary_classifier=False
	use_ok_instead_of_f1=True
	start_date='01/01/2016' #Utils.FIRST_DATE
	end_date='07/05/2021'
	test_date='07/02/2021'
	amount_companies=1
	train_percent=.8
	val_percent=.3
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]
		output_feature=Features.UP
		index_feature='Date'
		model_metrics=['accuracy','mean_squared_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]
		output_feature=Features.CLOSE
		index_feature='Date'
		model_metrics=['mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
		loss='mean_squared_error'

	ap = argparse.ArgumentParser()
	
	ap.add_argument('--bs', dest='bs', type=int, required=True, help='backwards_samples')
	ap.add_argument('--fs', dest='fs', type=int, required=True, help='forward_samples')
	ap.add_argument('--me', dest='me', type=int, required=True, help='max_epochs')
	ap.add_argument('--st', dest='st', type=str, required=True, help='stateful')
	ap.add_argument('--bs', dest='bs', type=int, required=True, help='batch_size')
	ap.add_argument('--do', dest='do', type=str, required=True, help='use_dense_on_output')
	ap.add_argument('--pes', dest='pes', type=int, required=True, help='patience_epochs_stop')
	ap.add_argument('--per', dest='per', type=int, required=True, help='patience_epochs_reduce')
	ap.add_argument('--rf', dest='rf', type=float, required=True, help='reduce_factor')
	ap.add_argument('--op', dest='op', type=int, required=True, help='optimizer')
	ap.add_argument('--sh', dest='sh', type=str, required=True, help='shuffle')
	ap.add_argument('--lrs', dest='lrs', type=int, required=True, help='lstm_layers')
	ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
	ap.add_argument('--config-id', dest='confid', type=str, required=True, help='config_id')

	args=parser.parse_known_args()

	lstm_layers=args.lrs

	layer_sizes=[]
	activation_funcs=[]
	recurrent_activation_funcs=[]
	dropouts=[]
	recurrent_dropouts=[]
	bias=[]
	unit_forget_bias=[]
	go_backwards=[]
	for l in range(lstm_layers):
		ap.add_argument('--ls-{}'.format(l), dest='ls-{}'.format(l), type=int, required=True, help='layer_sizes[{}]'.format(l))
		ap.add_argument('--af-{}'.format(l), dest='af-{}'.format(l), type=int, required=True, help='activation_funcs[{}]'.format(l))
		ap.add_argument('--raf-{}'.format(l), dest='raf-{}'.format(l), type=int, required=True, help='recurrent_activation_funcs[{}]'.format(l))
		ap.add_argument('--dr-{}'.format(l), dest='dr-{}'.format(l), type=float, required=True, help='dropouts[{}]'.format(l))
		ap.add_argument('--rdr-{}'.format(l), dest='rdr-{}'.format(l), type=float, required=True, help='recurrent_dropouts[{}]'.format(l))
		ap.add_argument('--bi-{}'.format(l), dest='bi-{}'.format(l), type=str, required=True, help='bias[{}]'.format(l))
		ap.add_argument('--ufb-{}'.format(l), dest='ufb-{}'.format(l), type=str, required=True, help='unit_forget_bias[{}]'.format(l))
		ap.add_argument('--gb-{}'.format(l), dest='gb-{}'.format(l), type=str, required=True, help='go_backwards[{}]'.format(l))

	args=ap.parse_args()
	args_dict=vars(args)

	for l in range(lstm_layers):
		layer_sizes.append(args_dict['ls-{}'.format(l)])
		activation_funcs.append(args_dict['af-{}'.format(l)])
		recurrent_activation_funcs.append(args_dict['raf-{}'.format(l)])
		dropouts.append(args_dict['dr-{}'.format(l)])
		recurrent_dropouts.append(args_dict['rdr-{}'.format(l)])
		bias.append(args_dict['bi-{}'.format(l)])
		unit_forget_bias.append(args_dict['ufb-{}'.format(l)])
		go_backwards.append(args_dict['gb-{}'.format(l)])

	main(stock_name,start_date,end_date,test_date,use_ok_instead_of_f1,binary_classifier,input_features,output_feature,index_feature,model_metrics,loss,normalize,args.bs,args.fs,args.me,args.st,args.bs,args.do,args.pes,args.per,args.rf,args.op,args.sh,args.lrs,layer_sizes,activation_funcs,recurrent_activation_funcs,dropouts,recurrent_dropouts,bias,unit_forget_bias,go_backwards,args.datfile,args.confid)
	
