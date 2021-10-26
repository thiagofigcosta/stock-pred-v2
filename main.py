#!/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import getopt
import matplotlib
from Enums import *
from matplotlib import pyplot as plt
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils

def getPredefHyperparams():
	MAX_EPOCHS=100
	hyperparameters=[]

	ADD_BINARY_CROSSEN_HYPER=False
	if ADD_BINARY_CROSSEN_HYPER:
		binary_classifier=True
		input_features=['up']
		output_feature='up'
		index_feature='Date'
		backwards_samples=21
		forward_samples=7
		lstm_layers=2
		max_epochs=MAX_EPOCHS
		patience_epochs_stop=0
		batch_size=8
		stateful=False
		dropout_values=[0.5,0.5]
		layer_sizes=[50,50]
		normalize=False
		optimizer='rmsprop'
		model_metrics=['accuracy','mean_squared_error']
		loss='categorical_crossentropy'
		train_percent=.8
		val_percent=.2
		amount_companies=1
		shuffle=False
		activation_functions='sigmoid'
		recurrent_activation_functions='hard_sigmoid'
		bias=[False,False]
		use_dense_on_output=True
		unit_forget_bias=True
		go_backwards=False
		recurrent_dropout_values=[0.01,0.01]
		hyperparameters.append(Hyperparameters(binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))

	binary_classifier=False
	feature_group=0
	input_features=[Features.CLOSE]+Hyperparameters.getFeatureGroups()[feature_group]
	output_feature=Features.CLOSE
	index_feature='Date'
	normalize=True
	train_percent=.8
	val_percent=.3
	amount_companies=1
	model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
	loss='mean_squared_error'
	# irace
	backwards_samples=39
	forward_samples=11
	max_epochs=2795
	stateful=False
	batch_size=106
	use_dense_on_output=True
	patience_epochs_stop=1304
	patience_epochs_reduce=945
	reduce_factor=0.02062
	optimizer=Optimizers.RMSPROP # Optimizers.ADAM
	shuffle=False
	lstm_layers=2
	layer_sizes=[196,165]
	activation_functions=[NodeType.RELU,NodeType.SIGMOID]
	recurrent_activation_functions=[NodeType.SIGMOID,NodeType.RELU]
	dropout_values=[0.09278,0.05087]
	recurrent_dropout_values=[0.11975,0.10689]
	bias=[True,False]
	unit_forget_bias=[True,True]
	go_backwards=[True,False]
	hyperparameters.append(Hyperparameters(binary_classifier=binary_classifier,input_features=input_features,output_feature=output_feature,index_feature=index_feature,backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,patience_epochs_stop=patience_epochs_stop,patience_epochs_reduce=patience_epochs_reduce,reduce_factor=reduce_factor,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,optimizer=optimizer,model_metrics=model_metrics,loss=loss,train_percent=train_percent,val_percent=val_percent,amount_companies=amount_companies,shuffle=shuffle,activation_functions=activation_functions,recurrent_activation_functions=recurrent_activation_functions,bias=bias,use_dense_on_output=use_dense_on_output,unit_forget_bias=unit_forget_bias,go_backwards=go_backwards,recurrent_dropout_values=recurrent_dropout_values))
	
	return hyperparameters



def run(train_model,force_train,eval_model,plot,plot_eval,plot_dataset,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks,only_first_hyperparam,add_more_fields_to_hyper,test_date):
	never_crawl=os.getenv('NEVER_CRAWL',default='False')
	never_crawl=never_crawl.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')
	
	crawler=Crawler()

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
	stocks=[]
	try:
		opts, _ = getopt.getopt(argv,'htep',['help','train','force-train','eval','plot','plot-eval','plot-dataset','blocking-plots','save-plots','force-no-plots','do-not-restore-checkpoints','do-not-download','stock=','start-date=','end-date=','test-date=','enrich-dataset','clear-plots-models-and-datasets','analyze-metrics','move-models-to-backup','restore-backups','dummy','run-all-stocks-together','use-all-hyper-on-all-stocks','only-first-hyperparam','do-not-test-hyperparams-with-more-fields'])
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
