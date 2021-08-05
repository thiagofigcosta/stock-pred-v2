#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import getopt
import pareto
import matplotlib
from matplotlib import pyplot as plt
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils

def getPredefHyperparams():
	MAX_EPOCHS=100
	hyperparameters=[]

	backwards_samples=30
	forward_samples=7 
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=MAX_EPOCHS
	batch_size=5
	stateful=False
	dropout_values=0
	normalize=True
	train_percent=.8
	val_percent=.2
	shuffle=False
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent,shuffle=shuffle))
	
	backwards_samples=20
	forward_samples=7 
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=MAX_EPOCHS
	batch_size=5
	stateful=False
	dropout_values=0
	normalize=True
	train_percent=.8
	val_percent=.2
	shuffle=False
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent,shuffle=shuffle))
	
	backwards_samples=30
	forward_samples=7 
	lstm_layers=3
	layer_sizes=[40,30,20]
	max_epochs=MAX_EPOCHS
	batch_size=5
	stateful=False
	dropout_values=[0,0,0.2]
	normalize=True
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	

	backwards_samples=7
	forward_samples=7 
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=MAX_EPOCHS
	batch_size=5
	stateful=True
	dropout_values=0
	normalize=True
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	
	backwards_samples=20
	forward_samples=7 
	lstm_layers=1
	layer_sizes=[25]
	max_epochs=MAX_EPOCHS
	batch_size=5
	stateful=False
	dropout_values=0
	normalize=True
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	
	return hyperparameters



def run(train_model,force_train,eval_model,plot,plot_eval,plot_dataset,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks):
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

	start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
	end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))

	filepaths={}
	for stock in stocks:
		filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
		filepath=crawler.getDatasetPath(filename)
		filepaths[stock]=filepath
		if (not Utils.checkIfPathExists(filepath) and download_if_needed) or force_train:
			crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
			# crawler.downloadStockDataCustomInterval(stock,filename,data_range='max') # just example
	
	hyperparameters_tmp=getPredefHyperparams()
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
				for new_input_field in new_input_fields:
					new_hyperparameters=hyper.copy()
					new_hyperparameters.input_features.append(new_input_field)
					new_hyperparameters.genAndSetUuid()
					hyperparameters[stock].append(new_hyperparameters)
		else:
			hyperparameters_tmp[i].setName('manual tunning - from: {} to: {}'.format(start_date,end_date))
			hyperparameters[stock]=[hyperparameters_tmp[i]]
			for new_input_field in new_input_fields:
				new_hyperparameters=hyperparameters[stock][-1].copy()
				new_hyperparameters.input_features.append(new_input_field)
				new_hyperparameters.genAndSetUuid()
				hyperparameters[stock].append(new_hyperparameters)
	hyperparameters_tmp=[]

	if enrich_dataset:
		for stock in stocks:
			neuralNetwork=NeuralNetwork(hyperparameters[stock][0],stock_name=stock,verbose=True)
			neuralNetwork.enrichDataset(filepaths[stock])
	
	if train_model or force_train:
		for stock in stocks:
			# build and train
			for hyperparameter in hyperparameters[stock]:
				neuralNetwork=NeuralNetwork(hyperparameter,stock_name=stock,verbose=True)
				if not neuralNetwork.checkTrainedModelExists() or force_train:
					neuralNetwork.loadDataset(filepaths[stock],plot=plot_dataset,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.buildModel()
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
				neuralNetwork.loadTestDataset(filepaths[stock],from_date='10/03/2021',blocking_plots=blocking_plots,save_plots=save_plots)
				neuralNetwork.eval(plot=(plot or plot_eval),print_prediction=True,blocking_plots=blocking_plots,save_plots=save_plots)
				neuralNetwork.destroy()

	if analyze_metrics:
		metrics_canditates=Utils.getFolderPathsThatMatchesPattern(NeuralNetwork.MODELS_PATH,r'[a-zA-Z0-9]*_.*metrics\.json')
		uuids=[]
		f1s=[]
		mean_squared_errors=[]
		for metrics_canditate in metrics_canditates:
			uuid=Utils.extractARegexGroup(Utils.filenameFromPath(metrics_canditate),r'^([a-zA-Z0-9]*)_.*$')
			metrics=Utils.loadJson(metrics_canditate)
			if 'test' in metrics:
				print('Found test metrics on {}'.format(metrics_canditate))
				uuids.append(uuid)
				f1s.append(metrics['test']['Class Metrics']['f1_monark'])
				mean_squared_errors.append(metrics['test']['Model Metrics']['mean_squared_error'])
		if len(uuids) > 0:
			table=[]
			for i in range(len(uuids)):
			 	table.append([uuids[i],f1s[i],mean_squared_errors[i]])
			default_epsilon=1e-9
			objectives_size=2 #(f1 and mean_squared_error)
			objectives = list(range(1,objectives_size+1)) # indices of objetives
			default_epsilons=[default_epsilon]*objectives_size
			pareto_kwargs={}
			pareto_kwargs['maximize']=[1] # F1 must be maximized 
			pareto_kwargs['attribution']=True # F1 must be maximized 
			solutions = pareto.eps_sort(table, objectives, default_epsilons,**pareto_kwargs)
			solution_labels=[]
			solution_coordinates=[[] for _ in range(objectives_size)]
			print('Pareto solutions:')
			for solution in solutions:
				solution_labels.append(solution[0])
				for i in range(objectives_size):
					solution_coordinates[i].append(solution[1+i])
				print('\t {}: {}'.format(solution[0],solution[1:]))

			if plot:
				# candidates and solutions
				plt.scatter([-f1 for f1 in f1s],mean_squared_errors,label='Solution candidates',color='blue') # f1 is inverted because it is a feature to maximize
				plt.scatter([-f1 for f1 in solution_coordinates[0]],solution_coordinates[1],label='Optimal solutions',color='red') # f1 is inverted because it is a feature to maximize
				plt.xlabel('f1 score')
				plt.ylabel('mean squared error')
				plt.legend(loc='best')
				plt.title('Pareto search space')
				plt.get_current_fig_manager().canvas.set_window_title('Pareto search space')
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('pareto_space_{}-{}'.format(start_date_formated_for_file,end_date_formated_for_file)))
					plt.figure()
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure()

				# solutions only
				plt.scatter([-f1 for f1 in solution_coordinates[0]],solution_coordinates[1],label='Optimal solutions',color='red') # f1 is inverted because it is a feature to maximize
				for i in range(len(solution_labels)):
					label=NeuralNetwork.getUuidLabel(solution_labels[i])
					plt.annotate(label,xy=(-solution_coordinates[0][i],solution_coordinates[1][i]),ha='center',fontsize=8,xytext=(0,8),textcoords='offset points')
				y_offset=max(solution_coordinates[1])*0.1
				plt.ylim([min(solution_coordinates[1])-y_offset, max(solution_coordinates[1])+y_offset])
				plt.xlabel('f1 score')
				plt.ylabel('mean squared error')
				plt.legend(loc='best')
				plt.title('Pareto solutions')
				plt.get_current_fig_manager().canvas.set_window_title('Pareto solutions')
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('pareto_solutions_{}-{}'.format(start_date_formated_for_file,end_date_formated_for_file)))
					plt.figure()
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure()
		else:
			print('Not enough metrics to optimize')

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

	
def main(argv):

	all_known_stocks=[
						'GOOG','AMD','CSCO','TSLA', # international companies
						'T','IBM', # dividend aristocrats
						'BTC-USD',	# crypto currencies
						'BRL=X', # currency exchange rate 
						r'%5EDJI',r'%5EBVSP', # stock market indexes
						'CESP3.SA','CPLE6.SA','CSMG3.SA','ENBR3.SA','TRPL4.SA' # brazilian stable companies
					]

	python_exec_name=Utils.getPythonExecName()
	help_str='main.py\n\t[-h | --help]\n\t[-t | --train]\n\t[--force-train]\n\t[-e | --eval]\n\t[-p | --plot]\n\t[--plot-eval]\n\t[--plot-dataset]\n\t[--blocking-plots]\n\t[--save-plots]\n\t[--force-no-plots]\n\t[--do-not-restore-checkpoints]\n\t[--do-not-download]\n\t[--stock <stock-name>]\n\t\t*default: all\n\t[--start-date <dd/MM/yyyy>]\n\t[--end-date <dd/MM/yyyy>]\n\t[--enrich-dataset]\n\t[--clear-plots-models-and-datasets]\n\t[--analyze-metrics]\n\t[--move-models-to-backup]\n\t[--restore-backups]\n\t[--dummy]\n\t[--run-all-stocks-together]\n\t[--use-all-hyper-on-all-stocks] *warning: heavy'
	help_str+='\n\n\t\t Example for testing datasets: '
	help_str+=r"""
{python} main.py --dummy --clear-plots-models-and-datasets \
echo -e "2018\n\n" >> log.txt; \
{python} main.py --train --eval --plot --plot-eval --plot-dataset --save-plots --enrich-dataset --start-date 01/01/2018 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt; \
echo -e "\n\n\n\n2015\n\n" >> log.txt; \
{python} main.py --train --eval --plot --plot-eval --plot-dataset --save-plots --enrich-dataset --start-date 01/01/2015 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt; \
echo -e "\n\n\n\nALL\n\n" >> log.txt; \
{python} main.py --train --eval --plot --plot-eval --plot-dataset --save-plots --enrich-dataset --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup >> log.txt \
{python} main.py --dummy --restore-backups >> log.txt; \
echo -e "\n\n\nDONE\n" >> log.txt
	""".format(python=python_exec_name)
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
	stocks=[]
	try:
		opts, _ = getopt.getopt(argv,'htep',['help','train','force-train','eval','plot','plot-eval','plot-dataset','blocking-plots','save-plots','force-no-plots','do-not-restore-checkpoints','do-not-download','stock=','start-date=','end-date=','enrich-dataset','clear-plots-models-and-datasets','analyze-metrics','move-models-to-backup','restore-backups','dummy','run-all-stocks-together','use-all-hyper-on-all-stocks'])
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
		elif opt == 'enrich-dataset':
			enrich_dataset=True
		elif opt == 'clear-plots-models-and-datasets':
			Utils.deleteFile('log.txt')
			print('Clearing contents of: {}'.format(NeuralNetwork.MODELS_PATH))
			Utils.deleteFolderContents(NeuralNetwork.MODELS_PATH)
			print('Clearing contents of: {}'.format(NeuralNetwork.SAVED_PLOTS_PATH))
			Utils.deleteFolderContents(NeuralNetwork.SAVED_PLOTS_PATH)
			print('Clearing contents of: {}'.format(Crawler.DATASET_PATH))
			Utils.deleteFolderContents(Crawler.DATASET_PATH)
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
		print('\tanalyze_metrics:',analyze_metrics)
		print('\tmove_models:',move_models)
		print('\trun_stocks_together:',run_stocks_together)
		print('\tall_hyper_for_all_stocks:',all_hyper_for_all_stocks)

	if run_stocks_together:
		stocks=[stocks]

	for stock in stocks:
		if type(stock) is not list:
			stocks_to_run=[stock]
		else:
			stocks_to_run=stock
		run(train_model,force_train,eval_model,plot and not force_no_plots,plot_eval and not force_no_plots,plot_dataset and not force_no_plots,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks_to_run,start_date,end_date,enrich_dataset,analyze_metrics,move_models,all_hyper_for_all_stocks)

if __name__ == '__main__':
	delta=-time.time()
	main(sys.argv[1:])
	delta+=time.time()
	print('\n\nTotal run time is {}'.format(Utils.msToHumanReadable(delta*1000)))
