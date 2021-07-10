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
from Utils import Utils

def getPredefHyperparams():
	hyperparameters=[]

	backwards_samples=30
	forward_samples=7 
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=100
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
	max_epochs=100
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
	max_epochs=100
	batch_size=5
	stateful=False
	dropout_values=[0,0,0.2]
	normalize=True
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	

	backwards_samples=20
	forward_samples=7 
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=100
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
	max_epochs=100
	batch_size=5
	stateful=False
	dropout_values=0
	normalize=True
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	
	return hyperparameters



def run(train_model,force_train,eval_model,plot,plot_eval,plot_dataset,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset):
	crawler=Crawler()

	if save_plots:
		matplotlib.use('Agg')

	if 'all' in stocks:
		stocks.remove('all')
		all_known_stocks=['CESP3.SA','CPLE6.SA','CSMG3.SA','ENBR3.SA','TRPL4.SA']
		for stock in all_known_stocks:
			if stock not in stocks:
				stocks.append(stock)

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
	hyperparameters={}
	for i,stock in enumerate(stocks):
		hyperparameters[stock]=[hyperparameters_tmp[i]]
		for new_input_field in ('fast_moving_avg','slow_moving_avg','Volume','Open','High','Low','Close','Adj Close'):
			new_hyperparameters=hyperparameters[stock][-1].copy()
			new_hyperparameters.input_features.append(new_input_field)
			new_hyperparameters.genAndSetUuid()
	hyperparameters_tmp=[]

	if enrich_dataset:
		for stock in stocks:
			neuralNetwork=NeuralNetwork(hyperparameters[stock][0],stock_name=stock,verbose=True)
			neuralNetwork.enrichDataset(filepaths[stock])
	
	if train_model:
		for stock in stocks:
			# build and train
			for hyperparameter in hyperparameters[stock]:
				neuralNetwork=NeuralNetwork(hyperparameter,stock_name=stock,verbose=True)
				if not neuralNetwork.checkTrainedModelExists():
					neuralNetwork.loadDataset(filepaths[stock],plot=plot_dataset,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.buildModel()
					neuralNetwork.train()
					neuralNetwork.eval(plot=plot,plot_training=plot,blocking_plots=blocking_plots,save_plots=save_plots)
					neuralNetwork.save()
	
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

	if not blocking_plots or save_plots:
		plt.close() # delete the last and empty figure
	if not blocking_plots:
		plt.show()

	
def main(argv):
	help_str='main.py\n\t[-h | --help]\n\t[-t | --train]\n\t[--force-train]\n\t[-e | --eval]\n\t[-p | --plot]\n\t[--plot-eval]\n\t[--plot-dataset]\n\t[--blocking-plots]\n\t[--save-plots]\n\t[--force-no-plots]\n\t[--do-not-restore-checkpoints]\n\t[--do-not-download]\n\t[--stock <stock-name>]\n\t\t*default: all\n\t[--start-date <dd/MM/yyyy>]\n\t[--end-date <dd/MM/yyyy>]\n\t[--enrich-dataset]'
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
	stocks=[]
	try:
		opts, _ = getopt.getopt(argv,'htep',['help','train','force-train','eval','plot','plot-eval','plot-dataset','blocking-plots','save-plots','force-no-plots','do-not-restore-checkpoints','do-not-download','stock=','start-date=','end-date=','enrich-dataset'])
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
	if len(stocks)==0:
		stocks.append('all')

	functional_args=('train','force-train','eval')
	if len(opts) == 0 or not any(i in used_args for i in functional_args):
		train_model=True
		force_train=False
		eval_model=True

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

	run(train_model,force_train,eval_model,plot and not force_no_plots,plot_eval and not force_no_plots,plot_dataset and not force_no_plots,blocking_plots,save_plots,restore_checkpoints,download_if_needed,stocks,start_date,end_date,enrich_dataset)

if __name__ == '__main__':
	delta=-time.time()
	main(sys.argv[1:])
	delta+=time.time()
	print('\n\nTotal run time is {}'.format(Utils.msToHumanReadable(delta*1000)))
