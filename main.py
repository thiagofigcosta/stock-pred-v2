#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import getopt
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
	normalize=False
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	
	backwards_samples=20
	forward_samples=7 
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=100
	batch_size=5
	stateful=False
	dropout_values=0
	normalize=False
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	
	backwards_samples=30
	forward_samples=7 
	lstm_layers=3
	layer_sizes=[40,30,20]
	max_epochs=100
	batch_size=5
	stateful=False
	dropout_values=[0,0,0.2]
	normalize=False
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
	normalize=False
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
	normalize=False
	train_percent=.8
	val_percent=.2
	hyperparameters.append(Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent))
	
	return hyperparameters



def run(train_model,eval_model,plot,plot_eval,restore_checkpoints,download_if_needed,stocks):
	crawler=Crawler()

	if 'all' in stocks:
		stocks.remove('all')
		all_known_stocks=['CESP3.SA','CPLE6.SA','CSMG3.SA','ENBR3.SA','TRPL4.SA']
		for stock in all_known_stocks:
			if stock not in stocks:
				stocks.append(stock)

	start_date=Utils.FIRST_DATE
	end_date='07/05/2021'

	start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
	end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))

	filepaths={}
	for stock in stocks:
		filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
		filepath=crawler.getDatasetPath(filename)
		filepaths[stock]=filepath
		if not Utils.checkIfPathExists(filepath) and download_if_needed:
			crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
			# crawler.downloadStockDataCustomInterval(stock,filename,data_range='max') # just example
	
	hyperparameters_tmp=getPredefHyperparams()
	hyperparameters={}
	for i,stock in enumerate(stocks):
		hyperparameters[stock]=hyperparameters_tmp[i]
	hyperparameters_tmp=[]
	
	if train_model:
		for stock in stocks:
			# build and train
			neuralNetwork=NeuralNetwork(hyperparameters[stock],stock_name=stock,verbose=True)
			if not neuralNetwork.checkTrainedModelExists():
				neuralNetwork.loadDataset(filepaths[stock],plot=False)
				neuralNetwork.buildModel()
				neuralNetwork.train()
				neuralNetwork.eval(plot=plot,plot_training=plot)
				neuralNetwork.save()
	
	if restore_checkpoints:
		NeuralNetwork.restoreAllBestModelsCPs() # restore the best models

	if eval_model:
		for stock in stocks:
			# load
			neuralNetwork=NeuralNetwork(hyperparameters[stock],stock_name=stock,verbose=True)
			neuralNetwork.load()
			neuralNetwork.loadTestDataset(filepaths[stock],from_date='10/03/2021')
			neuralNetwork.eval(plot=(plot or plot_eval),print_prediction=True)

	
def main(argv):
	help_str=r'main.py\n\t[-h | --help]\n\t[-t | --train]\n\t[-e | --eval]\n\t[-p | --plot]\n\t[--plot-eval]\n\t[--do-not-restore-checkpoints]\n\t[--do-not-download]\n\t[--stock <stock-name>]\n\t\t*default: all'
	# args vars
	train_model=False
	eval_model=False
	plot=False
	plot_eval=False
	restore_checkpoints=True
	download_if_needed=True
	stocks=[]
	args=[]
	try:
		opts, args = getopt.getopt(argv,'htep',['help','train','eval','plot','plot-eval','do-not-restore-checkpoints','do-not-download','stock='])
	except getopt.GetoptError:
		print (help_str)
		sys.exit(2)
	for opt, arg in opts:
		opt=Utils.removeStrPrefix(Utils.removeStrPrefix(opt,'--'),'-')
		if opt in ('h','help'):
			print (help_str)
			sys.exit()
		elif opt in ('t','train'):
			train_model=True
		elif opt in ('e','eval'):
			eval_model=True
		elif opt in ('p','plot'):
			plot=True
		elif opt == 'plot-eval':
			plot_eval=True
		elif opt == 'do-not-restore-checkpoints':
			restore_checkpoints=False
		elif opt == 'do-not-download':
			download_if_needed=False
		elif opt == 'stock':
			stocks.append(arg.stip())
	if len(stocks)==0:
		stocks.append('all')

	if len(opts) == 0:
		print('No arguments were found, using defaults')
		train_model=True
		eval_model=True
		plot=False
		plot_eval=True
		restore_checkpoints=True
		download_if_needed=True

	run(train_model,eval_model,plot,plot_eval,restore_checkpoints,download_if_needed,stocks)

if __name__ == '__main__':
	delta=-time.time()
	main(sys.argv[1:])
	delta+=time.time()
	print('\n\nTotal run time is {} s'.format(delta))
