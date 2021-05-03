#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils


def main(argv):
	crawler=Crawler()

	stocks=[]
	stocks.append('CPLE6.SA')
	stocks.append('CESP3.SA')
	stocks.append('CSMG3.SA')
	stocks.append('ENBR3.SA')
	stocks.append('TRPL4.SA')

	start_date=Utils.timestampToHumanReadable(0)
	end_date='28/04/2021'
	start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
	end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))

	filepaths=[]
	for stock in stocks:
		filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
		filepaths.append(crawler.getDatasetPath(filename))
		if not Utils.checkIfPathExists(filepaths[-1]):
			crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
			# crawler.downloadStockDataCustomInterval(stock,filename,data_range='max') # just example

	backwards_samples=20
	forward_samples=4
	lstm_layers=2
	layer_sizes=[25,15]
	max_epochs=1
	batch_size=5
	stateful=False
	dropout_values=0
	normalize=True
	train_percent=.8
	val_percent=.2
	hyperparameters=Hyperparameters(backwards_samples=backwards_samples,forward_samples=forward_samples,lstm_layers=lstm_layers,max_epochs=max_epochs,batch_size=batch_size,stateful=stateful,dropout_values=dropout_values,layer_sizes=layer_sizes,normalize=normalize,train_percent=train_percent,val_percent=val_percent)
	neuralNetwork=NeuralNetwork(hyperparameters,verbose=True)
	neuralNetwork.loadDataset('datasets/CESP3.SA_daily_19691231-20210428.csv',plot=True)
	neuralNetwork.buildModel()
	neuralNetwork.train()
	neuralNetwork.eval(plot=True)
	neuralNetwork.save()

if __name__ == "__main__":
	delta=-time.time()
	main(sys.argv[1:])
	delta+=time.time()
	print("\n\nTotal run time is {} s".format(delta))


# TODO: rewrite below code
# import shutil
# import sys, getopt
# def restoreBestModelCheckpoint():
# 	print_models=False
# 	models={}
# 	for file_str in os.listdir(MODELS_PATH):
# 		re_result=re.search(r'model_id-([0-9]+(?:-?[a-zA-Z]*?[0-9]*?)_.*?(?=_)_I[0-9]+[a-zA-Z]+).*\.(h5|json)', file_str)
# 		if re_result:
# 			model_id=re_result.group(1)
# 			if model_id not in models:
# 				models[model_id]=[file_str]
# 			else:
# 				models[model_id].append(file_str)
# 	if print_models:
# 		models_list = list(models.keys())
# 		models_list.sort()
# 		for key in models_list:
# 			print('Keys: {} len: {}'.format(key,len(models[key])))
# 	for _,files in models.items():
# 		checkpoint_filename=None
# 		model_filename=None
# 		metrics_filename=None
# 		last_patience_filename=None
# 		for file in files:
# 			if re.search(r'model_id-[0-9]+.*_checkpoint\.h5', file):
# 				checkpoint_filename=file
# 			elif re.search(r'model_id-[0-9]+.*(?<![_checkpoint|_last_patience])\.h5', file):
# 				model_filename=file
# 			elif re.search(r'model_id-[0-9]+.*(?<!_last_patience)_metrics\.json', file):
# 				metrics_filename=file
# 			elif re.search(r'model_id-[0-9]+.*_last_patience\.h5', file):
# 				last_patience_filename=file
# 		if checkpoint_filename is not None and model_filename is not None and last_patience_filename is None:
# 			print('Restoring checkpoint {}'.format(checkpoint_filename))
# 			shutil.move(MODELS_PATH+model_filename,MODELS_PATH+model_filename.split('.')[0]+'_last_patience.h5')
# 			shutil.move(MODELS_PATH+checkpoint_filename,MODELS_PATH+model_filename)
# 			if metrics_filename is not None:
# 				shutil.move(MODELS_PATH+metrics_filename,MODELS_PATH+metrics_filename.split('_metrics')[0]+'_last_patience_metrics.json')


# def removeStrPrefix(text, prefix):
# 	if text.startswith(prefix):
# 		return text[len(prefix):]
# 	return text


# def main(argv):
# 	HELP_STR=r'Pytho{N}.py stock_pred.py [-d|--download-datasets] [[--qp1 | --qp2 | --qp3 | --qp4 | --qp6] [--train_without_plot]] [--test-all-test-trained-models [--start-at <value>] --dataset-paths <values>] [--restore-best-checkpoints] [--train-all-test-models [--start-at <value>] --dataset-paths <values>] [--download-stock [--use-hour-interval] --stock-name <value> --start-date <value> --end-date <value>]'
# 	modules=["download-datasets","download-stock","train-all-test-models","test-all-test-trained-models","restore-best-checkpoints","qp1","qp2","qp3","qp4","qp6"]
# 	modules_to_run=[]
# 	args=[]
# 	use_hour_interval=False
# 	stock_name=''
# 	start_date=''
# 	end_date=''
# 	start_at=0
# 	plot_and_load=True
# 	dataset_paths=[]
# 	try:
# 		opts, args = getopt.getopt(argv,"hd",["use-hour-interval","stock-name=","start-date=","end-date=","start-at=","dataset-paths=","train_without_plot"]+modules)
# 	except getopt.GetoptError:
# 		print (HELP_STR)
# 		sys.exit(2)
# 	for opt, arg in opts:
# 		if opt == '-h':
# 			print (HELP_STR)
# 			sys.exit()
# 		elif opt == "--use-hour-interval":
# 			use_hour_interval=True
# 		elif opt == "--stock-name":
# 			stock_name=arg
# 		elif opt == "--start-date":
# 			try:
# 				extractFromStrDate(arg)
# 			except:
# 				raise Exception('Date must be in format {}'.format(DATE_FORMAT))
# 			start_date=arg
# 		elif opt == "--end-date":
# 			try:
# 				extractFromStrDate(arg)
# 			except:
# 				raise Exception('Date must be in format {}'.format(DATE_FORMAT))
# 			end_date=arg
# 		elif opt == "--start-at":
# 			start_at=int(arg)
# 		elif opt == "--dataset-paths":
# 			dataset_paths=arg.split(',')
# 		elif opt == "--train_without_plot":
# 			plot_and_load=False
# 		else:
# 			modules_to_run.append(opt)
# 	for module in modules_to_run:
# 		module=removeStrPrefix(module,'--')
# 		if module == "download-datasets":
# 			downloadAllReferenceDatasets()
# 		elif module == "train-all-test-models":
# 			trainAllProposedTestModels(dataset_paths,start_at=start_at)
# 		elif module == "test-all-test-trained-models":
# 			trainAllProposedTestModels(dataset_paths,start_at=start_at,plot_and_load=True)
# 		elif module == "restore-best-checkpoints":
# 			restoreBestModelCheckpoint()
# 		elif module == "qp1":
# 			QP1(plot_and_load=plot_and_load)
# 		elif module == "qp2":
# 			QP2(plot_and_load=plot_and_load)
# 		elif module == "qp3":
# 			QP3(plot_and_load=plot_and_load)
# 		elif module == "qp4":
# 			QP4(plot_and_load=plot_and_load)
# 		elif module == "qp6":
# 			QP6(plot_and_load=plot_and_load)
# 		elif module == "download-stock":
# 			start_day,start_month,start_year=extractFromStrDate(start_date)
# 			end_day,end_month,end_year=extractFromStrDate(end_date)
# 			if use_hour_interval :
# 				filename=DATASET_PATH+'{}_I1h_F{}{}{}_T{}{}{}.csv'.format(stock,start_year,start_month,start_day,end_year,end_month,end_day)
# 				getStockOnlineHistoryOneHour(stock,filename,start_timestamp=stringToSTimestamp(start_date),end_timestamp=stringToSTimestamp(end_date))
# 			else:
# 				filename=DATASET_PATH+'{}_I1d_F{}{}{}_T{}{}{}.csv'.format(stock,start_year,start_month,start_day,end_year,end_month,end_day)
# 				getStockHistoryOneDay(stock,filename,start_date=start_date,end_date=end_date)
# 		else:
# 			print("Unkown argument {}".format(module))
# 			print(HELP_STR)
# 			sys.exit(2)