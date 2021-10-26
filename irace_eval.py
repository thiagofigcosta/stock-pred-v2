#!/bin/python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import argparse
from Utils import Utils
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Enums import NodeType,Loss,Metric,Optimizers,Features
	
def main(stock,start_date,end_date,test_date,binary_classifier,input_features,output_feature,index_feature,metrics,loss,normalize,backwards_samples,forward_samples,max_epochs,stateful,batch_size,use_dense_on_output,patience_epochs_stop,patience_epochs_reduce,reduce_factor,optimizer,shuffle,lstm_layers,layer_sizes,activation_funcs,recurrent_activation_funcs,dropouts,recurrent_dropouts,bias,unit_forget_bias,go_backwards,datfile,confid=None):
	try:
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
		normalize=bool(normalize)
		optimizer=Optimizers(optimizer)
		shuffle=shuffle.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')

		lstm_layers=int(lstm_layers)
		for l in range(lstm_layers):
			layer_sizes[l]=int(layer_sizes[l])
			activation_funcs[l]=NodeType(activation_funcs[l])
			recurrent_activation_funcs[l]=NodeType(recurrent_activation_funcs[l])
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
		output=Utils.computeNNFitness(neuralNetwork.metrics)
		neuralNetwork.destroy()
	except:
		output=-2147483647 # worst result

	output*=-1 # irace minimizes results
	
	with open(Utils.joinPath('irace',datfile), 'w') as f:
		f.write(str(output))

	
if __name__ == '__main__':
	feature_group=0 #0-6
	binary_classifier=False
	stock_name='T'
	
	if os.getcwd().endswith('irace') or os.getcwd().endswith('irace/'):
		os.chdir('..')
	input_features=Hyperparameters.getFeatureGroups()
	start_date='01/01/2016' #Utils.FIRST_DATE
	end_date='07/05/2021'
	test_date='07/01/2021'
	amount_companies=1
	train_percent=.8
	val_percent=.3
	normalize=True
	if binary_classifier:
		input_features=[Features.UP]+input_features[feature_group]
		output_feature=Features.UP
		index_feature='Date'
		metrics=['accuracy','mean_squared_error']
		loss='categorical_crossentropy'
	else:
		input_features=[Features.CLOSE]+input_features[feature_group]
		output_feature=Features.CLOSE
		index_feature='Date'
		metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
		loss='mean_squared_error'

	ap = argparse.ArgumentParser()
	
	ap.add_argument('--bs', dest='bs', type=int, required=True, help='backwards_samples')
	ap.add_argument('--fs', dest='fs', type=int, required=True, help='forward_samples')
	ap.add_argument('--me', dest='me', type=int, required=True, help='max_epochs')
	ap.add_argument('--st', dest='st', type=str, required=True, help='stateful')
	ap.add_argument('--bts', dest='bts', type=int, required=True, help='batch_size')
	ap.add_argument('--do', dest='do', type=str, required=True, help='use_dense_on_output')
	ap.add_argument('--pes', dest='pes', type=int, required=True, help='patience_epochs_stop')
	ap.add_argument('--per', dest='per', type=int, required=True, help='patience_epochs_reduce')
	ap.add_argument('--rf', dest='rf', type=float, required=True, help='reduce_factor')
	ap.add_argument('--op', dest='op', type=int, required=True, help='optimizer')
	ap.add_argument('--sh', dest='sh', type=str, required=True, help='shuffle')
	ap.add_argument('--lrs', dest='lrs', type=int, required=True, help='lstm_layers')
	ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
	ap.add_argument('--config-id', dest='confid', type=str, required=False, help='config_id')

	pre_parsed_args,remaining_args=ap.parse_known_args()

	lstm_layers=pre_parsed_args.lrs

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

	args,remaining_args=ap.parse_known_args() # parse_args raises error when lstm_layers has not the maximum value
	allowed_args=re.compile(r'^--(ls|af|raf|dr|rdr|bi|ufb|gb)-[0-9]+=(True|False|[0-9.]*)$')
	for arg in remaining_args:
		if not allowed_args.match(arg):
			raise Exception('Invalid arg ({})'.format(arg))

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

	main(stock_name,start_date,end_date,test_date,binary_classifier,input_features,output_feature,index_feature,metrics,loss,normalize,args.bs,args.fs,args.me,args.st,args.bts,args.do,args.pes,args.per,args.rf,args.op,args.sh,args.lrs,layer_sizes,activation_funcs,recurrent_activation_funcs,dropouts,recurrent_dropouts,bias,unit_forget_bias,go_backwards,args.datfile,args.confid)
	
