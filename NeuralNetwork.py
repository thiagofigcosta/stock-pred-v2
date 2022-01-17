#!/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import re
from tensorflow import keras # import keras 
import shutil
import pareto
import pandas as pd
import numpy as np
import random as rd
import datetime as dt
import tensorflow as tf
from Hyperparameters import Hyperparameters
from CustomStatefulCallback import CustomStatefulCallback
from Dataset import Dataset
from NNDatasetContainer import NNDatasetContainer
from Utils import Utils
from Actuator import Actuator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout # from keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM # from keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model # from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import plot_model # from keras.utils.vis_utils import plot_model
import tensorflow.keras.backend as K # import keras.backend as K
from matplotlib import pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
plt.rcParams.update({'figure.max_open_warning': 0})
import warnings

warnings.filterwarnings('ignore') # turn off warnings for metrics
class NeuralNetwork:
	MODELS_PATH='saved_models/'
	BACKUP_MODELS_PATH='saved_models/backups/'
	SAVED_PLOTS_PATH='saved_plots/'
	SAVED_PLOTS_COUNTER=0
	SAVED_PLOTS_ID=rd.randint(0, 666666)
	CLIP_NORM_INSTEAD_OF_VALUE=False
	FIGURE_EXTRA_WIDTH_RATIO_FOR_HUGE_LEGEND=.3
	FIGURE_WIDTH=1920
	FIGURE_HEIGHT=1080
	FIGURE_DPI=150
	FIGURE_LEGEND_X_ANCHOR=1.0
	FIGURE_LEGEND_Y_ANCHOR=0.5
	MAIN_THREAD_FIGURE_MANAGER=None

	def __init__(self,hyperparameters=None,hyperparameters_path=None,stock_name='undefined',verbose=False):
		if hyperparameters is not None and type(hyperparameters)!=Hyperparameters:
			raise Exception('Wrong hyperparameters object type')
		Utils.createFolder(NeuralNetwork.MODELS_PATH)
		Utils.createFolder(NeuralNetwork.BACKUP_MODELS_PATH)
		Utils.createFolder(NeuralNetwork.SAVED_PLOTS_PATH)
		self.hyperparameters=hyperparameters
		self.verbose=verbose
		self.stock_name=stock_name
		self.data=None
		self.model=None
		self.callbacks=None
		self.history=[]
		self.metrics={}
		if hyperparameters is None:
			if hyperparameters_path is None:
				raise Exception('Either the hyperparameters or the hyperparameters_path must be defined')
			self.filenames={'hyperparameters':hyperparameters_path}
		else:
			self.setFilenames()
		tf.keras.backend.set_epsilon(1)

	@staticmethod
	def setFigureManagerFromMainThread():
		NeuralNetwork.MAIN_THREAD_FIGURE_MANAGER=plt.get_current_fig_manager()

	@staticmethod
	def getFigureManager():
		if NeuralNetwork.MAIN_THREAD_FIGURE_MANAGER is not None:
			return NeuralNetwork.MAIN_THREAD_FIGURE_MANAGER
		return plt.get_current_fig_manager()

	@staticmethod
	def resizeFigure(mng):
		mng.resize(NeuralNetwork.FIGURE_WIDTH, NeuralNetwork.FIGURE_HEIGHT)

	@staticmethod
	def getUuidLabel(uuid, max_uuid_length=10):
		label=uuid[:max_uuid_length]
		if len(uuid)>max_uuid_length:
			label+='...'
		return label

	@staticmethod
	def getNextPlotFilepath(prefix='plot',hyperparameters=None,append_ids=True):
		filename=prefix
		if hyperparameters is not None:
			label=NeuralNetwork.getUuidLabel(hyperparameters.uuid)
			filename+='-'+label
		if append_ids:
			filename+='-id-'+str(NeuralNetwork.SAVED_PLOTS_COUNTER)
			filename+='-gid-'+str(NeuralNetwork.SAVED_PLOTS_ID)
			NeuralNetwork.SAVED_PLOTS_COUNTER+=1
		filename+='.png'
		plotpath=Utils.joinPath(NeuralNetwork.SAVED_PLOTS_PATH,filename)
		return plotpath


	def _metricsFactory(self):
		def r_squared(y_true, y_pred):
			SS_res =  K.sum(K.square( y_true-y_pred ))
			SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
			return ( 1 - SS_res/(SS_tot + K.epsilon()) )
		def r_squared(y_true, y_pred):
			residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
			total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
			r2 = tf.subtract(1.0, tf.divide(residual, total))
			return r2

		r_squared.__name__ = 'R2'
		return {'R2':r_squared}

	def _metricsFactoryArray(self,metrics=[]):
		metrics=metrics.copy()
		if 'R2' in metrics:
			metrics.remove('R2')
			metrics.append(self._metricsFactory()['R2'])
		return metrics

	def setFilenames(self):
		self.basename=self.hyperparameters.uuid+'_'+self.stock_name
		self.filenames={}
		self.filenames['model']=self.basename+'.h5'
		self.filenames['hyperparameters']=self.basename+'_hyperparams.json'
		self.filenames['metrics']=self.basename+'_metrics.json'
		self.filenames['history']=self.basename+'_history.json'
		self.filenames['scaler']=self.basename+'_scaler.bin'
		self.filenames['checkpoint']=None

	def checkTrainedModelExists(self):
		return Utils.checkIfPathExists(self.getModelPath(self.filenames['hyperparameters'])) and Utils.checkIfPathExists(self.getModelPath(self.filenames['model']))

	def destroy(self):
		keras.backend.clear_session()

	def restoreCheckpointWeights(self,delete_after=True):
		if self.filenames['checkpoint'] is not None and Utils.checkIfPathExists(self.getModelPath(self.filenames['checkpoint'])):
			loaded_model=load_model(self.getModelPath(self.filenames['checkpoint']),custom_objects=self._metricsFactory())
			if self.verbose:
				print('Restoring model checkpoint...')
			if self.model is None:
				self.model=loaded_model
			else:
				self.model.set_weights(loaded_model.get_weights())
			if delete_after:
				Utils.deleteFile(self.getModelPath(self.filenames['checkpoint']))

	def load(self):
		self.hyperparameters=Hyperparameters.loadJson(self.getModelPath(self.filenames['hyperparameters']))
		self.setFilenames()
		self.model=load_model(self.getModelPath(self.filenames['model']),custom_objects=self._metricsFactory())
		self.batchSizeWorkaround() # needed to avoid cropping test data
		if self.data is None:
			self.data=NNDatasetContainer(None,Utils.loadObj(self.getModelPath(self.filenames['scaler'])),self.hyperparameters.train_percent,self.hyperparameters.val_percent,self.hyperparameters.backwards_samples,self.hyperparameters.forward_samples,self.hyperparameters.normalize,self.verbose)
		self.history=Utils.loadJson(self.getModelPath(self.filenames['history']))
		if Utils.checkIfPathExists(self.getModelPath(self.filenames['metrics'])):
			self.metrics=Utils.loadJson(self.getModelPath(self.filenames['metrics']))

	def save(self):
		self.model.save(self.getModelPath(self.filenames['model']))
		if self.verbose:
			print('Model saved at {};'.format(self.getModelPath(self.filenames['model'])))
		self.hyperparameters.saveJson(self.getModelPath(self.filenames['hyperparameters']))
		if self.verbose:
			print('Model Hyperparameters saved at {};'.format(self.getModelPath(self.filenames['hyperparameters'])))
		Utils.saveObj(self.data.scaler,self.getModelPath(self.filenames['scaler']))
		if self.verbose:
			print('Model Scaler saved at {};'.format(self.getModelPath(self.filenames['scaler'])))
		Utils.saveJson(self.history,self.getModelPath(self.filenames['history']))
		if self.verbose:
			print('Model History saved at {};'.format(self.getModelPath(self.filenames['history'])))
		Utils.saveJson(self.metrics,self.getModelPath(self.filenames['metrics']))
		if self.verbose:
			print('Model Metrics saved at {};'.format(self.getModelPath(self.filenames['metrics'])))
		
	def train(self):
		batch_size=None
		train_x=self.data.train_x
		train_y=self.data.train_y
		val_x=self.data.val_x
		val_y=self.data.val_y
		if self.hyperparameters.batch_size > 0:
			batch_size=self.hyperparameters.batch_size
			if self.hyperparameters.batch_size > 1:
				new_size_train=int(len(train_x)/batch_size)*batch_size
				train_x=train_x[:new_size_train]
				train_y=train_y[:new_size_train]
				if val_x is not None:
					new_size_val=int(len(val_x)/batch_size)*batch_size
					val_x=val_x[:new_size_val]
					val_y=val_y[:new_size_val]
		self.history=self.model.fit(train_x,train_y,epochs=self.hyperparameters.max_epochs,validation_data=(val_x,val_y),batch_size=batch_size,callbacks=self.callbacks,shuffle=self.hyperparameters.shuffle,verbose=2 if self.verbose else 0)
		self.parseHistoryToVanilla()

	def eval(self,plot=False,plot_training=False, print_prediction=False, blocking_plots=False, save_plots=False):
		self.batchSizeWorkaround() # needed to avoid cropping test data
		# add data to be evaluated
		data_to_eval=[]
		data_to_eval.append({'features':self.data.test_x,'labels':self.data.test_y,'index':self.data.test_start_idx,'name':'test'})		   
		if self.data.val_start_idx is not None:
			data_to_eval.append({'features':self.data.val_x,'labels':self.data.val_y,'index':self.data.val_start_idx,'name':'val'})
		if self.data.train_start_idx is not None:
			data_to_eval.append({'features':self.data.train_x,'labels':self.data.train_y,'index':self.data.train_start_idx,'name':'train'})

		eval_type_name=''
		if len(data_to_eval) >1:
			eval_type_name='train'
		else:
			eval_type_name='test'
	
		# predict values
		for data in data_to_eval:
			data['predicted']=self.model.predict(data['features'],batch_size=self.hyperparameters.batch_size,verbose=1 if self.verbose else 0)

		# join predictions
		full_predicted_values=None
		for i in reversed(range(len(data_to_eval))):
			data=data_to_eval[i]
			to_append=self.data.dataset.reshapeLabelsFromNeuralNetwork(data['predicted'].copy())
			if self.hyperparameters.binary_classifier:
				for y,entry in enumerate(to_append):
					for u,value in enumerate(entry):
						to_append[y][u]=Hyperparameters.valueToClass(value)
			else:
				to_append=self.data.dataset.denormalizeLabelsFromNeuralNetwork(to_append)
			if full_predicted_values is None:
				full_predicted_values=to_append
			else:
				full_predicted_values=np.concatenate((full_predicted_values, to_append))
		
		# get the real values and dates
		real_values=self.data.dataset.getValues(only_main_value=True)
		if type(real_values[0]) is list:
			tmp_real_values=[[] for _ in range(self.hyperparameters.amount_companies)]
			for val in real_values:
				for i in range(self.hyperparameters.amount_companies):
					tmp_real_values[i].append(val[i])
			real_values=tmp_real_values
		else:
			real_values=[real_values]
		dates=[dt.datetime.strptime(d,"%d/%m/%Y").date() for d in self.data.dataset.getIndexes()]

		# process predictions
		all_predictions=[[[] for _ in range(self.hyperparameters.forward_samples)] for _ in range(self.hyperparameters.amount_companies)]
		first_value_predictions=[[] for _ in range(self.hyperparameters.amount_companies)]
		last_value_predictions=[[] for _ in range(self.hyperparameters.amount_companies)]
		mean_value_predictions=[[] for _ in range(self.hyperparameters.amount_companies)]
		fl_mean_value_predictions=[[] for _ in range(self.hyperparameters.amount_companies)]
		for pred_val in full_predicted_values:
			pred_val=np.swapaxes(pred_val,0,1)
			for i in range(self.hyperparameters.amount_companies):
				for j in range(self.hyperparameters.forward_samples):	
					all_predictions[i][j].append(pred_val[i][j])
				first_value_predictions[i].append(pred_val[i][0])
				last_value_predictions[i].append(pred_val[i][-1])
				mean_value_predictions[i].append(pred_val[i].mean())
				fl_mean_value_predictions[i].append((pred_val[i][0]+pred_val[i][-1])/2.0)
				if self.hyperparameters.binary_classifier: # values must be only 1 or 0
					mean_value_predictions[i][-1]=1 if mean_value_predictions[i][-1] >=.5 else 0
					fl_mean_value_predictions[i][-1]=1 if fl_mean_value_predictions[i][-1] >=.5 else 0

		# assign predictions to dataset
		for data in data_to_eval:
			self.data.dataset.setNeuralNetworkResultArray(data['index'],data['predicted'])
		self.data.dataset.revertFromTemporalValues()

		# compute verification predictions
		if eval_type_name == 'test':
			verification_pred_dates=dates[self.hyperparameters.backwards_samples-1:self.hyperparameters.backwards_samples-1+self.hyperparameters.forward_samples]
			verification_all_predictions=[[[] for _ in range(self.hyperparameters.forward_samples)] for _ in range(self.hyperparameters.amount_companies)]
			verification_real_values=[[] for _ in range(self.hyperparameters.amount_companies)]
			verification_previous_real_value=[]
			for i in range(self.hyperparameters.amount_companies):
				verification_real_values[i]=real_values[i][self.hyperparameters.backwards_samples-1:self.hyperparameters.backwards_samples-1+self.hyperparameters.forward_samples]
				verification_previous_real_value.append(real_values[i][self.hyperparameters.backwards_samples-2])
				for j in range(self.hyperparameters.forward_samples):
					verification_all_predictions[i][j]=all_predictions[i][j][:self.hyperparameters.forward_samples-j]

		# compute future predictions
		pred_dates,pred_values=self.data.dataset.getDatesAndPredictions()
		pred_dates=[dt.datetime.strptime(d,"%d/%m/%Y").date() for d in pred_dates]
		tmp_pred_values=[[[] for _ in range(self.hyperparameters.forward_samples)] for _ in range(self.hyperparameters.amount_companies)]
		for i,day_samples in enumerate(pred_values):
			for j,a_prediction in enumerate(day_samples): 
				for k,company in enumerate(a_prediction):
					if self.hyperparameters.binary_classifier:
						company=Hyperparameters.valueToClass(company)
					tmp_pred_values[k][j].append(company)
		pred_values=tmp_pred_values

		# compute metrics
		model_metrics=self.model.evaluate(data_to_eval[-1]['features'][:len(data_to_eval[-1]['labels'])],data_to_eval[-1]['labels'],batch_size=self.hyperparameters.batch_size,verbose=1 if self.verbose else 0)
		aux={}
		has_r2=False
		for i in range(len(model_metrics)):
			aux[self.model.metrics_names[i]] = model_metrics[i]
			if self.model.metrics_names[i]=='R2': # TODO r2 might not be a good metric for this regression
				has_r2=True
		model_metrics=aux
		metrics={'Model Metrics':model_metrics,'Strategy Metrics':[],'Class Metrics':[]}
		for i in range(self.hyperparameters.amount_companies):
			real_value_without_backwards=real_values[i][self.hyperparameters.backwards_samples-1:]
			if self.hyperparameters.binary_classifier:
				swing_return,buy_hold_return,class_metrics_tmp=Actuator.analyzeStrategiesAndClassMetrics(real_value_without_backwards,last_value_predictions[i],binary=True)
			else:
				swing_return,buy_hold_return,class_metrics_tmp=Actuator.analyzeStrategiesAndClassMetrics(real_value_without_backwards,fl_mean_value_predictions[i])
				if has_r2:
					metrics['Model Metrics']['R2_manual_c{}'.format(i)]=Actuator.R2manual(real_value_without_backwards,fl_mean_value_predictions[i])
			viniccius13_return=Actuator.autoBuy13(real_value_without_backwards,fl_mean_value_predictions[i])
			strategy_metrics={}
			class_metrics={}
			if self.hyperparameters.amount_companies>1:
				company_text='{}|{} of {}'.format(self.data.dataset.getDatasetName(at=i),i+1,self.hyperparameters.amount_companies)
				strategy_metrics['Company']=company_text
				class_metrics['Company']=company_text
			strategy_metrics['Daily Swing Trade Return']=swing_return
			strategy_metrics['Buy & Hold Return']=buy_hold_return
			strategy_metrics['Auto13(${}) Return'.format(Actuator.INITIAL_INVESTIMENT)]=viniccius13_return
			for key, value in class_metrics_tmp.items():
				class_metrics[key]=value
			metrics['Strategy Metrics']=strategy_metrics # it was metrics['Strategy Metrics'].append(strategy_metrics) 
			metrics['Class Metrics']=class_metrics # it was metrics['Class Metrics'].append(class_metrics)
			if self.verbose:
				print('Metrics {}:'.format(eval_type_name))
				Utils.printDict(model_metrics,'Model metrics')
				Utils.printDict(class_metrics,'Class metrics')
				Utils.printDict(strategy_metrics,'Strategy metrics')

		# plot trainning stats			
		if plot_training:
			plt.plot(self.history['loss'], label='loss')
			plt.plot(self.history['val_loss'], label='val_loss')
			plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
			plt.title('Training loss of {}'.format(self.data.dataset.name))
			plt.tight_layout(rect=[0, 0, 1.1, 1])
			mng=NeuralNetwork.getFigureManager()
			mng.canvas.set_window_title('Training loss of {}'.format(self.data.dataset.name))
			NeuralNetwork.resizeFigure(mng)
			if save_plots:
				plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_trainning_loss'.format(self.data.dataset.name),hyperparameters=self.hyperparameters), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
				plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
			else:
				if blocking_plots:
					plt.show()
				else:
					plt.show(block=False)
					plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

		# plot data
		if plot:
			for i in range(self.hyperparameters.amount_companies):
				if self.hyperparameters.binary_classifier:
					plt.bar(dates,real_values[i],width=1,color='g',alpha=.8, label='Real - Ups',zorder=0)
					plt.bar(dates,[1 if rv_el==0 else 0 for rv_el in real_values[i]],width=1,color='r',alpha=.8, label='Real - Downs',zorder=0)
					amount_of_bars=4
					real_free_space_ratio=.1
					plt.bar(dates[self.hyperparameters.backwards_samples-1:],[ (1-real_free_space_ratio)/amount_of_bars if real_values[i][self.hyperparameters.backwards_samples-1:][x]==el else 0 for x,el in enumerate(first_value_predictions[i])],bottom=0,width=1,linewidth=0,edgecolor='k', color=Utils.getPlotColorWithIndex(0,colours_to_avoid=['r','g']), label='Pred F macthed')
					plt.bar(dates[self.hyperparameters.backwards_samples-1:],[ (1-real_free_space_ratio)/amount_of_bars if real_values[i][self.hyperparameters.backwards_samples-1:][x]==el else 0 for x,el in enumerate(last_value_predictions[i])],bottom=1*(1-real_free_space_ratio)/amount_of_bars,width=1,linewidth=0,edgecolor='k', color=Utils.getPlotColorWithIndex(1,colours_to_avoid=['r','g']), label='Pred L macthed')
					plt.bar(dates[self.hyperparameters.backwards_samples-1:],[ (1-real_free_space_ratio)/amount_of_bars if real_values[i][self.hyperparameters.backwards_samples-1:][x]==el else 0 for x,el in enumerate(mean_value_predictions[i])],bottom=2*(1-real_free_space_ratio)/amount_of_bars,width=1,linewidth=0,edgecolor='k', color=Utils.getPlotColorWithIndex(2,colours_to_avoid=['r','g']), label='Pred Mean macthed')
					plt.bar(dates[self.hyperparameters.backwards_samples-1:],[ (1-real_free_space_ratio)/amount_of_bars if real_values[i][self.hyperparameters.backwards_samples-1:][x]==el else 0 for x,el in enumerate(fl_mean_value_predictions[i])],bottom=3*(1-real_free_space_ratio)/amount_of_bars,width=1,linewidth=0,edgecolor='k', color=Utils.getPlotColorWithIndex(3,colours_to_avoid=['r','g']), label='Pred FL Mean macthed')
				else:
					plt.plot(dates,real_values[i], color='b',label='Real')
					plt.plot(dates[self.hyperparameters.backwards_samples-1:],first_value_predictions[i], color=Utils.getPlotColorWithIndex(0,colours_to_avoid=['b']), label='Predicted F')
					plt.plot(dates[self.hyperparameters.backwards_samples-1:],last_value_predictions[i], color=Utils.getPlotColorWithIndex(1,colours_to_avoid=['b']), label='Predicted L')
					plt.plot(dates[self.hyperparameters.backwards_samples-1:],mean_value_predictions[i], color=Utils.getPlotColorWithIndex(2,colours_to_avoid=['b']), label='Predicted Mean')
					plt.plot(dates[self.hyperparameters.backwards_samples-1:],fl_mean_value_predictions[i], color=Utils.getPlotColorWithIndex(3,colours_to_avoid=['b']), label='Predicted FL Mean')
				if self.hyperparameters.amount_companies>1:
					plt.title('Stock values {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
				else:
					plt.title('Stock values {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
				plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
				plt.tight_layout(rect=[0, 0, 1.1, 1])
				mng=NeuralNetwork.getFigureManager()
				mng.canvas.set_window_title('Stock {} values of {}'.format(eval_type_name,self.data.dataset.getDatasetName(at=i)))
				NeuralNetwork.resizeFigure(mng)
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_stock_values_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
					plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

		if eval_type_name == 'test':
			ok_rates=[[] for _ in range(self.hyperparameters.amount_companies)]
			total_oks=0
			total_samples=0
			for i in range(self.hyperparameters.amount_companies):
				if print_prediction:
					print('Company {} - {} of {} | {}:'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,NeuralNetwork.getUuidLabel(self.hyperparameters.uuid)))
				summary={}
				for j in reversed(range(self.hyperparameters.forward_samples)):
					if print_prediction:
						print('\tPred {} of {}:'.format((self.hyperparameters.forward_samples-j),self.hyperparameters.forward_samples))
					previous_value=verification_previous_real_value[i]
					tmp_dates=verification_pred_dates[:len(verification_all_predictions[i][j])]
					for k,value in enumerate(verification_all_predictions[i][j]):
						if self.hyperparameters.binary_classifier:
							up=verification_all_predictions[i][j][k]>0
							up_real=verification_real_values[i][k]>0
							up_ok=up_real==up
							if print_prediction:
								print('\t\t{}: pred_up: {} real_up: {} - up: {} '.format(tmp_dates[k],up,up_real,up_ok))
							if tmp_dates[k] not in summary:
								summary[tmp_dates[k]]={'pred_up':0,'real_up':0,'pred_down':0,'real_down':0,'up_ok':0}
							if up:
								summary[tmp_dates[k]]['pred_up']+=1
							else:
								summary[tmp_dates[k]]['pred_down']+=1
							if up_real:
								summary[tmp_dates[k]]['real_up']+=1
							else:
								summary[tmp_dates[k]]['real_down']+=1
							if up_ok:
								summary[tmp_dates[k]]['up_ok']+=1
						else:
							diff=round(verification_all_predictions[i][j][k]-previous_value,2)
							up=diff>0
							diff_from_real=round(verification_all_predictions[i][j][k]-verification_real_values[i][k],2)
							diff_real=round(verification_real_values[i][k]-previous_value,2)
							up_real=diff_real>0
							up_ok=up_real==up
							if print_prediction:
								print('\t\t{}: pred: {:+.2f} pred_delta: {:+.2f} real: {:+.2f} real_delta: {:+.2f} | pred_up: {} real_up: {} - OK: {} | diff_pred-real: {}'.format(tmp_dates[k],round(verification_all_predictions[i][j][k],2),diff,round(verification_real_values[i][k],2),diff_real,up,up_real,up_ok,diff_from_real))
							previous_value=verification_real_values[i][k]
							if tmp_dates[k] not in summary:
								summary[tmp_dates[k]]={'pred_up':0,'real_up':0,'pred_down':0,'real_down':0,'up_ok':0}
							if up:
								summary[tmp_dates[k]]['pred_up']+=1
							else:
								summary[tmp_dates[k]]['pred_down']+=1
							if up_real:
								summary[tmp_dates[k]]['real_up']+=1
							else:
								summary[tmp_dates[k]]['real_down']+=1
							if up_ok:
								summary[tmp_dates[k]]['up_ok']+=1
				if print_prediction:
					Utils.printDict(summary,'Summary - Verify',1)
				for c,verification_date in enumerate(verification_pred_dates):
					oks=float(summary[verification_date]['up_ok'])
					samples=float(self.hyperparameters.forward_samples-c)
					total_oks+=oks
					total_samples+=samples
					ok_rates[i].append(oks/samples*100.0)
			total_ok_rate=total_oks/total_samples*100.0
			if print_prediction:
				print('OK_Rate: {:.2f}%'.format(total_ok_rate))
			metrics['Class Metrics']['OK_Rate']=total_ok_rate

			# plot verification predictions
			if plot:
				for i in range(self.hyperparameters.amount_companies):
					if self.hyperparameters.binary_classifier:
						plt.bar(verification_pred_dates[:len(verification_real_values[i])],verification_real_values[i],width=1,color='g',alpha=.8, label='Real - Ups',zorder=0)
						plt.bar(verification_pred_dates[:len(verification_real_values[i])],[1 if rv_el==0 else 0 for rv_el in verification_real_values[i]],width=1,color='r',alpha=.8, label='Real downs',zorder=0)
						real_free_space_ratio=.2
						for j in reversed(range(self.hyperparameters.forward_samples)):
							j_crescent=self.hyperparameters.forward_samples-j
							plt.bar(verification_pred_dates[:len(verification_all_predictions[i][j])],[ (1-real_free_space_ratio)/self.hyperparameters.forward_samples if verification_real_values[i][x]==el else 0 for x,el in enumerate(verification_all_predictions[i][j]) ],color=Utils.getPlotColorWithIndex(j_crescent,colours_to_avoid=['g','r']),width=0.7,linewidth=0,edgecolor='k',bottom=(1-real_free_space_ratio)*(j/self.hyperparameters.forward_samples), label='Pred {} matched'.format(j_crescent), zorder=j)
					else:
						plt.plot(verification_pred_dates[:len(verification_real_values[i])],verification_real_values[i], '-o', color='b',label='Real')
						for j in reversed(range(self.hyperparameters.forward_samples)):
							j_crescent=self.hyperparameters.forward_samples-j
							plt.plot(verification_pred_dates[:len(verification_all_predictions[i][j])],verification_all_predictions[i][j], '-o',color=Utils.getPlotColorWithIndex(j_crescent,colours_to_avoid=['b']), label='Pred {}'.format(j_crescent), zorder=j)
					if self.hyperparameters.amount_companies>1:
						plt.title('Pred verification values {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
					else:
						plt.title('Pred verification values {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
					plt.xticks(verification_pred_dates,rotation=30,ha='right')
					plt.tight_layout(rect=[0, 0, 1.1, 1])
					mng=NeuralNetwork.getFigureManager()
					mng.canvas.set_window_title('Predictions verification {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					NeuralNetwork.resizeFigure(mng)
					if save_plots:
						plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_preds_verify_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
					else:
						if blocking_plots:
							plt.show()
						else:
							plt.show(block=False)
							plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

				for i in range(self.hyperparameters.amount_companies):
					plt.plot(verification_pred_dates,ok_rates[i], '-o')
					if self.hyperparameters.amount_companies>1:
						plt.title('Verification OK Rate {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
					else:
						plt.title('Verification OK Rate {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					plt.xticks(verification_pred_dates,rotation=30,ha='right')
					plt.tight_layout(rect=[0, 0, 1.1, 1])
					mng=NeuralNetwork.getFigureManager()
					mng.canvas.set_window_title('Verification OK Rate {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					NeuralNetwork.resizeFigure(mng)
					if save_plots:
						plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_okrate_verify_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
					else:
						if blocking_plots:
							plt.show()
						else:
							plt.show(block=False)
							plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

		metrics['Fitness']=Utils.computeNNFitness(metrics,self.hyperparameters.binary_classifier,section=None)

		# print future predictions
		if print_prediction:
			for i in range(self.hyperparameters.amount_companies):
				print('Company {} - {} of {} | {}:'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,NeuralNetwork.getUuidLabel(self.hyperparameters.uuid)))
				summary={}
				for j in reversed(range(self.hyperparameters.forward_samples)):
					print('\tPred {} of {}:'.format((self.hyperparameters.forward_samples-j),self.hyperparameters.forward_samples))
					previous_value=real_values[i][-1]
					tmp_dates=pred_dates[:len(pred_values[i][j])]
					for k,value in enumerate(pred_values[i][j]):
						if self.hyperparameters.binary_classifier:
							up=pred_values[i][j][k]>0
							print('\t\t{}: up: {}'.format(tmp_dates[k],up))
							if tmp_dates[k] not in summary:
								summary[tmp_dates[k]]={'up':0,'down':0}
							if up:
								summary[tmp_dates[k]]['up']+=1
							else:
								summary[tmp_dates[k]]['down']+=1
						else:
							diff=round(pred_values[i][j][k]-previous_value,2)
							up=diff>0
							print('\t\t{}: pred: {:+.2f} delta: {:+.2f} up: {}'.format(tmp_dates[k],round(pred_values[i][j][k],2),diff,up))
							previous_value=pred_values[i][j][k]
							if tmp_dates[k] not in summary:
								summary[tmp_dates[k]]={'up':0,'down':0}
							if up:
								summary[tmp_dates[k]]['up']+=1
							else:
								summary[tmp_dates[k]]['down']+=1
				Utils.printDict(summary,'Summary - Future',1)

		# plot future predictions
		if plot:
			amount_of_previous_data_points=5
			for i in range(self.hyperparameters.amount_companies):
				if self.hyperparameters.binary_classifier:
					if amount_of_previous_data_points>0:
						plt.bar(dates[-amount_of_previous_data_points:],real_values[i][-amount_of_previous_data_points:],width=1,color='g',alpha=0.8, label='Last Real - Ups',zorder=0)
						plt.bar(dates[-amount_of_previous_data_points:],[1 if rv_el==0 else 0 for rv_el in real_values[i][-amount_of_previous_data_points:]],width=1,color='r',alpha=0.8, label='Last Real downs',zorder=0)
					real_free_space_ratio=0
					for j in reversed(range(self.hyperparameters.forward_samples)):
						j_crescent=self.hyperparameters.forward_samples-j
						plt.bar(pred_dates[:len(pred_values[i][j])],[ (1-real_free_space_ratio)/self.hyperparameters.forward_samples if el>0 else 0 for el in pred_values[i][j] ],color=Utils.getPlotColorWithIndex(j_crescent,colours_to_avoid=['r','g']),width=0.7,linewidth=0,edgecolor='k',bottom=(1-real_free_space_ratio)*(j/self.hyperparameters.forward_samples),label='Pred {}'.format(j_crescent), zorder=j)
				else:
					if amount_of_previous_data_points>0:
						plt.plot(dates[-amount_of_previous_data_points:],real_values[i][-amount_of_previous_data_points:], '-o',color='b', label='Real values')
					for j in reversed(range(self.hyperparameters.forward_samples)):
						j_crescent=self.hyperparameters.forward_samples-j
						if amount_of_previous_data_points>0:
							plt.plot([dates[-1],pred_dates[0]],[real_values[i][-1],pred_values[i][j][0]], color='k', zorder=-666, linewidth=0.5)
						plt.plot(pred_dates[:len(pred_values[i][j])],pred_values[i][j], '-o',color=Utils.getPlotColorWithIndex(j_crescent,colours_to_avoid=['b','k']), label='Pred {}'.format(j_crescent), zorder=j)
				if self.hyperparameters.amount_companies>1:
					plt.title('Pred future values {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
				else:
					plt.title('Pred future values {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
				plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
				if amount_of_previous_data_points>0:
					plt.xticks(dates[-amount_of_previous_data_points:]+pred_dates,rotation=30,ha='right')
				else:
					plt.xticks(pred_dates,rotation=30,ha='right')
				plt.tight_layout(rect=[0, 0, 1.1, 1])
				mng=NeuralNetwork.getFigureManager()
				mng.canvas.set_window_title('Predictions future {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
				NeuralNetwork.resizeFigure(mng)
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_preds_future_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
					plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
			
		
		# save metrics
		if Utils.checkIfPathExists(self.getModelPath(self.filenames['metrics'])):
			self.metrics=Utils.loadJson(self.getModelPath(self.filenames['metrics']))
		self.metrics[eval_type_name]=metrics
		Utils.saveJson(self.metrics,self.getModelPath(self.filenames['metrics']),sort_keys=False)
		if self.verbose:
			print('Metrics at {} were updated;'.format(self.getModelPath(self.filenames['metrics'])))
		return metrics
		

	def batchSizeWorkaround(self): # replacement for statefulModelWorkaround
		verbose_bkp=self.verbose
		self.verbose=False
		new_model,_=self._buildModel(force_stateless=True)
		self.verbose=verbose_bkp
		new_model.set_weights(self.model.get_weights())
		self.model=new_model

	def buildModel(self,plot_model_to_file=False):
		self.model,self.callbacks=self._buildModel()
		if plot_model_to_file:
			filepath=NeuralNetwork.getNextPlotFilepath('model_{}'.format(self.hyperparameters.uuid),append_ids=False)
			if self.verbose:
				print('Saving model diagram to file: {}'.format(filepath))
			plot_model(self.model,to_file=filepath,show_shapes=True,show_layer_names=True,rankdir="TB",expand_nested=False,dpi=300) # show_dtype=False,
		
	def _buildModel(self,force_stateless=False):
		model=Sequential()
		input_features_size=len(self.hyperparameters.input_features)
		for l in range(self.hyperparameters.lstm_layers):
			is_stateful=self.hyperparameters.stateful and not force_stateless
			input_shape=(self.hyperparameters.layer_sizes[l],self.hyperparameters.amount_companies*input_features_size)
			return_sequences=True if l+1<self.hyperparameters.lstm_layers else not self.hyperparameters.use_dense_on_output
			batch_input_shape=None
			if l==0 and not force_stateless:
				batch_size=self.hyperparameters.batch_size
				if self.hyperparameters.batch_size==0:
					batch_size=None
				batch_input_shape=tuple([batch_size])+input_shape
			if batch_input_shape is not None:
				model.add(LSTM(self.hyperparameters.layer_sizes[l+1],batch_input_shape=batch_input_shape, stateful=is_stateful, return_sequences=return_sequences,use_bias=self.hyperparameters.bias[l],activation=self.hyperparameters.activation_functions[l],recurrent_activation=self.hyperparameters.recurrent_activation_functions[l],unit_forget_bias=self.hyperparameters.unit_forget_bias[l],recurrent_dropout=self.hyperparameters.recurrent_dropout_values[l],go_backwards=self.hyperparameters.go_backwards[l],time_major=False))
			else:
				model.add(LSTM(self.hyperparameters.layer_sizes[l+1],input_shape=input_shape, stateful=is_stateful, return_sequences=return_sequences,use_bias=self.hyperparameters.bias[l],activation=self.hyperparameters.activation_functions[l],recurrent_activation=self.hyperparameters.recurrent_activation_functions[l],unit_forget_bias=self.hyperparameters.unit_forget_bias[l],recurrent_dropout=self.hyperparameters.recurrent_dropout_values[l],go_backwards=self.hyperparameters.go_backwards[l],time_major=False))
			if self.hyperparameters.dropout_values[l]>0:
				model.add(Dropout(self.hyperparameters.dropout_values[l]))
		output_activation=Hyperparameters.REGRESSION_OUTPUT_ACTIVATION_FUNCTION  # activation=None = 'linear'
		if self.hyperparameters.binary_classifier:
			output_activation=Hyperparameters.BINARY_OUTPUT_ACTIVATION_FUNCTION
		if self.hyperparameters.lstm_layers > 0:
			if self.hyperparameters.use_dense_on_output:
				model.add(Dense(self.hyperparameters.forward_samples*self.hyperparameters.amount_companies,activation=output_activation))
			else:
				model.add(LSTM(self.hyperparameters.forward_samples*self.hyperparameters.amount_companies, activation=output_activation,time_major=False))
		else: # No dense layer for hidden_lstm_layers=0
			input_shape=(self.hyperparameters.backwards_samples,self.hyperparameters.amount_companies*input_features_size)
			batch_size=self.hyperparameters.batch_size
			if self.hyperparameters.batch_size==0 or True: # TODO remove this workaround, for feature we should copy the weights of the trained model to a new model with batch_size==1, to train with a batch size and predict with any
				batch_size=None
			batch_input_shape=tuple([batch_size])+input_shape
			model.add(LSTM(self.hyperparameters.forward_samples*self.hyperparameters.amount_companies,batch_input_shape=batch_input_shape, activation=output_activation,time_major=False))
		
		if self.verbose:
			model_summary_lines=[]
			model.summary(print_fn=lambda x: model_summary_lines.append(x))
			model_summary_str='\n'.join(model_summary_lines)+'\n'
			print(model_summary_str)
		
		clip_dict={}
		if NeuralNetwork.CLIP_NORM_INSTEAD_OF_VALUE:
			clip_dict['clipnorm']=1.0
		else:
			clip_dict['clipvalue']=0.5
		if self.hyperparameters.optimizer=='adam':
			opt=Adam(**clip_dict)
		elif self.hyperparameters.optimizer=='sgd':
			opt=SGD(**clip_dict)
		elif self.hyperparameters.optimizer=='rmsprop':
			opt=RMSprop(**clip_dict)
		else:
			raise Exception('Unknown optimizer {}'.format(self.hyperparameters.optimizer))
		model.compile(loss=self.hyperparameters.loss,optimizer=opt,metrics=self._metricsFactoryArray(self.hyperparameters.model_metrics))
		callbacks=[]
		if self.hyperparameters.patience_epochs_stop>0:
			early_stopping=EarlyStopping(monitor='val_loss', mode='auto', patience=self.hyperparameters.patience_epochs_stop, verbose=1 if self.verbose else 0)
			callbacks.append(early_stopping)
		if self.hyperparameters.patience_epochs_reduce>0:
			reduce_lr=ReduceLROnPlateau(monitor='val_loss', mode='min', factor=self.hyperparameters.reduce_factor, patience=self.hyperparameters.patience_epochs_reduce,verbose=1 if self.verbose else 0)
			callbacks.append(reduce_lr)
		checkpoint_filename=self.basename+'_cp.h5'
		self.filenames['checkpoint']=checkpoint_filename
		checkpoint_filepath=Utils.joinPath(NeuralNetwork.MODELS_PATH,checkpoint_filename)
		checkpoint=ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1 if self.verbose else 0, save_best_only=True, mode='auto')
		callbacks.append(checkpoint)
		if self.hyperparameters.stateful:
			reset_states_after_epoch=CustomStatefulCallback(verbose=self.verbose)
			callbacks.append(reset_states_after_epoch)
		return model,callbacks

	def getModelPath(self,filename):
		path=filename
		if NeuralNetwork.MODELS_PATH+Utils.FILE_SEPARATOR not in path:
			path=Utils.joinPath(NeuralNetwork.MODELS_PATH,filename)
		return path

	def parseNumpyToVanillaRecursivelly(self,element):
		if type(element) == np.ndarray:
			return self.parseNumpyToVanillaRecursivelly(element.tolist())
		elif type(element) == list:
			if len(element) == 0:
				return element
			elif type(element[0]) in (np.float64,np.float32,np.float):
				return list(map(float,element))
			elif type(element[0]) in (np.int64,np.int32,np.int):
				return list(map(int,element))
			else:
				raise Exception('Unhandled type {}'.format(type(element[0])))
		else:
			raise Exception('Unhandled type {}'.format(type(element)))


	def parseHistoryToVanilla(self):
		new_hist = {}
		for key in list(self.history.history.keys()):
			new_hist[key]=self.parseNumpyToVanillaRecursivelly(self.history.history[key])
		self.history=new_hist

	@staticmethod
	def enrichDataset(paths,base_column_name='Close'):
		if type(paths)!=list:
			paths=[paths]
		cloh_enriched_column_names=('oc','oh','ol','ch','cl','lh')
		for path in paths:
			frame=pd.read_csv(path)
			if 'Open' in frame and 'Close' in frame and 'Low' in frame and 'High' in frame:
				open_column=frame['Open']
				close_column=frame['Close']
				low_column=frame['Low']
				high_column=frame['High']
				enriched_columns={}
				for enriched_column_name in cloh_enriched_column_names:
					if enriched_column_name not in frame.columns:
						# enrich
						print('Enriching column {} on {}...'.format(enriched_column_name,path))
						if enriched_column_name == 'oc':
							enriched_column=((close_column-open_column)/open_column).tolist()
						elif enriched_column_name == 'oh':
							enriched_column=((high_column-open_column)/open_column).tolist()
						elif enriched_column_name == 'ol':
							enriched_column=((low_column-open_column)/open_column).tolist()
						elif enriched_column_name == 'ch':
							enriched_column=((high_column-close_column)/close_column).tolist()
						elif enriched_column_name == 'cl':
							enriched_column=((low_column-close_column)/close_column).tolist()
						elif enriched_column_name == 'lh':
							enriched_column=((high_column-low_column)/low_column).tolist()
						enriched_columns[enriched_column_name]=enriched_column
				new_size=len(frame.index)
				for k,v in enriched_columns.items():
					frame.insert(len(frame.columns), k, v[-new_size:])
				frame.to_csv(path,index=False)

		stock_enriched_column_names=('fast_moving_avg','slow_moving_avg','up','log_return','fast_exp_moving_avg','slow_exp_moving_avg')
		for path in paths:
			frame=pd.read_csv(path)
			if base_column_name in frame:
				rows_to_crop=0
				stock_column=frame[base_column_name].tolist()
				enriched_columns={}
				for enriched_column_name in stock_enriched_column_names:
					if enriched_column_name not in frame.columns:
						# enrich
						print('Enriching column {} on {}...'.format(enriched_column_name,path))
						if enriched_column_name == 'fast_moving_avg':
							enriched_column=Utils.calcMovingAverage(stock_column,13)
						elif enriched_column_name == 'slow_moving_avg':
							enriched_column=Utils.calcMovingAverage(stock_column,21)
						elif enriched_column_name == 'up':
							enriched_column=Utils.calcDiffUp(stock_column)
						elif enriched_column_name == 'log_return':
							enriched_column=Utils.logReturn(stock_column)
						elif enriched_column_name == 'fast_exp_moving_avg':
							enriched_column=Utils.calcExpMovingAverage(stock_column,13)
						elif enriched_column_name == 'slow_exp_moving_avg':
							enriched_column=Utils.calcExpMovingAverage(stock_column,21)
						rows_to_crop=max(rows_to_crop,(len(stock_column)-len(enriched_column)))
						enriched_columns[enriched_column_name]=enriched_column
				frame = frame.loc[rows_to_crop:]
				new_size=len(frame.index)
				for k,v in enriched_columns.items():
					frame.insert(len(frame.columns), k, v[-new_size:])
				frame.to_csv(path,index=False)


	def loadTestDataset(self,paths,company_index_array=[0],from_date=None,plot=False,blocking_plots=False, save_plots=False):
		self.loadDataset(paths,company_index_array=company_index_array,from_date=from_date,plot=plot,train_percent=0,val_percent=0,blocking_plots=blocking_plots,save_plots=save_plots)

	# company_index_array defines which dataset files are from each company, enables to load multiple companies and use multiple files for the same company
	# from_date format : '01/03/2002'
	def loadDataset(self,paths,company_index_array=[0],from_date=None,plot=False,train_percent=None,val_percent=None,blocking_plots=False, save_plots=False):
		if type(paths)!=list:
			paths=[paths]
		if train_percent is None:
			train_percent=self.hyperparameters.train_percent
		if val_percent is None:
			val_percent=self.hyperparameters.val_percent

		amount_of_companies=len(set(company_index_array))
		if amount_of_companies!=self.hyperparameters.amount_companies:
			raise Exception('Amount of dataset companies different from the hyperparameters amount of companies')
		if len(company_index_array)!=len(paths):
			raise Exception('Company index array ({}) must have the same lenght than Paths array ({}) '.format(len(company_index_arary),len(paths)))

		# load raw data
		dataset_names_array=[]
		full_data=[]
		dataset_full_name=''
		if amount_of_companies>1:
			dataset_full_name=[]
			frames=[]
			last_company=None
			for i in range(len(company_index_array)):
				company=company_index_array[i]
				path=paths[i]
				if last_company != company:
					last_company=company
					current_filename=Utils.filenameFromPath(path)
					if i!=len(company_index_array)-1:
						current_filename=current_filename.split('_')[0]
					dataset_full_name.append(current_filename.split('_')[0])
					dataset_names_array.append(current_filename)
					if len(frames)>0:
						full_data.append(pd.concat(frames))
						frames=[]
				frames.append(pd.read_csv(path))
			dataset_full_name='+'.join(dataset_full_name)
			if len(frames)>0:
				full_data.append(pd.concat(frames))
				frames=[]
		else:
			dataset_full_name=Utils.filenameFromPath(paths[0])
			dataset_names_array.append(dataset_full_name.split('_')[0])
			if len(paths)>1:
				dataset_full_name+=str(len(paths))
			frames=[]
			for path in paths:
				frames.append(pd.read_csv(path))
			full_data.append(pd.concat(frames))

		# get loaded data arrays
		fields=self.hyperparameters.input_features.copy()
		extra_fields=self.hyperparameters.input_features.copy()
		if self.hyperparameters.output_feature not in fields:
			fields.append(self.hyperparameters.output_feature)
		extra_fields.remove(self.hyperparameters.output_feature)
		datasets_to_load=[]
		extra_dataset_name=''
		for i in range(len(full_data)):
			dataset_to_load={'dates':None,'main_feature':None,'other_features':None}
			if self.hyperparameters.index_feature is not None:
				date_index_array = pd.to_datetime(full_data[i][self.hyperparameters.index_feature])
				if from_date is not None:
					from_date_formated=Utils.timestampToStrDateTime(Utils.dateToTimestamp(from_date),date_format='%Y-%m-%d')
					if len(date_index_array[date_index_array >= from_date_formated])>self.hyperparameters.forward_samples+self.hyperparameters.backwards_samples:
						date_index_array=date_index_array[date_index_array >= from_date_formated]
				full_data[i][self.hyperparameters.index_feature] = date_index_array
				full_data[i].set_index(self.hyperparameters.index_feature, inplace=True)
				dataset_to_load['dates']=[date.strftime('%d/%m/%Y') for date in date_index_array]
			full_data[i]=full_data[i][fields]
			if from_date is not None:
				if pd.to_datetime(from_date,format=Utils.DATE_FORMAT) in full_data[i].index:
					full_data[i]=full_data[i][pd.to_datetime(from_date,format=Utils.DATE_FORMAT):]
					d,m,y=Utils.extractNumbersFromDate(from_date)
					extra_dataset_name='trunc{}{}{}'.format(y,m,d)
				else:
					from_date=None
			dataset_to_load['main_feature']=full_data[i][self.hyperparameters.output_feature].to_list()
			if len(extra_fields)>0:
				dataset_to_load['other_features']=full_data[i][extra_fields].values.tolist()
			datasets_to_load.append(dataset_to_load)	

		# truncate multiple companies
		if amount_of_companies>1:
			first_common_date=None
			if datasets_to_load[0]['dates'] is not None:
				first_common_date=datasets_to_load[0]['dates'][0]
			minimum_size=len(datasets_to_load[0]['main_feature'])
			for dataset in datasets_to_load:
				if len(dataset['main_feature']) < minimum_size:
					minimum_size=len(dataset['main_feature'])
					if dataset['dates'] is not None:
						first_common_date=dataset['dates'][0]
			if first_common_date is not None:
				d,m,y=Utils.extractNumbersFromDate(first_common_date)
				extra_dataset_name='trunc{}{}{}'.format(y,m,d)
			for i in range(len(datasets_to_load)):
				datasets_to_load[i]['main_feature']=datasets_to_load[i]['main_feature'][-minimum_size:]
				if datasets_to_load[i]['dates'] is not None:
					datasets_to_load[i]['dates']=datasets_to_load[i]['dates'][-minimum_size:]
				if datasets_to_load[i]['other_features'] is not None:
					datasets_to_load[i]['other_features']=datasets_to_load[i]['other_features'][-minimum_size:]
		dataset_full_name+=extra_dataset_name

		# parse dataset
		parsed_dataset=Dataset(name=dataset_full_name, dataset_names_array=dataset_names_array)
		for dataset in datasets_to_load:
			parsed_dataset.addCompany(dataset['main_feature'],date_array=dataset['dates'],features_2d_array=dataset['other_features'])

		if self.data is not None:
			scaler=self.data.scaler
		else:
			scaler=tuple()
		self.data=NNDatasetContainer(parsed_dataset,scaler,train_percent,val_percent,self.hyperparameters.backwards_samples,self.hyperparameters.forward_samples,self.hyperparameters.normalize,self.verbose)
		self.data.deployScaler()
		self.data.generateNNArrays()

		# plot
		if plot:
			# plt_indedex=self.data.dataset.getIndexes()
			plt_values=self.data.getValuesSplittedByFeature()
			for c,plt_company in enumerate(plt_values):
				for f,features in enumerate(plt_company):
					if f==0:
						label='Stock Values of {}'.format(self.data.dataset.getDatasetName(at=c))
					else:
						label='{} of {}'.format(self.hyperparameters.input_features[f],self.data.dataset.getDatasetName(at=c))
					# plt.plot(plt_indedex,features, label=label)
					plt.plot(features, label=label)
				plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
				plt_title='Loaded dataset {}'.format(self.data.dataset.getDatasetName(at=c))
				if self.hyperparameters.normalize:
					plt_title+=' - normalized'
				plt.tight_layout(rect=[0, 0, 1.1+NeuralNetwork.FIGURE_EXTRA_WIDTH_RATIO_FOR_HUGE_LEGEND, 1])
				mng=NeuralNetwork.getFigureManager()
				mng.canvas.set_window_title(plt_title)
				NeuralNetwork.resizeFigure(mng)
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_loaded_dataset'.format(self.data.dataset.getDatasetName(at=c)),hyperparameters=self.hyperparameters), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
					plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

	@staticmethod
	def restoreAllBestModelsCPs(print_models=False):
		models={}
		for file_str in os.listdir(NeuralNetwork.MODELS_PATH):
			re_result=re.search(r'([a-z0-9]*).*\.(h5|json)', file_str)
			if re_result:
				model_id=re_result.group(1)
				if model_id not in models:
					models[model_id]=[file_str]
				else:
					models[model_id].append(file_str)
		if print_models:
			models_list = list(models.keys())
			models_list.sort()
			for key in models_list:
				print('Keys: {} len: {}'.format(key,len(models[key])))
		for _,files in models.items():
			checkpoint_filename=None
			model_filename=None
			metrics_filename=None
			last_patience_filename=None
			for file in files:
				if re.search(r'.*_cp\.h5', file):
					checkpoint_filename=file
				elif re.search(r'.*(?<![_cp|_last_patience])\.h5', file):
					model_filename=file
				elif re.search(r'.*(?<!_last_patience)_metrics\.json', file):
					metrics_filename=file
				elif re.search(r'.*_last_patience\.h5', file):
					last_patience_filename=file
			if checkpoint_filename is not None and model_filename is not None and last_patience_filename is None:
				print('Restoring checkpoint {}'.format(checkpoint_filename))
				shutil.move(Utils.joinPath(NeuralNetwork.MODELS_PATH,model_filename),Utils.joinPath(NeuralNetwork.MODELS_PATH,model_filename.split('.')[0]+'_last_patience.h5'))
				shutil.move(Utils.joinPath(NeuralNetwork.MODELS_PATH,checkpoint_filename),Utils.joinPath(NeuralNetwork.MODELS_PATH,model_filename))
				if metrics_filename is not None:
					shutil.move(Utils.joinPath(NeuralNetwork.MODELS_PATH,metrics_filename),Utils.joinPath(NeuralNetwork.MODELS_PATH,metrics_filename.split('_metrics')[0]+'_last_patience_metrics.json'))

	@staticmethod
	def runPareto(use_ok_instead_of_f1,plot,blocking_plots=False,save_plots=False,label=''):
		metrics_canditates=Utils.getFolderPathsThatMatchesPattern(NeuralNetwork.MODELS_PATH,r'[a-zA-Z0-9]*_.*metrics\.json')
		uuids=[]
		f1s=[]
		oks=[]
		mean_squared_errors=[]
		for metrics_canditate in metrics_canditates:
			uuid=Utils.extractARegexGroup(Utils.filenameFromPath(metrics_canditate),r'^([a-zA-Z0-9]*)_.*$')
			metrics=Utils.loadJson(metrics_canditate)
			if 'test' in metrics:
				print('Found test metrics on {}'.format(metrics_canditate))
				uuids.append(uuid)
				f1s.append(metrics['test']['Class Metrics']['f1_monark'])
				oks.append(metrics['test']['Class Metrics']['OK_Rate'])
				mean_squared_errors.append(metrics['test']['Model Metrics']['mean_squared_error'])
		if len(uuids) > 0:
			table=[]
			for i in range(len(uuids)):
				if use_ok_instead_of_f1:
					if oks[i]==oks[i] and mean_squared_errors[i]==mean_squared_errors[i]:
						table.append([uuids[i],oks[i],mean_squared_errors[i]])
				else:
					if f1s[i]==f1s[i] and mean_squared_errors[i]==mean_squared_errors[i]:
						table.append([uuids[i],f1s[i],mean_squared_errors[i]])
			default_epsilon=1e-9
			objectives_size=2 #(f1/ok and mean_squared_error)
			objectives = list(range(1,objectives_size+1)) # indices of objetives
			default_epsilons=[default_epsilon]*objectives_size
			pareto_kwargs={}
			pareto_kwargs['maximize']=[1] # F1/OK must be maximized 
			pareto_kwargs['attribution']=True
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
				if use_ok_instead_of_f1:
					plt.scatter([-ok for ok in oks],mean_squared_errors,label='Solution candidates',color='blue') # f1/ok is inverted because it is a feature to maximize
				else:
					plt.scatter([-f1 for f1 in f1s],mean_squared_errors,label='Solution candidates',color='blue') # f1/ok is inverted because it is a feature to maximize
				plt.scatter([-el for el in solution_coordinates[0]],solution_coordinates[1],label='Optimal solutions',color='red') # f1/ok is inverted because it is a feature to maximize
				if use_ok_instead_of_f1:
					plt.xlabel('ok score')
				else:
					plt.xlabel('f1 score')
				plt.ylabel('mean squared error')
				plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
				plt.title('Pareto search space')
				plt.tight_layout(rect=[0, 0, 1.1, 1])
				mng=NeuralNetwork.getFigureManager()
				mng.canvas.set_window_title('Pareto search space')
				NeuralNetwork.resizeFigure(mng)
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('pareto_space_{}'.format(label)), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
					plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)

				# solutions only
				plt.scatter([-el for el in solution_coordinates[0]],solution_coordinates[1],label='Optimal solutions',color='red') # f1/ok is inverted because it is a feature to maximize
				for i in range(len(solution_labels)):
					label=NeuralNetwork.getUuidLabel(solution_labels[i])
					plt.annotate(label,xy=(-solution_coordinates[0][i],solution_coordinates[1][i]),ha='center',fontsize=8,xytext=(0,8),textcoords='offset points')
				y_offset=max(solution_coordinates[1])*0.1
				plt.ylim([min(solution_coordinates[1])-y_offset, max(solution_coordinates[1])+y_offset])
				if use_ok_instead_of_f1:
					plt.xlabel('ok score')
				else:
					plt.xlabel('f1 score')
				plt.ylabel('mean squared error')
				plt.legend(loc='center left', bbox_to_anchor=(NeuralNetwork.FIGURE_LEGEND_X_ANCHOR, NeuralNetwork.FIGURE_LEGEND_Y_ANCHOR)) # plt.legend(loc='best')
				plt.title('Pareto solutions')
				plt.tight_layout(rect=[0, 0, 1.1, 1])
				mng=NeuralNetwork.getFigureManager()
				mng.canvas.set_window_title('Pareto solutions')
				NeuralNetwork.resizeFigure(mng)
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('pareto_solutions_{}'.format(label)), bbox_inches="tight", dpi=NeuralNetwork.FIGURE_DPI)
					plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure(dpi=NeuralNetwork.FIGURE_DPI)
		else:
			print('Not enough metrics to optimize')


	def datasetMagnitudeCalc(self):
		# real_values=self.data.dataset.getValues(only_main_value=True)
		real_values=self.data.getValuesSplittedByFeature()[0][0] # get feature 0 of company 0

		real_values=real_values[self.hyperparameters.backwards_samples-1:]

		real_train=None
		real_val=None
		real_test=None
		if self.data.train_start_idx is not None:
			real_test=real_values[int(len(real_values)*self.data.train_percent):-self.hyperparameters.forward_samples]
			real_train=real_values[:int(len(real_values)*self.data.train_percent)]
			if self.data.val_start_idx is not None:
				real_val=real_train[int(len(real_train)*(1-self.data.val_percent)):]
				real_train=real_train[:int(len(real_train)*(1-self.data.val_percent))]
		else:
			real_test=real_values[:-self.hyperparameters.forward_samples]
			real_values=None


		if real_values is not None:
			total_sum=sum(real_values)
			count=len(real_values)
			average=0
			if count>0:
				average=total_sum/count
			print('All values sum: {}, average: {}, count: {}'.format(total_sum,average,count))

		if real_train is not None:
			total_sum=sum(real_train)
			count=len(real_train)
			average=0
			if count>0:
				average=total_sum/count
			print('Train values sum: {}, average: {}, count: {}'.format(total_sum,average,count))


		if real_val is not None:
			total_sum=sum(real_val)
			count=len(real_val)
			average=0
			if count>0:
				average=total_sum/count
			print('Val values sum: {}, average: {}, count: {}'.format(total_sum,average,count))


		if real_test is not None:
			total_sum=sum(real_test)
			count=len(real_test)
			average=0
			if count>0:
				average=total_sum/count
			print('Test values sum: {}, average: {}, count: {}'.format(total_sum,average,count))
