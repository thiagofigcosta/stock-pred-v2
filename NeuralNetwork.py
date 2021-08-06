#!/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import re
import keras
import shutil
import pandas as pd
import numpy as np
import random as rd
import datetime as dt
import tensorflow as tf
from Hyperparameters import Hyperparameters
from Dataset import Dataset
from NNDatasetContainer import NNDatasetContainer
from Utils import Utils
from Actuator import Actuator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
plt.rcParams.update({'figure.max_open_warning': 0})

class NeuralNetwork:
	MODELS_PATH='saved_models/'
	BACKUP_MODELS_PATH='saved_models/backups/'
	SAVED_PLOTS_PATH='saved_plots/'
	SAVED_PLOTS_COUNTER=0
	SAVED_PLOTS_ID=rd.randint(0, 666666)

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
			
	@staticmethod
	def getUuidLabel(uuid, max_uuid_length=10):
		label=uuid[:max_uuid_length]
		if len(uuid)>max_uuid_length:
			label+='...'
		return label

	@staticmethod
	def getNextPlotFilepath(prefix='plot',hyperparameters=None):
		if hyperparameters is None:
			filename='{}-id-{}-gid-{}.png'.format(prefix,NeuralNetwork.SAVED_PLOTS_COUNTER,NeuralNetwork.SAVED_PLOTS_ID)
		else:
			label=NeuralNetwork.getUuidLabel(hyperparameters.uuid)
			filename='{}-{}-id-{}-gid-{}.png'.format(prefix,label,NeuralNetwork.SAVED_PLOTS_COUNTER,NeuralNetwork.SAVED_PLOTS_ID)
		plotpath=Utils.joinPath(NeuralNetwork.SAVED_PLOTS_PATH,filename)
		NeuralNetwork.SAVED_PLOTS_COUNTER+=1
		return plotpath

	def setFilenames(self):
		self.basename=self.hyperparameters.uuid+'_'+self.stock_name
		self.filenames={}
		self.filenames['model']=self.basename+'.h5'
		self.filenames['hyperparameters']=self.basename+'_hyperparams.json'
		self.filenames['metrics']=self.basename+'_metrics.json'
		self.filenames['history']=self.basename+'_history.json'
		self.filenames['scaler']=self.basename+'_scaler.bin'

	def checkTrainedModelExists(self):
		return Utils.checkIfPathExists(self.getModelPath(self.filenames['hyperparameters'])) and Utils.checkIfPathExists(self.getModelPath(self.filenames['model']))

	def destroy(self):
		keras.backend.clear_session()

	def load(self):
		self.hyperparameters=Hyperparameters.loadJson(self.getModelPath(self.filenames['hyperparameters']))
		self.setFilenames()
		self.model=load_model(self.getModelPath(self.filenames['model']))
		self.statefulModelWorkaround()
		if self.data is None:
			self.data=NNDatasetContainer(None,Utils.loadObj(self.getModelPath(self.filenames['scaler'])),self.hyperparameters.train_percent,self.hyperparameters.val_percent,self.hyperparameters.backwards_samples,self.hyperparameters.forward_samples,self.hyperparameters.normalize)
		self.history=Utils.loadJson(self.getModelPath(self.filenames['history']))
		if Utils.checkIfPathExists(self.getModelPath(self.filenames['metrics'])):
			self.metrics=Utils.loadJson(self.getModelPath(self.filenames['metrics']))

	def save(self):
		self.model.save(self.getModelPath(self.filenames['model']))
		print('Model saved at {};'.format(self.getModelPath(self.filenames['model'])))
		self.hyperparameters.saveJson(self.getModelPath(self.filenames['hyperparameters']))
		print('Model Hyperparameters saved at {};'.format(self.getModelPath(self.filenames['hyperparameters'])))
		Utils.saveObj(self.data.scaler,self.getModelPath(self.filenames['scaler']))
		print('Model Scaler saved at {};'.format(self.getModelPath(self.filenames['scaler'])))
		Utils.saveJson(self.history,self.getModelPath(self.filenames['history']))
		print('Model History saved at {};'.format(self.getModelPath(self.filenames['history'])))
		Utils.saveJson(self.metrics,self.getModelPath(self.filenames['metrics']))
		print('Model Metrics saved at {};'.format(self.getModelPath(self.filenames['metrics'])))
		
	def train(self):
		self.history=self.model.fit(self.data.train_x,self.data.train_y,epochs=self.hyperparameters.max_epochs,validation_data=(self.data.val_x,self.data.val_y),batch_size=self.hyperparameters.batch_size,callbacks=self.callbacks,shuffle=self.hyperparameters.shuffle,verbose=2)
		self.parseHistoryToVanilla()
		self.statefulModelWorkaround()

	def eval(self,plot=False,plot_training=False, print_prediction=False, blocking_plots=False, save_plots=False):
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
			data['predicted']=self.model.predict(data['features'])

		# join predictions
		full_predicted_values=None
		for i in reversed(range(len(data_to_eval))):
			data=data_to_eval[i]
			to_append=self.data.dataset.reshapeLabelsFromNeuralNetwork(data['predicted'].copy())
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
				fl_mean_value_predictions[i].append((pred_val[i][0]+pred_val[i][1])/2.0)

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
					tmp_pred_values[k][j].append(company)
		pred_values=tmp_pred_values

		# compute metrics
		model_metrics=self.model.evaluate(data_to_eval[-1]['features'][:len(data_to_eval[-1]['labels'])],data_to_eval[-1]['labels'])
		aux={}
		for i in range(len(model_metrics)):
			aux[self.model.metrics_names[i]] = model_metrics[i]
		model_metrics=aux
		metrics={'Model Metrics':model_metrics,'Strategy Metrics':[],'Class Metrics':[]}
		for i in range(self.hyperparameters.amount_companies):
			real_value_without_backwards=real_values[i][self.hyperparameters.backwards_samples-1:]
			swing_return,buy_hold_return,class_metrics_tmp=Actuator.analyzeStrategiesAndClassMetrics(real_value_without_backwards,fl_mean_value_predictions[i])
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
			plt.legend(loc='best')
			plt.title('Training loss of {}'.format(self.data.dataset.name))
			plt.get_current_fig_manager().canvas.set_window_title('Training loss of {}'.format(self.data.dataset.name))
			if save_plots:
				plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_trainning_loss'.format(self.data.dataset.name),hyperparameters=self.hyperparameters))
				plt.figure()
			else:
				if blocking_plots:
					plt.show()
				else:
					plt.show(block=False)
					plt.figure()

		# plot data
		if plot:
			for i in range(self.hyperparameters.amount_companies):
				plt.plot(dates,real_values[i], label='Real')
				plt.plot(dates[self.hyperparameters.backwards_samples-1:],first_value_predictions[i], color='r', label='Predicted F')
				plt.plot(dates[self.hyperparameters.backwards_samples-1:],last_value_predictions[i], color='g', label='Predicted L')
				plt.plot(dates[self.hyperparameters.backwards_samples-1:],mean_value_predictions[i], color='c', label='Predicted Mean')
				plt.plot(dates[self.hyperparameters.backwards_samples-1:],fl_mean_value_predictions[i], color='k', label='Predicted FL Mean')
				if self.hyperparameters.amount_companies>1:
					plt.title('Stock values {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
				else:
					plt.title('Stock values {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
				plt.legend(loc='best')
				plt.get_current_fig_manager().canvas.set_window_title('Stock {} values of {}'.format(eval_type_name,self.data.dataset.getDatasetName(at=i)))
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_stock_values_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters))
					plt.figure()
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure()

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
						diff=round(verification_all_predictions[i][j][k]-previous_value,2)
						up=diff>0
						diff_from_real=round(verification_all_predictions[i][j][k]-verification_real_values[i][k],2)
						diff_real=round(verification_real_values[i][k]-previous_value,2)
						up_real=diff_real>0
						up_ok=up_real==up
						if print_prediction:
							print('\t\t{}: pred: {:+.2f} pred_delta: {:+.2f} real: {:+.2f} real_delta: {:+.2f} | pred_up: {} real_up: {} - up: {} | diff_pred-real: {}'.format(tmp_dates[k],round(verification_all_predictions[i][j][k],2),diff,round(verification_real_values[i][k],2),diff_real,up,up_real,up_ok,diff_from_real))
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
			print('OK_Rate: {:.2f}%'.format(total_ok_rate))
			metrics['Class Metrics']['OK_Rate']=total_ok_rate

			# plot verification predictions
			if plot:
				for i in range(self.hyperparameters.amount_companies):
					plt.plot(verification_pred_dates[:len(verification_real_values[i])],verification_real_values[i], '-o', label='Real')
					for j in reversed(range(self.hyperparameters.forward_samples)):
						plt.plot(verification_pred_dates[:len(verification_all_predictions[i][j])],verification_all_predictions[i][j], '-o', label='Prediction {}'.format(self.hyperparameters.forward_samples-j), zorder=j)
					if self.hyperparameters.amount_companies>1:
						plt.title('Pred verification values {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
					else:
						plt.title('Pred verification values {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					plt.legend(loc='best')
					plt.xticks(verification_pred_dates,rotation=30,ha='right')
					plt.tight_layout()
					plt.get_current_fig_manager().canvas.set_window_title('Predictions verification {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					if save_plots:
						plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_preds_verify_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters))
						plt.figure()
					else:
						if blocking_plots:
							plt.show()
						else:
							plt.show(block=False)
							plt.figure()

				for i in range(self.hyperparameters.amount_companies):
					plt.plot(verification_pred_dates,ok_rates[i], '-o')
					if self.hyperparameters.amount_companies>1:
						plt.title('Verification OK Rate {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
					else:
						plt.title('Verification OK Rate {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					plt.xticks(verification_pred_dates,rotation=30,ha='right')
					plt.tight_layout()
					plt.get_current_fig_manager().canvas.set_window_title('Verification OK Rate {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
					if save_plots:
						plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_okrate_verify_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters))
						plt.figure()
					else:
						if blocking_plots:
							plt.show()
						else:
							plt.show(block=False)
							plt.figure()

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
				if amount_of_previous_data_points>0:
					plt.plot(dates[-amount_of_previous_data_points:],real_values[i][-amount_of_previous_data_points:], '-o', label='Last real values')
				for j in reversed(range(self.hyperparameters.forward_samples)):
					if amount_of_previous_data_points>0:
						plt.plot([dates[-1],pred_dates[0]],[real_values[i][-1],pred_values[i][j][0]], color='k', zorder=-666, linewidth=0.5)
					plt.plot(pred_dates[:len(pred_values[i][j])],pred_values[i][j], '-o', label='Prediction {}'.format(self.hyperparameters.forward_samples-j), zorder=j)
				if self.hyperparameters.amount_companies>1:
					plt.title('Pred future values {} | Company {} of {} | {}'.format(self.data.dataset.getDatasetName(at=i),(i+1),self.hyperparameters.amount_companies,eval_type_name))
				else:
					plt.title('Pred future values {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
				plt.legend(loc='best')
				if amount_of_previous_data_points>0:
					plt.xticks(dates[-amount_of_previous_data_points:]+pred_dates,rotation=30,ha='right')
				else:
					plt.xticks(pred_dates,rotation=30,ha='right')
				plt.tight_layout()
				plt.get_current_fig_manager().canvas.set_window_title('Predictions future {} | {}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name))
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_preds_future_{}'.format(self.data.dataset.getDatasetName(at=i),eval_type_name),hyperparameters=self.hyperparameters))
					plt.figure()
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure()
			
		
		# save metrics
		if Utils.checkIfPathExists(self.getModelPath(self.filenames['metrics'])):
			self.metrics=Utils.loadJson(self.getModelPath(self.filenames['metrics']))
		self.metrics[eval_type_name]=metrics
		Utils.saveJson(self.metrics,self.getModelPath(self.filenames['metrics']),sort_keys=False)
		print('Metrics at {} were updated;'.format(self.getModelPath(self.filenames['metrics'])))
		return metrics
		

	def statefulModelWorkaround(self):
		if self.hyperparameters.stateful: # workaround because model.predict was not working for trained stateful models
			verbose_bkp=self.verbose
			self.verbose=False
			new_model,_=self._buildModel(force_stateless=True)
			self.verbose=verbose_bkp
			new_model.set_weights(self.model.get_weights())
			self.model=new_model

	def buildModel(self):
		self.model,self.callbacks=self._buildModel()
		
	def _buildModel(self,force_stateless=False):
		model=Sequential()
		input_features_size=len(self.hyperparameters.input_features)
		for l in range(self.hyperparameters.lstm_layers):
			input_shape=(self.hyperparameters.layer_sizes[l],self.hyperparameters.amount_companies*input_features_size)
			batch_input_shape=tuple([self.hyperparameters.batch_size])+input_shape
			return_sequences=True if l+1<self.hyperparameters.lstm_layers else False
			is_stateful=self.hyperparameters.stateful and not force_stateless
			if is_stateful:
				if l==0:
					model.add(LSTM(self.hyperparameters.layer_sizes[l+1],batch_input_shape=batch_input_shape, stateful=is_stateful, return_sequences=return_sequences))
				else:
					model.add(LSTM(self.hyperparameters.layer_sizes[l+1],input_shape=input_shape, stateful=is_stateful, return_sequences=return_sequences))
			else:
				model.add(LSTM(self.hyperparameters.layer_sizes[l+1],input_shape=input_shape, stateful=is_stateful, return_sequences=return_sequences))
			if self.hyperparameters.dropout_values[l]>0:
				model.add(Dropout(self.hyperparameters.dropout_values[l]))
		model.add(Dense(self.hyperparameters.forward_samples*self.hyperparameters.amount_companies))
		if self.verbose:
			print(model.summary())
		model.compile(loss=self.hyperparameters.loss,optimizer=self.hyperparameters.optimizer,metrics=self.hyperparameters.model_metrics)
		early_stopping=EarlyStopping(monitor='val_loss', mode='min', patience=self.hyperparameters.patience_epochs,verbose=1)
		checkpoint_filename=self.basename+'_cp.h5'
		self.filenames['checkpoint']=checkpoint_filename
		checkpoint_filepath=Utils.joinPath(NeuralNetwork.MODELS_PATH,checkpoint_filename)
		checkpoint=ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
		callbacks=[early_stopping,checkpoint]
		return model,callbacks

	def getModelPath(self,filename):
		path=filename
		if NeuralNetwork.MODELS_PATH+Utils.FILE_SEPARATOR not in path:
			path=Utils.joinPath(NeuralNetwork.MODELS_PATH,filename)
		return path

	def parseHistoryToVanilla(self):
		new_hist = {}
		for key in list(self.history.history.keys()):
			new_hist[key]=self.history.history[key]
			if type(self.history.history[key]) == np.ndarray:
				new_hist[key] = self.history.history[key].tolist()
			elif type(self.history.history[key]) == list:
				if  type(self.history.history[key][0]) == np.float64:
					new_hist[key] = list(map(float, self.history.history[key]))
		self.history=new_hist


	def enrichDataset(self,paths):
		if type(paths)!=list:
			paths=[paths]
		enriched_column_names=('fast_moving_avg','slow_moving_avg')
		for path in paths:
			frame=pd.read_csv(path)
			rows_to_crop=0
			stock_column=frame[self.hyperparameters.output_feature].tolist()
			enriched_columns={}
			for enriched_column_name in enriched_column_names:
				if enriched_column_name not in frame.columns:
					# enrich
					print('Enriching column {} on {}...'.format(enriched_column_name,path))
					if enriched_column_name == 'fast_moving_avg':
						enriched_column=Utils.calcMovingAverage(stock_column,13)
					elif enriched_column_name == 'slow_moving_avg':
						enriched_column=Utils.calcMovingAverage(stock_column,21)
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
					date_index_array=date_index_array[date_index_array >= from_date_formated]
				full_data[i][self.hyperparameters.index_feature] = date_index_array
				full_data[i].set_index(self.hyperparameters.index_feature, inplace=True)
				dataset_to_load['dates']=[date.strftime('%d/%m/%Y') for date in date_index_array]
			full_data[i]=full_data[i][fields]
			if from_date is not None:
				full_data[i]=full_data[i][pd.to_datetime(from_date,format=Utils.DATE_FORMAT):]
				d,m,y=Utils.extractNumbersFromDate(from_date)
				extra_dataset_name='trunc{}{}{}'.format(y,m,d)
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
		self.data=NNDatasetContainer(parsed_dataset,scaler,train_percent,val_percent,self.hyperparameters.backwards_samples,self.hyperparameters.forward_samples,self.hyperparameters.normalize)
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
						label='Feature {} of Company {}'.format(self.hyperparameters.input_features[f],self.data.dataset.getDatasetName(at=c))
					# plt.plot(plt_indedex,features, label=label)
					plt.plot(features, label=label)
				plt.legend(loc='best')
				plt_title='Loaded dataset {}'.format(self.data.dataset.getDatasetName(at=c))
				if self.hyperparameters.normalize:
					plt_title+=' - normalized'
				plt.get_current_fig_manager().canvas.set_window_title(plt_title)
				if save_plots:
					plt.savefig(NeuralNetwork.getNextPlotFilepath('{}_loaded_dataset'.format(self.data.dataset.getDatasetName(at=c)),hyperparameters=self.hyperparameters))
					plt.figure()
				else:
					if blocking_plots:
						plt.show()
					else:
						plt.show(block=False)
						plt.figure()

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
