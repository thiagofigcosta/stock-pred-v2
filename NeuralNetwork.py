#!/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import re
import shutil
import pandas as pd
import numpy as np
import datetime as dt
from Hyperparameters import Hyperparameters
from Dataset import Dataset
from NNDatasetContainer import NNDatasetContainer
from Utils import Utils
from Actuator import Actuator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class NeuralNetwork:
	MODELS_PATH='saved_models/'

	def __init__(self,hyperparameters=None,hyperparameters_path=None,stock_name='undefined',verbose=False):
		if hyperparameters is not None and type(hyperparameters)!=Hyperparameters:
			raise Exception('Wrong hyperparameters object type')
		Utils.createFolder(NeuralNetwork.MODELS_PATH)
		self.hyperparameters=hyperparameters
		self.verbose=verbose
		self.stock_name=stock_name
		self.data=None
		self.model=None
		self.callbacks=None
		self.history=[]
		if hyperparameters is None:
			if hyperparameters_path is None:
				raise Exception('Either the hyperparameters or the hyperparameters_path must be defined')
			self.filenames={'hyperparameters':hyperparameters_path}
		else:
			self.setFilenames()

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

	def load(self):
		self.hyperparameters=Hyperparameters.loadJson(self.getModelPath(self.filenames['hyperparameters']))
		self.setFilenames()
		self.model=load_model(self.getModelPath(self.filenames['model']))
		self.statefulModelWorkaround()
		if self.data is None:
			self.data=NNDatasetContainer(None,Utils.loadObj(self.getModelPath(self.filenames['scaler'])),self.hyperparameters.train_percent,self.hyperparameters.val_percent,self.hyperparameters.backwards_samples,self.hyperparameters.forward_samples,self.hyperparameters.normalize)
		self.history=Utils.loadJson(self.getModelPath(self.filenames['history']))

	def save(self):
		self.model.save(self.getModelPath(self.filenames['model']))
		print('Model saved at {};'.format(self.getModelPath(self.filenames['model'])))
		self.hyperparameters.saveJson(self.getModelPath(self.filenames['hyperparameters']))
		print('Model Hyperparameters saved at {};'.format(self.getModelPath(self.filenames['hyperparameters'])))
		Utils.saveObj(self.data.scaler,self.getModelPath(self.filenames['scaler']))
		print('Model Scaler saved at {};'.format(self.getModelPath(self.filenames['scaler'])))
		Utils.saveJson(self.history,self.getModelPath(self.filenames['history']))
		print('Model History saved at {};'.format(self.getModelPath(self.filenames['history'])))
		
	def train(self):
		self.history=self.model.fit(self.data.train_x,self.data.train_y,epochs=self.hyperparameters.max_epochs,validation_data=(self.data.val_x,self.data.val_y),batch_size=self.hyperparameters.batch_size,callbacks=self.callbacks,shuffle=self.hyperparameters.shuffle,verbose=2)
		self.parseHistoryToVanilla()
		self.statefulModelWorkaround()

	def eval(self,plot=False,plot_training=False, print_prediction=False):
		# add data to be evaluated
		data_to_eval=[]
		data_to_eval.append({'features':self.data.test_x,'labels':self.data.test_y,'index':self.data.test_start_idx,'name':'test'})		   
		if self.data.val_start_idx is not None:
			data_to_eval.append({'features':self.data.val_x,'labels':self.data.val_y,'index':self.data.val_start_idx,'name':'val'})
		if self.data.train_start_idx is not None:
			data_to_eval.append({'features':self.data.train_x,'labels':self.data.train_y,'index':self.data.train_start_idx,'name':'train'})
	
		# predict values
		for data in data_to_eval:
			data['predicted']=self.model.predict(data['features'])

		# join predictions
		full_predicted_values=None
		for i in reversed(range(len(data_to_eval))):
			data=data_to_eval[i]
			to_append=self.data.dataset.reshapeLabelsFromNeuralNetwork(data['predicted'])
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
		first_value_predictions=[real_values[i][:self.hyperparameters.backwards_samples-1] for i in range(self.hyperparameters.amount_companies)]
		last_value_predictions=[real_values[i][:self.hyperparameters.backwards_samples-1] for i in range(self.hyperparameters.amount_companies)]
		mean_value_predictions=[real_values[i][:self.hyperparameters.backwards_samples-1] for i in range(self.hyperparameters.amount_companies)]
		fl_mean_value_predictions=[real_values[i][:self.hyperparameters.backwards_samples-1] for i in range(self.hyperparameters.amount_companies)]
		for pred_val in full_predicted_values:
			pred_val=np.swapaxes(pred_val,0,1)
			for i in range(self.hyperparameters.amount_companies):
				first_value_predictions[i].append(pred_val[i][0])
				last_value_predictions[i].append(pred_val[i][-1])
				mean_value_predictions[i].append(pred_val[i].mean())
				fl_mean_value_predictions[i].append((pred_val[i][0]+pred_val[i][1])/2.0)

		# assign predictions to dataset
		for data in data_to_eval:
			self.data.dataset.setNeuralNetworkResultArray(data['index'],data['predicted'])
		self.data.dataset.revertFromTemporalValues()

		# compute only predictions
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
			swing_return,buy_hold_return,class_metrics_tmp=Actuator.analyzeStrategiesAndClassMetrics(real_values[i],fl_mean_value_predictions[i])
			viniccius13_return=Actuator.autoBuy13(real_values[i],fl_mean_value_predictions[i])
			strategy_metrics={}
			class_metrics={}
			if self.hyperparameters.amount_companies>1:
				company_text='{} of {}'.format(i+1,self.hyperparameters.amount_companies)
				strategy_metrics['Company']=company_text
				class_metrics['Company']=company_text
			strategy_metrics['Daily Swing Trade Return']=swing_return
			strategy_metrics['Buy & Hold Return']=buy_hold_return
			strategy_metrics['Auto13(${}) Return'.format(Actuator.INITIAL_INVESTIMENT)]=viniccius13_return
			for key, value in class_metrics_tmp.items():
				class_metrics[key]=value
			metrics['Strategy Metrics'].append(strategy_metrics)
			metrics['Class Metrics'].append(class_metrics)
			if self.verbose:
				Utils.printDict(model_metrics,'Model metrics')
				Utils.printDict(class_metrics,'Class metrics')
				Utils.printDict(strategy_metrics,'Strategy metrics')

		# plot trainning stats			
		if plot_training:
			plt.plot(self.history['loss'], label='loss')
			plt.plot(self.history['val_loss'], label='val_loss')
			plt.legend(loc='best')
			plt.title('Training loss of {}'.format(self.data.dataset.name))
			plt.show()

		# plot data
		if plot:
			for i in range(self.hyperparameters.amount_companies):
				plt.plot(dates,real_values[i], label='Real')
				plt.plot(dates,real_values[i], color='r', label='Predicted F')
				plt.plot(dates,real_values[i], color='g', label='Predicted L')
				plt.plot(dates,real_values[i], color='c', label='Predicted Mean')
				plt.plot(dates,real_values[i], color='k', label='Predicted FL Mean')
				if self.hyperparameters.amount_companies>1:
					plt.title('Stock values {} | Company {} of {}'.format(self.data.dataset.name,(i+1),self.hyperparameters.amount_companies))
				else:
					plt.title('Stock values {}'.format(self.data.dataset.name))
				plt.legend(loc='best')
				plt.show()

		# plot predictions
		if plot:
			for i in range(self.hyperparameters.amount_companies):
				for j in reversed(range(self.hyperparameters.forward_samples)):
					plt.plot(pred_dates[:len(pred_values[i][j])],pred_values[i][j], '-o', label='Prediction {}'.format(self.hyperparameters.forward_samples-j))
				if self.hyperparameters.amount_companies>1:
					plt.title('Pred values {} | Company {} of {}'.format(self.data.dataset.name,(i+1),self.hyperparameters.amount_companies))
				else:
					plt.title('Pred values {}'.format(self.data.dataset.name))
				plt.legend(loc='best')
				plt.xticks(pred_dates,rotation=30,ha='right')
				plt.tight_layout()
				plt.show()
		
		# print predictions
		if print_prediction:
			pass # TODO code me
		
		# save metrics
		Utils.saveJson(metrics,self.getModelPath(self.filenames['metrics']),sort_keys=False)
		print('Model Metrics saved at {};'.format(self.getModelPath(self.filenames['metrics'])))
		return metrics


	def processPredictedArray(self,Y_pred):
		magic_offset=1 # align pred with real
		Y_pred_first_val=self.unwrapFoldedArray(Y_pred,magic_offset=0)
		Y_pred_last_val=self.unwrapFoldedArray(Y_pred,use_last=True,magic_offset=0)
		Y_pred_mean_val=self.unwrapFoldedArray(Y_pred,use_mean=True,magic_offset=0)
		
		Y_pred_first_val=Y_pred_first_val[magic_offset:]
		Y_pred_last_val=Y_pred_last_val[magic_offset:]
		Y_pred_mean_val=Y_pred_mean_val[magic_offset:]
		Y_pred_fl_mean_val=[(Y_pred_first_val[i]+Y_pred_last_val[i])/2 for i in range(len(Y_pred_first_val))]
		
		return Y_pred_first_val, Y_pred_last_val, Y_pred_mean_val, Y_pred_fl_mean_val
		
	def uncompactMultiCompanyArray(self,compacted_array):
		shape=compacted_array.shape
		newshape=(shape[0], int(shape[1]/self.hyperparameters.amount_companies), self.hyperparameters.amount_companies)
		return np.reshape(compacted_array, newshape=newshape)

	def isolateMultiCompanyArray(self,uncompacted_array):
		isolated_array=[]
		for i in range(self.hyperparameters.amount_companies):
			isolated_array.append([])
		for samples in uncompacted_array:
			for i in range(self.hyperparameters.amount_companies):
				company_sample=[]
				for j in range(len(samples)):
					company_sample.append(samples[j][i])
				
				isolated_array[i].append(company_sample)
		return np.array(isolated_array)

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

	def loadTestDataset(self,paths,company_index_array=[0],from_date=None,plot=False):
		self.loadDataset(paths,company_index_array=company_index_array,from_date=from_date,plot=plot,train_percent=0,val_percent=0)

	# company_index_array defines which dataset files are from each company, enables to load multiple companies and use multiple files for the same company
	# from_date format : '01/03/2002'
	def loadDataset(self,paths,company_index_array=[0],from_date=None,plot=False,train_percent=None,val_percent=None):
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
		full_data=[]
		dataset_name=''
		if amount_of_companies>1:
			dataset_name=[]
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
					dataset_name.append(current_filename)
					if len(frames)>0:
						full_data.append(pd.concat(frames))
						frames=[]
				frames.append(pd.read_csv(path))
			dataset_name='+'.join(dataset_name)
			if len(frames)>0:
				full_data.append(pd.concat(frames))
				frames=[]
		else:
			dataset_name=Utils.filenameFromPath(paths[0])
			if len(paths)>1:
				dataset_name+=str(len(paths))
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
					from_date_formated=Utils.timestampToHumanReadable(Utils.dateToTimestamp(from_date),date_format='%Y-%m-%d')
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
				dataset_to_load['other_features']=full_data[i][extra_fields].to_list()
			datasets_to_load.append(dataset_to_load)	

			if plot:
				if amount_of_companies==1 :
					label='Stock Values of {}'.format(dataset_name)
				else:
					label='Stock Values Company {} from {}'.format(i+1,dataset_name)
				plt.plot(full_data[i], label=label)
				plt.legend(loc='best')
				plt.show()

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
		dataset_name+=extra_dataset_name

		# parse dataset
		parsed_dataset=Dataset(name=dataset_name)
		for dataset in datasets_to_load:
			parsed_dataset.addCompany(dataset['main_feature'],date_array=dataset['dates'],features_2d_array=dataset['other_features'])

		if self.data is not None:
			scaler=self.data.scaler
		else:
			scaler=tuple()
		self.data=NNDatasetContainer(parsed_dataset,scaler,train_percent,val_percent,self.hyperparameters.backwards_samples,self.hyperparameters.forward_samples,self.hyperparameters.normalize)
		self.data.deployScaler()
		self.data.generateNNArrays()

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
