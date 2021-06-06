#!/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
from Hyperparameters import Hyperparameters
from Dataset import Dataset
from DatasetOld import DatasetOld
from Utils import Utils
from Actuator import Actuator
import pandas as pd
import numpy as np
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
		if len(self.hyperparameters.input_features) > 1 or self.hyperparameters.amount_companies>1:
			self.filenames['scaler']=[self.basename+'_scaler_labels.bin',self.basename+'_scaler_feats.bin']
		else:
			self.filenames['scaler']=[self.basename+'_scaler.bin']

	def checkTrainedModelExists(self):
		return Utils.checkIfPathExists(self.getModelPath(self.filenames['hyperparameters'])) and Utils.checkIfPathExists(self.getModelPath(self.filenames['model']))

	def load(self):
		self.hyperparameters=Hyperparameters.loadJson(self.getModelPath(self.filenames['hyperparameters']))
		self.setFilenames()
		self.model=load_model(self.getModelPath(self.filenames['model']))
		self.statefulModelWorkaround()
		if self.data is None:
			self.data=DatasetOld(None,None,None,None,None,None,None,None,None,None,None,None)
		self.data.scalers=[]
		for scaler_filename in self.filenames['scaler']:
			self.data.scalers.append(Utils.loadObj(self.getModelPath(scaler_filename)))
		self.history=Utils.loadJson(self.getModelPath(self.filenames['history']))

	def save(self):
		self.model.save(self.getModelPath(self.filenames['model']))
		print('Model saved at {};'.format(self.getModelPath(self.filenames['model'])))
		self.hyperparameters.saveJson(self.getModelPath(self.filenames['hyperparameters']))
		print('Model Hyperparameters saved at {};'.format(self.getModelPath(self.filenames['hyperparameters'])))
		for i in range(len(self.data.scalers)):
			Utils.saveObj(self.data.scalers[i],self.getModelPath(self.filenames['scaler'][i]))
			print('Model Scaler {} saved at {};'.format(i,self.getModelPath(self.filenames['scaler'][i])))
		Utils.saveJson(self.history,self.getModelPath(self.filenames['history']))
		print('Model History saved at {};'.format(self.getModelPath(self.filenames['history'])))
		
	def train(self):
		self.history=self.model.fit(self.data.train.features,self.data.train.labels,epochs=self.hyperparameters.max_epochs,validation_data=(self.data.val.features,self.data.val.labels),batch_size=self.hyperparameters.batch_size,callbacks=self.callbacks,shuffle=self.hyperparameters.shuffle,verbose=2)
		self.parseHistoryToVanilla()
		self.statefulModelWorkaround()

	def eval(self,data_to_eval=None,plot=False,plot_training=False, print_prediction=False):
		if data_to_eval is not None:
			if type(data_to_eval)!=list:
				data_to_eval=[data_to_eval]
			for data in data_to_eval:
				if type(data)!=DatasetOld.EvalData:
					raise Exception('Wrong data object type')
		else:
			data_to_eval=[]
			if self.data.train_val.features.shape[0] > 0:
				# data_to_eval.append(DatasetOld.EvalData(self.data.train_val.features,real=self.data.train_val.labels,index=self.data.indexes.train)) # TODO era pra ser assim
				data_to_eval.append(DatasetOld.EvalData(self.data.train_val.features,real=self.data.train_val.labels[:-self.hyperparameters.forward_samples, :],index=Utils.estimateNextElements(self.data.indexes.train,((len(self.data.indexes.test)-len(self.data.test.features))-(len(self.data.indexes.train)-len(self.data.train_val.features))))))  # TODO gambiarra avançada  
			data_to_eval.append(DatasetOld.EvalData(self.data.test.features,real=self.data.test.labels,index=self.data.indexes.test))		   
            
		for data in data_to_eval:
			data.predicted=self.model.predict(data.features)

		model_metrics=self.model.evaluate(data.features[:len(data.real)],data.real)
		# model_metrics=self.model.evaluate(data.features,data.real) # TODO test
		aux={}
		for i in range(len(model_metrics)):
			aux[self.model.metrics_names[i]] = model_metrics[i]
		model_metrics=aux

		if self.hyperparameters.normalize:
			for data in data_to_eval:
				data.predicted=self.data.scalers[0].inverse_transform(data.predicted)
				data.real=self.data.scalers[0].inverse_transform(data.real)

		if plot_training:
			plt.plot(self.history['loss'], label='loss')
			plt.plot(self.history['val_loss'], label='val_loss')
			plt.legend(loc='best')
			plt.title('Training loss of {}'.format(self.data.name))
			plt.show()

		if self.hyperparameters.amount_companies>1:
			for data in data_to_eval:
				data.predicted=self.uncompactMultiCompanyArray(data.predicted)
				data.real=self.uncompactMultiCompanyArray(data.real)
				data.predicted=self.isolateMultiCompanyArray(data.predicted)
				data.real=self.isolateMultiCompanyArray(data.real)
		else:
			for data in data_to_eval:
				data.predicted=[data.predicted]
				data.real=[data.real]

		metrics={'Model Metrics':model_metrics,'Strategy Metrics':[],'Class Metrics':[]}


		predicted_array_labels=['Predicted F','Predicted L','Predicted Mean','Predicted FL Mean']
		predicted_array_colors=['r','g','c','k']
		for k in range(len(data_to_eval)):
			data=data_to_eval[k]
			for i in range(self.hyperparameters.amount_companies):
				data.real[i]=self.unwrapFoldedArray(data.real[i])
				data.predicted[i]=self.processPredictedArray(data.predicted[i])

				swing_return,buy_hold_return,class_metrics_tmp=Actuator.analyzeStrategiesAndClassMetrics(data.real[i],data.predicted[i][3]) #3 = FL mean
				viniccius13_return=Actuator.autoBuy13(data.real[i],data.predicted[i][3]) #3 = FL mean

				strategy_metrics={}
				class_metrics={}
				company_text=''
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

				if plot:
					try:
						magic=0 # hyperparameters['amount_companies']-1 #?
						input_size=self.hyperparameters.backwards_samples
						output_size=self.hyperparameters.forward_samples

						plt.plot(data.index[input_size+magic:-output_size+1],data.real[i], label='Real')
						for j in range(len(predicted_array_labels)):
							plt.plot(data.index[input_size+magic:],data.predicted[i][j], color=predicted_array_colors[j], label=predicted_array_labels[j])
							if print_prediction:
								print(self.stock_name)
								print(data.predicted[i][j])
						plt.title('Stock values {} | {} - Company {}'.format(self.data.name,k,company_text))
						plt.legend(loc='best')
						plt.show()
					except Exception as e:
						print("Error on plot")
						print(type(e))
						print(e.args)
						print(e)

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
		for l in range(self.hyperparameters.lstm_layers):
			deepness=1
			input_features_size=len(self.hyperparameters.input_features)
			if input_features_size>1:
				deepness=input_features_size
			elif self.hyperparameters.amount_companies>1:
				deepness=self.hyperparameters.amount_companies
			input_shape=(self.hyperparameters.layer_sizes[l],deepness)
			is_stateful=self.hyperparameters.stateful and not force_stateless
			if is_stateful:
				if l==0:
					batch_input_shape=(self.hyperparameters.batch_size,self.hyperparameters.layer_sizes[l],deepness)
					model.add(LSTM(self.hyperparameters.layer_sizes[l+1],batch_input_shape=batch_input_shape, stateful=is_stateful, return_sequences=True if l+1<self.hyperparameters.lstm_layers else False))
				else:
					model.add(LSTM(self.hyperparameters.layer_sizes[l+1],input_shape=input_shape, stateful=is_stateful, return_sequences=True if l+1<self.hyperparameters.lstm_layers else False))
			else:
				model.add(LSTM(self.hyperparameters.layer_sizes[l+1],input_shape=input_shape, stateful=is_stateful, return_sequences=True if l+1<self.hyperparameters.lstm_layers else False))
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
		self.loadDatasetOld(paths,company_index_array=company_index_array,from_date=from_date,plot=plot,train_percent=0,val_percent=0)

	# company_index_array defines which dataset files are from each company, enables to load multiple companies and use multiple files for the same company
	# from_date format : '01/03/2002'
	def loadDatasetOld(self,paths,company_index_array=[0],from_date=None,plot=False,train_percent=None,val_percent=None):
		if type(paths)!=list:
			paths=[paths]

		if train_percent is None:
			train_percent=self.hyperparameters.train_percent
		if val_percent is None:
			val_percent=self.hyperparameters.val_percent

		full_data=[]
		amount_of_companies=len(set(company_index_array))
		if amount_of_companies!=self.hyperparameters.amount_companies:
			raise Exception('Amount of dataset companies different from the hyperparameters amount of companies')
		if amount_of_companies>1:
			if len(company_index_array)!=len(paths):
				raise Exception('Company index array ({}) must have the same lenght than Paths array ({}) '.format(len(company_index_arary),len(paths)))
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
		fields=self.hyperparameters.input_features.copy()
		if self.hyperparameters.output_feature not in fields:
			fields.append(self.hyperparameters.output_feature)
		fields.remove(self.hyperparameters.output_feature)
		fields.append(self.hyperparameters.output_feature) # ensure that the last one is the result field

		for i in range(len(full_data)):
			date_index_array=None
			if self.hyperparameters.index_feature is not None:
				date_index_array = pd.to_datetime(full_data[i][self.hyperparameters.index_feature])
				if from_date is not None:
					from_date_formated=Utils.timestampToHumanReadable(Utils.dateToTimestamp(from_date),date_format='%Y-%m-%d')
					date_index_array=date_index_array[date_index_array >= from_date_formated]
				full_data[i][self.hyperparameters.index_feature] = date_index_array
				full_data[i].set_index(self.hyperparameters.index_feature, inplace=True)
			full_data[i]=full_data[i][fields]
			if from_date is not None:
				full_data[i]=full_data[i][pd.to_datetime(from_date,format=Utils.DATE_FORMAT):]
				d,m,y=Utils.extractNumbersFromDate(from_date)
				dataset_name+='trunc{}{}{}'.format(y,m,d)
			if plot:
				if amount_of_companies==1 :
					label='Stock Values of {}'.format(dataset_name)
				else:
					label='Stock Values Company {} from {}'.format(i+1,dataset_name)
				
				plt.plot(full_data[i], label=label)
				plt.legend(loc='best')
				plt.show()

		if amount_of_companies==1:
			full_data=full_data[0].values.reshape(full_data[0].shape[0],len(fields))
		else:
			full_data=Utils.truncateArraysOnCommonIndexes(full_data)
			date_index_array=full_data[0].index
			for i in range(len(full_data)):
				full_data[i]=full_data[i].values.reshape(full_data[i].shape[0],len(fields))
			
			tuple_of_companies=tuple(full_data)
			full_data = np.concatenate((tuple_of_companies), axis=1)

		unitedScaler = MinMaxScaler(feature_range=(0, 1))
		if self.hyperparameters.normalize:
			unitedScaler = unitedScaler.fit(full_data)
			if self.verbose:
				print('Min: {}, Max: {}'.format(unitedScaler.data_min_, unitedScaler.data_max_))
			full_plot_data = unitedScaler.transform(full_data)
			if len(self.hyperparameters.input_features) == 1:
				full_data=full_plot_data
			if plot:
				plt.plot(full_plot_data, label='Stock Values of {} normalized'.format(dataset_name))
				plt.legend(loc='best')
				plt.show()
		while full_data.shape[0]<self.hyperparameters.backwards_samples+self.hyperparameters.forward_samples: # TODO gambiarra q nao funciona
			full_data = np.insert(full_data,0,full_data[0],axis=0)
			raise Exception ('Please, increase the amount of data (\'from_date\')') # TODO remove
		data_size=len(full_data)

		X_full_data=[]
		Y_full_data=[]
		if amount_of_companies==1 :
			for i in range(self.hyperparameters.backwards_samples,data_size-self.hyperparameters.forward_samples+1):
				X_full_data.append(np.array([x[:len(self.hyperparameters.input_features)] for x in full_data[i-self.hyperparameters.backwards_samples:i]]))
				Y_full_data.append(np.array([x[-1] for x in full_data[i:i+self.hyperparameters.forward_samples]]))
			# going beyond labels
			for i in range(data_size-self.hyperparameters.forward_samples+1,data_size+1):
				X_full_data.append(np.array([x[:len(self.hyperparameters.input_features)] for x in full_data[i-self.hyperparameters.backwards_samples:i]]))
		else:
			for i in range(self.hyperparameters.backwards_samples,data_size-self.hyperparameters.forward_samples+1):
				X_full_data.append(np.array([Utils.extractElementsFromInsideLists(x) for x in full_data[i-self.hyperparameters.backwards_samples:i]]))
				Y_full_data.append(np.array([Utils.extractElementsFromInsideLists(x,last_instead_of_all_but_last=True) for x in full_data[i:i+self.hyperparameters.forward_samples]]))
			# going beyond labels
			for i in range(data_size-self.hyperparameters.forward_samples+1,data_size+1):
				X_full_data.append(np.array([Utils.extractElementsFromInsideLists(x) for x in full_data[i-self.hyperparameters.backwards_samples:i]]))
		X_full_data=np.array(X_full_data)
		Y_full_data=np.array(Y_full_data)
		if amount_of_companies>1:
			Y_shape=Y_full_data.shape
			Y_full_data = Y_full_data.reshape(Y_shape[0], Y_shape[1]*Y_shape[2])
		
		if len(self.hyperparameters.input_features) > 1 or amount_of_companies>1:
			scalerX = MinMaxScaler(feature_range=(0, 1))
			scalerY = MinMaxScaler(feature_range=(0, 1))
			if self.hyperparameters.normalize:
				instances, time_steps, features = X_full_data.shape
				X_full_data = np.reshape(X_full_data, newshape=(-1, features))
				scalerX = scalerX.fit(X_full_data)
				scalerY = scalerY.fit(Y_full_data)
				X_full_data = scalerX.transform(X_full_data)
				Y_full_data = scalerY.transform(Y_full_data)
				X_full_data = np.reshape(X_full_data, newshape=(instances, time_steps, features))
			firstScaler=scalerX
			secondScaler=scalerY
		else:
			firstScaler=unitedScaler
			secondScaler=None
		
		train_idx=int(len(X_full_data)*train_percent)

		X_train_full=X_full_data[:train_idx]
		Y_train_full=Y_full_data[:train_idx]
		X_test=X_full_data[train_idx:]
		Y_test=Y_full_data[train_idx:]
		X_val=[]
		Y_val=[]
		
		date_index_array=date_index_array.tolist()
		train_date_index=date_index_array[:train_idx+self.hyperparameters.backwards_samples+1]
		test_date_index=date_index_array[train_idx:]
		minutes_step=int(Utils.computeArrayIntervals(date_index_array)*60)
		last_date=date_index_array[len(date_index_array)-1].to_numpy()
		for i in range(1,self.hyperparameters.forward_samples):
			new_date=pd.to_datetime(last_date+np.timedelta64(minutes_step*i,'m'))
			test_date_index.append(new_date)
		
		if val_percent>0:
			X_train,X_val,Y_train,Y_val = train_test_split(X_train_full, Y_train_full, test_size=val_percent)
		else:
			X_train=X_train_full
			Y_train=Y_train_full
		
		scalers=[firstScaler]
		if secondScaler is not None:
			scalers.insert(0,secondScaler)

		print('X_train',X_train.shape)
		print('Y_train',Y_train.shape)
		print()
		print('X_val',X_val.shape)
		print('Y_val',Y_val.shape)
		print()
		print('X_test',X_test.shape)
		print('Y_test',Y_test.shape)

		if self.data is not None:
			scalers=self.data.scalers
		self.data=DatasetOld(dataset_name,scalers,X_train,Y_train,X_val,Y_val,X_test,Y_test,X_train_full,Y_train_full,train_date_index,test_date_index)