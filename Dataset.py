#!/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum
from Utils import Utils

class Dataset:
	class Normalization(Enum):
		DONT_NORMALIZE=0
		NORMALIZE=1
		NORMALIZE_WITH_GAP=2
		NORMALIZE_WITH_EXTERNAL_MAXES=3

	class Entry:
		def __init__(self,index=None,date_index=None,stock_value=None,features=None,backward_values=[],forward_values=None,has_only_indexes=False,has_only_previous_values=False,predicted_values=False):
			self.index=index # of position 0 backward value
			self.date_index=date_index # of position 0 backward value
			self.has_only_indexes=has_only_indexes
			self.has_only_previous_values=has_only_previous_values
			self.predicted_values=predicted_values
			if predicted_values:
				self.has_only_indexes=True
			if stock_value is not None:
				backward_values=[[[stock_value]]]
				forward_values=None
				if features is not None:
					backward_values[0][0]+=features
			self.backward_values=backward_values # 2d array of values
			self.forward_values=forward_values # 2d array of values
			self.next=None # next entry

		def copy(self):
			new_entry=None
			previous_entry=None
			first=None
			cur=self
			while True:
				cur_backward_values=None
				if cur.backward_values is not None:
					cur_backward_values=cur.backward_values.copy()
				cur_forward_values=None
				if cur.forward_values is not None:
					cur_forward_values=cur.forward_values.copy()
				new_entry=Dataset.Entry(index=cur.index,date_index=cur.date_index,has_only_indexes=cur.has_only_indexes,has_only_previous_values=cur.has_only_previous_values,predicted_values=cur.predicted_values,backward_values=cur_backward_values,forward_values=cur_forward_values)
				if first is None:
					first=new_entry
					previous_entry=new_entry
				else:
					previous_entry.next=new_entry
					previous_entry=previous_entry.next
				if cur.next is None:
					break
				cur=cur.next
			return first

		def regenerateIndexes(self):
			index=0
			cur=self
			while True:
				if not cur.has_only_indexes:
					cur.index=index
					index+=1
				if cur.next is None:
					break
				else:
					cur=cur.next

		def getSize(self,consider_indexes=False):
			size=0
			cur=self
			while True:
				if not cur.has_only_indexes or consider_indexes:
					size+=1
				if cur.next is None:
					break
				cur=cur.next
			return size

		def getAbsMaxes(self):
			max_vals=[]
			cur=self
			amount_of_features=None
			while True:
				if not cur.has_only_indexes:
					if cur.backward_values is not None:
						if amount_of_features is None:
							amount_of_features=len(cur.backward_values[0][0])
							max_vals=[0]*amount_of_features
						for a in cur.backward_values:
							for b in a:
								for i,c in enumerate(b):
									if abs(c)>max_vals[i]:
										max_vals[i]=c
				if cur.next is None:
					break
				cur=cur.next
			return tuple(max_vals)

		def print(self):
			i=0
			size=self.getSize(consider_indexes=True)
			cur=self
			while i<size:
				to_print=''
				if cur.date_index is not None:
					to_print+=str(cur.date_index)
				elif cur.index is not None:
					to_print+=str(cur.index)
				if cur.predicted_values:
					to_print+=' *pred*'
				if cur.date_index is not None or cur.index is not None:
					to_print+=': '
				to_print+=str(cur.backward_values)
				if cur.forward_values is not None:
					to_print+=' -> '+str(cur.forward_values)
				print(to_print)
				i+=1
				if cur.next is None:
					break
				cur=cur.next

		def getNthPointer(self,n):
			i=0
			cur=self
			while i<n:
				i+=1
				if cur.next is None:
					break
				cur=cur.next
			return cur

		def getPointerAtIndex(self,index):
			cur=self
			while True:
				if cur.index==index:
					return cur
				if cur.next is None:
					break
				cur=cur.next
			return None

		def getPointerAtDate(self,date):
			cur=self
			while True:
				if cur.date_index==date:
					return cur
				if cur.next is None:
					break
				cur=cur.next
			return None

		def getValues(self,degree=0,only_main_value=False,no_simplification=False):
			val_arr=[]
			if degree > 0:
				cur=self
				to_break=False
				i=0
				while True:
					if i>=degree and cur.backward_values is not None:
						if type(cur.backward_values[0][0]) is list:
							to_append=[]
							for el in cur.backward_values[0]:
								to_append.append(el[0])
							val_arr.append(to_append)
						else:
							val_arr.append([cur.backward_values[0][0]])
					if cur.next is None or to_break:
						break
					if cur.forward_values is not None:
						to_break=True
					cur=cur.next
					i+=1
			if degree == 1:
				val_arr.pop()
			cur=self
			while True:
				if not cur.has_only_indexes:
					if degree>0:
						if cur.forward_values is not None:
							val_arr.append(cur.forward_values[abs(degree-1)])
					else:
						if only_main_value:
							if type(cur.backward_values[abs(degree)][0]) is list:
								to_append=[]
								for el in cur.backward_values[abs(degree)]:
									to_append.append(el[0])
								val_arr.append(to_append)
							else:
								val_arr.append([cur.backward_values[abs(degree)][0]])
						else:	
							val_arr.append(cur.backward_values[abs(degree)])
				if cur.next is None:
					break
				cur=cur.next
			if no_simplification:
				return val_arr
			val_arr_filtered=[]
			for el in val_arr:
				for i in range(2):
					if type(el) is list and len(el)==1:
						el=el[0]
				val_arr_filtered.append(el)
			return val_arr_filtered

		def getIndexes(self,degree=0,int_index=False,get_all=False):
			idx_arr=[]
			cur=self
			extra=0
			while True:
				if int_index:
					idx_arr.append(cur.index)
				else:
					idx_arr.append(cur.date_index)
				if cur.next is None:
					break
				cur=cur.next
				if cur.has_only_indexes:
					extra+=1
			if degree>0 and not get_all:
				idx_arr=idx_arr[degree:]
			if extra>0 and not get_all:
				to_remove=extra-degree
				if to_remove>0:
					idx_arr=idx_arr[:-to_remove]
			return idx_arr

	def getDatesAndPredictions(self):
		if self.converted:
			raise Exception('Already converted, please fill the predictions from NN and revert')
		idx_arr=[]
		pred_arr=[]
		cur=self.data
		while True:
			if cur.predicted_values:
				if cur.date_index is None:
					idx_arr.append(cur.index)
				else:
					idx_arr.append(cur.date_index)
				cur_backward_values=cur.backward_values.copy()
				if type(cur_backward_values) is list:
					if len(cur_backward_values)==1:
						cur_backward_values=cur_backward_values[0]
					else:
						for i,el in enumerate(cur_backward_values):
							if len(el)==1:
								cur_backward_values[i]=el[0]
				pred_arr.append(cur_backward_values)
			if cur.next is None:
				break
			cur=cur.next
		return idx_arr,pred_arr
			
	def convertToTemporalValues(self,back_samples,forward_samples):
		if self.converted:
			raise Exception('Already converted')
		self.converted=not self.converted
		values=self.getValues(no_simplification=True)
		size=self.getSize()
		new_size=size-back_samples+1
		new_useful_size=size-back_samples+1-forward_samples
		new_full_size=new_size+forward_samples
		last=self.data.getNthPointer(size-1)
		if last.date_index is not None:
			next_dates=Utils.getStrNextNWorkDays(last.date_index,forward_samples)
		else:
			next_dates=[None]*forward_samples
		dates=self.getIndexes()
		new_first=None
		cur=new_first
		for i in range(back_samples-1):
			new_el=Dataset.Entry(index=i,date_index=dates[i],backward_values=[values[i]],has_only_previous_values=True)
			if new_first is None:
				new_first=new_el
				cur=new_first
			else:
				cur.next=new_el
				cur=cur.next
		dates=dates[back_samples-1:]+next_dates
		starting_at=self.data.getNthPointer(back_samples-1).index
		if starting_at is None:
			starting_at=0
		indexes=list(range(starting_at,new_full_size+starting_at,1))
		for i in range(new_full_size):
			if i<new_size:
				back_values=[]
				for j in range(back_samples):
					back_values.append(values[back_samples+i-j-1].copy())
				for_values=None
				if i<new_useful_size:
					for_values=[]
					for j in range(forward_samples):
						future_vals=[]
						for k in range(len(values[back_samples+i+j])):
							future_val=values[back_samples+i+j][k]
							to_break=False
							if type(future_val) is list:
								future_val=future_val[0]
							else:
								to_break=True
							future_vals.append(future_val) 
							if to_break:
								break
						for_values.append(future_vals.copy())
				new_el=Dataset.Entry(index=indexes[i],date_index=dates[i],backward_values=back_values,forward_values=for_values)
			else:
				new_el=Dataset.Entry(index=indexes[i],date_index=dates[i],backward_values=None,forward_values=None,has_only_indexes=True)

			if new_first is None:
				new_first=new_el
				cur=new_first
			else:
				cur.next=new_el
				cur=cur.next
		self.data=new_first
		self.converted_params=(back_samples,forward_samples)
		self.regenerateIndexes()

	def revertFromTemporalValues(self):
		if not self.converted:
			raise Exception('Not converted yet')
		self.converted=not self.converted
		back_samples,forward_samples=self.converted_params
		self.converted_params=tuple()
		# get current values
		values=self.getValues(no_simplification=True)
		dates=self.getIndexes(get_all=True)
		# get future values 
		cur=self.data
		missing_values=[]
		has_future=True
		while True:
			if not cur.has_only_previous_values and not cur.has_only_indexes:
				if cur.forward_values is not None:
					missing_values.append(cur.forward_values)
				else:
					has_future=False
					break
			if cur.next is None:
				break
			cur=cur.next
		if has_future:
			missing_values=missing_values[-forward_samples:]
			future_list=[[] for _ in range(forward_samples)]
			for i,list_of_preds in enumerate(missing_values):
				k=0
				for j,el in enumerate(list_of_preds):
					if i+j>=forward_samples-1:
						future_list[k].append(el)
						k+=1
			amount_of_companies=len(future_list[0])
			for lists in future_list:
				new_list=[ [] for _ in range(amount_of_companies) ]
				for j,el in enumerate(lists):
					new_list[j].append(el)
				values.append(new_list)

		values_len=float('inf')
		if values is not None:
			values_len=len(values)
		dates_len=float('inf')
		if dates is not None:
			dates_len=len(dates)
		length=min(dates_len,values_len)
		cur=None
		self.data=None
		for i in range(length):
			cur_date=None
			if dates is not None:
				cur_date=dates[i]
			cur_values=values[i]
			predicted=False
			if type(cur_values[0]) is list:
				if type(cur_values[0][0]) is not list:
					cur_values=[cur_values]
				else:
					predicted=True
			else:
				cur_values=[[cur_values]]
			entry=Dataset.Entry(index=i,date_index=cur_date,backward_values=cur_values,predicted_values=predicted)
			if cur is None:
				cur=entry
			else:
				cur.next=entry
				cur=cur.next
			if self.data is None:
				self.data=entry
		self.regenerateIndexes()

	@staticmethod
	def splitNeuralNetworkArray(np_array,p1_percentage=None,part2_index=None):
		if p1_percentage is not None and (p1_percentage<0 or p1_percentage>1):
			raise Exception('Invalid percentage ({})'.format(p1_percentage))
		if p1_percentage is None and part2_index is None:
			raise Exception('proveide either p1_percentage or part2_index')
		if part2_index is None:
			part2_index=int(len(np_array)*p1_percentage)
		return part2_index, np_array[:part2_index], np_array[part2_index:]

	def getNeuralNetworkArrays(self,include_test_data=False,only_test_data=False,normalization=Normalization.DONT_NORMALIZE,external_maxes=None):
		# X (samples, for/backwards, company, features)
		# Y (samples, for/backwards, company)
		if not self.converted:
			raise Exception('Not converted yet')
		normalize=False
		self.normalization_method=normalization
		if normalization in (Dataset.Normalization.NORMALIZE,Dataset.Normalization.NORMALIZE_WITH_GAP):
			maxes=self.getAbsMaxes()
			if normalization == Dataset.Normalization.NORMALIZE_WITH_GAP:
				maxes=list(maxes)
				for i in range(len(maxes)):
					maxes[i]*=1.1 # gap percentage
				maxes=tuple(maxes)
			self.normalization_params=maxes
			normalize=True
		if normalization == Dataset.Normalization.NORMALIZE_WITH_EXTERNAL_MAXES:
			if external_maxes is None or type(external_maxes) is not tuple:
				raise Exception('Provide an external maxes tuple')
			self.normalization_params=external_maxes
			normalize=True
			
		X=[]
		Y=[]
		i=0
		start_index=None
		cur=self.data
		while True:
			if (not only_test_data and (not cur.has_only_indexes and not cur.has_only_previous_values and (cur.forward_values is not None or include_test_data))) or (only_test_data and cur.forward_values is None and not cur.has_only_previous_values and not cur.has_only_indexes):
				if start_index is None:
					start_index=i
				x=np.array(cur.backward_values, ndmin=2, order='C', subok=True, dtype=float)
				if len(x.shape)==2:
					x=np.flip(x, 1)
					x=np.expand_dims(x,axis=1)
				if normalize:
					for a,el_1 in enumerate(x):
						if type(el_1) in (list,np.ndarray):
							for b,el_2 in enumerate(el_1):
								if type(el_2) in (list,np.ndarray):
									for c,el_3 in enumerate(el_2):
										x[a][b][c]=el_3/float(self.normalization_params[c])
				X.append(x)
				if cur.forward_values is not None:
					y=np.array(cur.forward_values, ndmin=2, order='C', subok=True, dtype=float)
					if normalize:
						for a,el_1 in enumerate(y):
							if type(el_1) in (list,np.ndarray):
								for b,el_2 in enumerate(el_1):
									y[a][b]=el_2/float(self.normalization_params[0])
					Y.append(y)
			if cur.next is None:
				break
			cur=cur.next
			i+=1
		X=np.array(X, order='C', subok=True, dtype=float)
		Y=np.array(Y, order='C', subok=True, dtype=float)
		X=self.reshapeFeaturesToNeuralNetwork(X)
		Y=self.reshapeLabelsToNeuralNetwork(Y)
		return start_index,X,Y

	def setNeuralNetworkResultArray(self,start_index,Y):
		if not self.converted:
			raise Exception('Not converted yet')
		if len(Y.shape)==2:
			Y=self.reshapeLabelsFromNeuralNetwork(Y)
		Y=self.denormalizeLabelsFromNeuralNetwork(Y)
		starting_point=self.data.getPointerAtIndex(start_index)
		if starting_point is None:
			return
		i=0
		cur=starting_point
		while i<len(Y):
			cur.forward_values=Y[i].tolist()
			if cur.next is None:
				break
			cur=cur.next
			i+=1

	def denormalizeLabelsFromNeuralNetwork(self,Y):
		normalize = self.normalization_method is not None and self.normalization_method in (Dataset.Normalization.NORMALIZE,Dataset.Normalization.NORMALIZE_WITH_GAP,Dataset.Normalization.NORMALIZE_WITH_EXTERNAL_MAXES)
		if normalize and (self.normalization_params is None or len(self.normalization_params)==0):
			raise Exception('No normalization params found')
		if normalize:
			for a,el_1 in enumerate(Y):
				if type(el_1) in (list,np.ndarray):
					for b,el_2 in enumerate(el_1):
						Y[a][b]=el_2*float(self.normalization_params[0])
		return Y

	def reshapeFeaturesToNeuralNetwork(self,X):
		in_shape=X.shape
		if in_shape[0]==0:
			return X
		return X.reshape(in_shape[0],in_shape[1],-1)

	def reshapeFeaturesFromNeuralNetwork(self,X):
		in_shape=X.shape
		if in_shape[0]==0:
			return X
		return X.reshape(in_shape[0],in_shape[1],self.companies,-1)

	def reshapeLabelsToNeuralNetwork(self,Y):
		in_shape=Y.shape
		if in_shape[0]==0:
			return Y
		return Y.reshape(in_shape[0],-1)

	def reshapeLabelsFromNeuralNetwork(self,Y):
		in_shape=Y.shape
		if in_shape[0]==0:
			return Y
		return Y.reshape(in_shape[0],-1,self.companies)
	
	def addCompany(self,stock_value_array,date_array=None,features_2d_array=None):
		if self.converted:
			raise Exception('Already converted')
		date_array_len=float('inf')
		if date_array is not None:
			date_array_len=len(date_array)
		stock_value_array_len=float('inf')
		if stock_value_array is not None:
			stock_value_array_len=len(stock_value_array)
		length=min(stock_value_array_len,date_array_len)
		cur=self.data
		first_company=cur is None
		for i in range(length):
			# get data
			cur_date=None
			if date_array is not None:
				cur_date=date_array[i]
			cur_stock=stock_value_array[i]
			cur_features=None
			if features_2d_array is not None:
				cur_features=features_2d_array[i]
				if type(cur_features) is not list:
					if type(cur_features) is tuple:
						cur_features=list(cur_features)
					else:
						cur_features=[cur_features]

			# add data to linked list
			if first_company:
				entry=Dataset.Entry(index=i,stock_value=cur_stock,date_index=cur_date,features=cur_features)
				if cur is None:
					cur=entry
				else:
					cur.next=entry
					cur=cur.next
				if self.data is None:
					self.data=entry
			else:
				if not cur.has_only_indexes:
					new_company_values=[cur_stock]
					if cur_features is not None:
						new_company_values+=cur_features
					if cur_date is None or cur_date==cur.date_index:
						cur.backward_values[0].append(new_company_values)
				if cur.next is None:
					break
				else:
					cur=cur.next
		if first_company:
			self.data.regenerateIndexes()
		self.companies+=1

	def print(self):
		print('Name: '+self.name)
		print('Converted:',self.converted)
		print('Data: ')
		self.data.print()

	def getSize(self,consider_indexes=False):
		return self.data.getSize(consider_indexes=consider_indexes)

	def getValues(self,degree=0,only_main_value=False,no_simplification=False):
		return self.data.getValues(degree=degree,only_main_value=only_main_value,no_simplification=no_simplification)

	def getIndexes(self,degree=0,int_index=False,get_all=False):
		return self.data.getIndexes(degree=degree,int_index=int_index,get_all=get_all)

	def regenerateIndexes(self):
		self.data.regenerateIndexes()

	def copy(self):
		new_dataset=Dataset(name=self.name)
		new_dataset.companies=self.companies
		new_dataset.converted=self.converted
		new_dataset.converted_params=self.converted_params+tuple() # tuple copy
		new_dataset.normalization_method=self.normalization_method
		new_dataset.normalization_params=self.normalization_params+tuple() # tuple copy
		new_dataset.data=self.data.copy()
		return new_dataset
	
	def getAbsMaxes(self):
		return self.data.getAbsMaxes()

	def __init__(self,name):
		self.name=name
		self.data=None
		self.companies=0
		self.converted=False
		self.converted_params=tuple()
		self.normalization_method=None
		self.normalization_params=tuple()
