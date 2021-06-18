#!/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from Utils import Utils

class Dataset:
	class Entry:
		def __init__(self,index=None,date_index=None,stock_value=None,features=None,backward_values=[],forward_values=None,has_only_indexes=False,has_only_previous_values=False):
			self.index=index # of position 0 backward value
			self.date_index=date_index # of position 0 backward value
			self.has_only_indexes=has_only_indexes
			self.has_only_previous_values=has_only_previous_values
			if stock_value is not None:
				backward_values=[[[stock_value]]]
				forward_values=None
				if features is not None:
					backward_values[0][0]+=features
			self.backward_values=backward_values # 2d array of values
			self.forward_values=forward_values # 2d array of values
			self.next=None # next entry

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

		def printRawValues(self):
			i=0
			size=self.getSize(consider_indexes=True)
			cur=self
			while i<size:
				to_print=''
				if cur.date_index is not None:
					to_print+=str(cur.date_index)+': '
				elif cur.index is not None:
					to_print+=str(cur.index)+': '
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

		def getValues(self,degree=0):
			val_arr=[]
			cur=self
			while True:
				if not cur.has_only_indexes:
					if degree>0:
						if cur.forward_values is not None:
							val_arr.append(cur.forward_values[abs(degree-1)])
						else:
							break
					else:
						val_arr.append(cur.backward_values[abs(degree)])
				if cur.next is None:
					break
				cur=cur.next
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
			first_value_found=False
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
				else:
					first_value_found=True
			if degree>0 and not get_all:
				idx_arr=idx_arr[degree:]
			if extra>0 and not get_all:
				to_remove=extra-degree
				if to_remove>0:
					idx_arr=idx_arr[:-to_remove]
			return idx_arr
			
	def convertToTemporalValues(self,back_samples,forward_samples):
		if self.converted:
			raise Exception('Already converted')
		self.converted=not self.converted
		values=self.getValues()
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
						for_values.append(values[back_samples+i+j].copy())
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

		values=self.getValues()
		dates=self.getIndexes(get_all=True)
		raise Exception('TODO code me')
		self.regenerateIndexes()

	@staticmethod
	def splitNeuralNetworkArray(np_array,percentage):
		raise Exception('TODO not implemented')
		#return part2_star_index, part1, part2

	def getNeuralNetworkArrays(self,include_test_data=False,only_test_data=False):
		# (amostras, for/backwards, empresa, features)
		if not self.converted:
			raise Exception('Not converted yet')
		X=[]
		Y=[]
		i=0
		start_index=None
		cur=self.data
		while True:
			if (not only_test_data and (not cur.has_only_indexes and not cur.has_only_previous_values and (cur.forward_values is not None or include_test_data))) or (only_test_data and cur.forward_values is None and not cur.has_only_previous_values and not cur.has_only_indexes):
				if start_index is None:
					start_index=i
				x=np.array(cur.backward_values, ndmin=2, order='C', subok=True)
				if len(x.shape)==2:
					x=np.expand_dims(x,axis=2)
				X.append(x)
				if cur.forward_values is not None:
					y=np.array(cur.forward_values, ndmin=2, order='C', subok=True)
					if len(y.shape)==2:
						y=np.expand_dims(y,axis=2)
					Y.append(y)
			if cur.next is None:
				break
			cur=cur.next
			i+=1
		return start_index,np.array(X, order='C', subok=True),np.array(Y, order='A', subok=True)
	
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

	def printRawValues(self):
		print('Name: '+self.name)
		print('Converted:',self.converted)
		print('Data: ')
		self.data.printRawValues()

	def getSize(self,consider_indexes=False):
		return self.data.getSize(consider_indexes=consider_indexes)

	def getValues(self,degree=0):
		return self.data.getValues(degree=degree)

	def getIndexes(self,degree=0,int_index=False,get_all=False):
		return self.data.getIndexes(degree=degree,int_index=int_index,get_all=get_all)

	def regenerateIndexes(self):
		self.data.regenerateIndexes()
	
	def __init__(self,name):
		self.name=name
		self.data=None
		self.converted=False
		self.converted_params=tuple()
