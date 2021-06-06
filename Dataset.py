#!/bin/python3
# -*- coding: utf-8 -*-

from Utils import Utils

class Dataset:
	class Entry:
		def __init__(self,index=None,date_index=None,value=None,backward_values=[],forward_values=None):
			self.index=index # of position 0 backward value
			self.date_index=date_index # of position 0 backward value
			if value is not None:
				backward_values=[tuple([value])]
				forward_values=None
			self.backward_values=backward_values # array of tuple of values
			self.forward_values=forward_values # array of tuple of values
			self.next=None # next entry

		def mergeWith(self,dataset):
			i=0
			size_a=self.getSize(consider_null_forwards=True)
			size_b=dataset.getSize(consider_null_forwards=True)
			cur_a=self
			cur_b=dataset
			while i<size_a and i<size_b:
				if cur_a.backward_values is None:
					cur_a.backward_values=cur_b.backward_values
				elif cur_b.backward_values is None:
					cur_a.backward_values=cur_a.backward_values
				else:
					largest=cur_a.backward_values
					if len(cur_a.backward_values)<len(cur_b.backward_values):
						largest=cur_b.backward_values
					for j in range(min(len(cur_a.backward_values),len(cur_b.backward_values))):
						largest[j]=cur_a.backward_values[j]+cur_b.backward_values[j]
					cur_a.backward_values=largest

				if cur_a.forward_values is None:
					cur_a.forward_values=cur_b.forward_values
				elif cur_b.forward_values is None:
					cur_a.forward_values=cur_a.forward_values
				else:
					largest=cur_a.forward_values
					if len(cur_a.forward_values)<len(cur_b.forward_values):
						largest=cur_b.forward_values
					for j in range(min(len(cur_a.forward_values),len(cur_b.forward_values))):
						largest[j]=cur_a.forward_values[j]+cur_b.forward_values[j]
					cur_a.forward_values=largest
				i+=1
				if cur_a.next is None or cur_b.next  is None:
					break
				cur_a=cur_a.next
				cur_b=cur_b.next

		def getNeuralNetworkArray(self):
			pass # TODO return X and Y arrays for LSTM network

			# como é hoje:
				# X_train (3418, 20, 1)
				# Y_train (3418, 7)
				# X_val (855, 20, 1)
				# Y_val (855, 7)
				# X_test (1069, 20, 1)
				# Y_test (1062, 7)


			# como vou fazer:
				# X_train (3418, 20, 1, 1) # (amostras, for/backwards, empresa, features)
				# Y_train (3418, 7, 1) # (amostras, for/backwards, empresa)
				# X_val (855, 20, 1, 1) # (amostras, for/backwards, empresa, features)
				# Y_val (855, 7, 1) # (amostras, for/backwards, empresa)
				# X_test (1069, 20, 1, 1)# (amostras, for/backwards, empresa, features)
				# Y_test (1062, 7, 1) # (amostras, for/backwards, empresa)

		@staticmethod
		def datasetFromNeuralNetworkArray(X,Y):
			pass # TODO

		def generateBackAndForward(self,back_samples,forward_samples):
			values=self.getValueArray()
			size=self.getSize(consider_null_forwards=True)
			new_size=size-back_samples+1
			new_useful_size=size-back_samples+1-forward_samples
			new_full_size=new_size+forward_samples
			last=self.getNthPointer(size-1)
			if last.date_index is not None:
				next_dates=Utils.getStrNextNWorkDays(last.date_index,forward_samples)
			else:
				next_dates=[None]*forward_samples
			dates=self.getDateArray(consider_null_forwards=True)
			new_first=None
			cur=new_first
			for i in range(back_samples-1):
				new_el=Dataset.Entry(index=i,date_index=dates[i],backward_values=[values[i]])
				if new_first is None:
					new_first=new_el
					cur=new_first
				else:
					cur.next=new_el
					cur=cur.next
			dates=dates[back_samples-1:]+next_dates
			starting_at=self.getNthPointer(back_samples-1).index
			if starting_at is None:
				starting_at=0
			indexes=list(range(starting_at,new_full_size+starting_at,1))
			for i in range(new_full_size):
				if i<new_size:
					back_values=[]
					for j in range(back_samples):
						back_values.append(values[back_samples+i-j-1]+tuple()) # copy of tuple
					for_values=[]
					if i<new_useful_size:
						for j in range(forward_samples):
							for_values.append(values[back_samples+i+j]+tuple()) # copy of tuple
					new_el=Dataset.Entry(index=indexes[i],date_index=dates[i],backward_values=back_values,forward_values=for_values)
				else:
					new_el=Dataset.Entry(index=indexes[i],date_index=dates[i],backward_values=None,forward_values=None)

				if new_first is None:
					new_first=new_el
					cur=new_first
				else:
					cur.next=new_el
					cur=cur.next
			return new_first

		def getValueArray(self,degree=0):
			val_arr=[]
			cur=self
			while True:
				if cur.backward_values is None:
					break
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
			return val_arr

		def getDateArray(self,degree=0,consider_null_forwards=False):
			date_arr=[]
			cur=self
			extra=0
			while True:
				date_arr.append(cur.date_index)
				if cur.next is None:
					break
				cur=cur.next
				if cur.forward_values is None and not consider_null_forwards:
					if degree==0:
						break
					else:
						extra+=1
			if degree>0:
				date_arr=date_arr[degree:]
				if extra>0:
					to_remove=extra-degree
					if to_remove>0:
						date_arr=date_arr[:-to_remove]
			return date_arr

		def getIndexArray(self,degree=0,consider_null_forwards=True):
			index_arr=[]
			cur=self
			extra=0
			while True:
				index_arr.append(cur.index)
				if cur.next is None:
					break
				cur=cur.next
				if cur.forward_values is None and not consider_null_forwards:
					if degree==0:
						break
					else:
						extra+=1
			if degree>0:
				index_arr=index_arr[degree:]
				if extra>0:
					to_remove=extra-degree
					if to_remove>0:
						index_arr=index_arr[:-to_remove]
			return index_arr

		def getSize(self,consider_null_forwards=False):
			size=0
			cur=self
			while True:
				size+=1
				if cur.next is None or (cur.forward_values is None and not consider_null_forwards):
					break
				cur=cur.next
			return size

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
			i=0
			size=self.getSize()
			cur=self
			while i<size:
				if cur.index==index:
					return cur
				i+=1
				if cur.next is None:
					break
				cur=cur.next
			return None

		def getPointerAtDate(self,date):
			i=0
			size=self.getSize()
			cur=self
			while i<size:
				if cur.date_index==date:
					return cur
				i+=1
				if cur.next is None:
					break
				cur=cur.next
			return None

		def regenerateIndexes(self):
			index=0
			cur=self
			while True:
				cur.index=index
				index+=1
				if cur.next is None:
					break
				cur=cur.next

		def printDatesAndValues(self,degree=0,consider_null_forwards=False):
			dates=self.getDateArray(degree=degree,consider_null_forwards=True)
			values=self.getValueArray(degree=degree)
			size=self.getSize(consider_null_forwards=consider_null_forwards)

			for i in range(size):
				print('{}-{}'.format(dates[i],values[i]))

		def printRawDatesAndValues(self):
			i=0
			size=self.getSize(consider_null_forwards=True)
			cur=self
			while i<size:
				print('{}: {}->{}'.format(cur.date_index,cur.backward_values,cur.forward_values))
				i+=1
				if cur.next is None:
					break
				cur=cur.next
	
	def __init__(self,name,data,scaler):
		self.name=name
		self.data=data
		self.scaler=scaler