#!/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import codecs
import joblib
import numpy as np
from datetime import datetime
import datetime as dt
import pandas as pd

class Utils:
	if os.name == 'nt':
		FILE_SEPARATOR='\\'
	else:
		FILE_SEPARATOR='/'
	DATE_FORMAT='%d/%m/%Y'
	DATETIME_FORMAT='%d/%m/%Y %H:%M:%S'
	FIRST_DATE='01/01/1970'

	def __init__(self):
		pass

	@staticmethod
	def createFolder(path):
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

	@staticmethod
	def checkIfPathExists(path):
		return os.path.exists(path)

	@staticmethod
	def joinPath(parent,child):
		parent=Utils.appendToStrIfDoesNotEndsWith(parent,Utils.FILE_SEPARATOR)
		return parent+child

	@staticmethod
	def appendToStrIfDoesNotEndsWith(base,suffix):
		if not base.endswith(suffix):
			return base+suffix
		return base

	@staticmethod
	def dateToTimestamp(string,include_time=False,date_format=DATE_FORMAT):
		if not include_time :
			return int(time.mktime(datetime.strptime(string,date_format).timetuple()))
		else:
			return int(time.mktime(datetime.strptime(string,Utils.DATETIME_FORMAT).timetuple()))

	@staticmethod
	def getNextNWorkDays(from_date, add_days):
		business_days_to_add = add_days
		current_date = from_date
		dates=[]
		while business_days_to_add > 0:
			current_date += dt.timedelta(days=1)
			weekday = current_date.weekday()
			if weekday >= 5: # sunday = 6
				continue
			business_days_to_add -= 1
			dates.append(current_date)
		return dates

	@staticmethod
	def getStrNextNWorkDays(from_date, add_days, date_format=DATE_FORMAT):
		from_date=datetime.strptime(from_date,date_format)
		dates=Utils.getNextNWorkDays(from_date,add_days)
		dates=[date.strftime(date_format) for date in dates]
		return dates

	@staticmethod
	def timestampToHumanReadable(timestamp,include_time=False,date_format=DATE_FORMAT):
		timestamp=int(timestamp)
		if not include_time :
			return datetime.fromtimestamp(timestamp).strftime(date_format)
		else:
			return datetime.fromtimestamp(timestamp).strftime(Utils.DATETIME_FORMAT)

	@staticmethod
	def extractNumbersFromDate(str_date,reverse=False):
		re_result=re.search(r'([0-9][0-9])\/([0-9][0-9])\/([0-9][0-9][0-9][0-9]|[0-9][0-9])', str_date)
		if not reverse:
			return re_result.group(1),re_result.group(2),re_result.group(3)
		else:
			return re_result.group(3),re_result.group(2),re_result.group(1)

	@staticmethod
	def truncateArraysOnCommonIndexes(array_of_data):
		array_of_indexes=[]
		for data in array_of_data:
			array_of_indexes.append(data.index.tolist())
		aligned_array_of_indexes=Utils.alignIndexesOnFirstCommonValue(array_of_indexes)
		for i in range(len(array_of_data)):
			array_of_data[i]=array_of_data[i][aligned_array_of_indexes[i][0]:]
		aligned_array_of_indexes=Utils.alignIndexesOnFirstCommonValue(array_of_indexes,reverse=True)
		for i in range(len(array_of_data)):
			array_of_data[i]=array_of_data[i][:aligned_array_of_indexes[i][0]]
		return array_of_data

	@staticmethod
	def alignIndexesOnFirstCommonValue(array_of_indexes,reverse=False):
		start=0
		last_common=None
		limit=len(array_of_indexes)-1
		while start<limit:
			f_array,s_array,common=Utils.alignAndCropTwoArrays(array_of_indexes[start],array_of_indexes[start+1],reverse=reverse)
			array_of_indexes[start]=f_array
			array_of_indexes[start+1]=s_array
			if common != last_common:
				if last_common is not None:
					start=-1
				last_common=common
			start+=1
		if reverse:
			[el.reverse() for el in array_of_indexes]
		return array_of_indexes

	@staticmethod
	def unwrapFoldedArray(array,use_last=False,use_mean=False,magic_offset=0):
		fold_size=len(array[0])
		array_size=len(array)
		unwraped_size=array_size+fold_size-1
		if use_mean:
			aux_sum_array_tuple=([0]*unwraped_size,[0]*unwraped_size)
			for i in range(magic_offset,array_size):
				for j in range(fold_size):
					aux_sum_array_tuple[0][i+j]+=array[i][j]
					aux_sum_array_tuple[1][i+j]+=1
			unwraped=[]
			for i in range(magic_offset,unwraped_size):
				unwraped.append(aux_sum_array_tuple[0][i]/aux_sum_array_tuple[1][i])
		else:
			position=0
			if use_last:
				#then use last
				position=fold_size-1
			unwraped=[array[i][position] for i in range(magic_offset,array_size)]
			for i in range(1,fold_size):
				unwraped.append(array[array_size-1][i])
		return unwraped

	@staticmethod
	def alignAndCropTwoArrays(first,second,reverse=False):
		sorted_second=second
		sorted_second.sort()
		used_first=first.copy()
		if reverse:
			used_first.reverse()
		common=None
		for el in used_first:
			ind=Utils.binarySearch(sorted_second,el)
			if ind is not None:
				common=el
				break
		if common is None:
			raise Exception('No common element between arrays')
		else:
			if reverse:
				return first[:first.index(common)+1], second[:second.index(common)+1], common
			else:
				return first[first.index(common):], second[second.index(common):], common

	@staticmethod
	def extractElementsFromInsideLists(y,last_instead_of_all_but_last=False):
		new_y=[]
		for x in y:
			if isinstance(x, (list,pd.core.series.Series,np.ndarray)):
				if last_instead_of_all_but_last:
					new_y.append(x[-1])
				else:
					new_y.append(x[:-1])
			else:
				new_y.append(x)
		return new_y

	@staticmethod
	def computeArrayIntervals(array):
		diff=[]
		for i in range(len(array)-1):
			if str(type(array[i]))=="<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
				diff.append(float((array[i+1]-array[i]) / np.timedelta64(1, 'h')))
			else:
				diff.append(array[i+1]-array[i])
		return max(set(diff), key=diff.count)

	@staticmethod
	def filenameFromPath(path,get_extension=False):
		if get_extension :
			re_result=re.search(r'.*\/(.*\..+)', path)
			return re_result.group(1) if re_result is not None else path
		else:
			re_result=re.search(r'.*\/(.*)\..+', path)
			return re_result.group(1) if re_result is not None else path

	@staticmethod
	def saveObj(obj,path):
		joblib.dump(obj, path)

	@staticmethod
	def loadObj(path):
		return joblib.load(path)

	@staticmethod
	def saveJson(json_dict,path,sort_keys=True,indent=True):
		with codecs.open(path, 'w', encoding='utf-8') as file:
			if indent:
				json.dump(json_dict, file, separators=(',', ':'), sort_keys=sort_keys, indent=4)
			else:
				json.dump(json_dict, file, separators=(',', ':'), sort_keys=sort_keys)
		
	@staticmethod
	def loadJson(path):
		with codecs.open(path, 'r', encoding='utf-8') as file:
			n=json.loads(file.read())
		return n

	@staticmethod
	def binarySearch(lis,el): # list must be sorted
		low=0
		high=len(lis)-1
		ret=None
		while low<=high:
			mid=(low+high)//2
			if el<lis[mid]:
				high=mid-1
			elif el>lis[mid]:
				low=mid+1
			else:
				ret=mid
				break
		return ret

	@staticmethod
	def printDict(dictionary,name=None):
		start=''
		if name is not None:
			print('{}:'.format(name))
			start='\t'
		for key,value in dictionary.items():
			print('{}{}: {}'.format(start,key,value))

	@staticmethod
	def estimateNextElements(array,n):
		diff=array[-1]-array[-2]
		for i in range(n):
			array.append(array[-1]+diff)
		return array

	@staticmethod
	def changeDateFormat(date,in_format,out_format):
		return dt.datetime.strptime(date, in_format).strftime(out_format)
