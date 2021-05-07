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
		aligned_array_of_indexes=alignIndexesOnFirstCommonValue(array_of_indexes)
		for i in range(len(array_of_data)):
			array_of_data[i]=array_of_data[i][aligned_array_of_indexes[i][0]:]
		aligned_array_of_indexes=alignIndexesOnFirstCommonValue(array_of_indexes,reverse=True)
		for i in range(len(array_of_data)):
			array_of_data[i]=array_of_data[i][:aligned_array_of_indexes[i][0]]
		return array_of_data

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
