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
	def printDict(dictionary,name=None,tabs=0):
		start=''
		if name is not None:
			print('{}{}:'.format('\t'*tabs,name))
			start='\t'
		for key,value in dictionary.items():
			print('{}{}{}: {}'.format('\t'*tabs,start,key,value))

	@staticmethod
	def changeDateFormat(date,in_format,out_format):
		return dt.datetime.strptime(date, in_format).strftime(out_format)

	@staticmethod
	def removeStrPrefix(text, prefix):
		if text.startswith(prefix):
			return text[len(prefix):]
		return text

	@staticmethod
	def assertDateFormat(date_str, date_format=DATE_FORMAT):
		try:
			dt.datetime.strptime(date_str, date_format)
		except ValueError:
			raise Exception('The date ({}) must obey the {} format and must be a valid date'.format(date_str,date_format))

	@staticmethod
	def msToHumanReadable(timestamp):
		timestamp=int(timestamp)
		D=int(timestamp/1000/60/60/24)
		H=int(timestamp/1000/60/60%24)
		M=int(timestamp/1000/60%60)
		S=int(timestamp/1000%60)
		MS=int(timestamp%1000)
		out='' if timestamp > 0 else 'FINISHED'
		if D > 0:
			out+='{} days '.format(D)
		if D > 0 and MS == 0 and S == 0 and M == 0 and H > 0:
			out+='and '
		if H > 0:
			out+='{} hours '.format(H)
		if (D > 0 or H > 0) and MS == 0 and S == 0 and M > 0:
			out+='and '
		if M > 0:
			out+='{} minutes '.format(M)
		if (D > 0 or H > 0 or M > 0) and MS == 0 and S > 0:
			out+='and '
		if S > 0:
			out+='{} seconds '.format(S)
		if (D > 0 or H > 0 or M > 0 or S > 0) and MS > 0:
			out+='and '
		if MS > 0:
			out+='{} milliseconds '.format(MS)
		return out
