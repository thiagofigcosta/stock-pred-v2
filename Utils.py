#!/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import shutil
import math
import codecs
import joblib
import gzip
import zlib
import uuid
import base64
import random as rd
from datetime import datetime
from numpy.random import Generator, MT19937
import datetime as dt

class Utils:
	if os.name == 'nt':
		FILE_SEPARATOR='\\'
	else:
		FILE_SEPARATOR='/'
	DATE_FORMAT='%d/%m/%Y'
	DATETIME_FORMAT='%d/%m/%Y %H:%M:%S'
	FIRST_DATE='01/01/1970'
	RNG=Generator(MT19937(int(time.time()*rd.random())))

	def __init__(self):
		pass

	@staticmethod
	def shuffle(list_to_shuffle):
		list_to_shuffle=list_to_shuffle.copy()
		Utils.RNG.shuffle(list_to_shuffle)
		return list_to_shuffle

	@staticmethod
	def random():
		return Utils.RNG.random()

	@staticmethod
	def randomInt(min_inclusive,max_inclusive,size=1):
		out=Utils.RNG.integers(low=min_inclusive, high=max_inclusive+1, size=size)
		if size == 1:
			out=out[0]
		return out

	@staticmethod
	def randomFloat(min_value,max_value,size=1):
		out=[]
		for _ in range(size):
			out.append(min_value + (Utils.random() * (max_value - min_value)))
		if size == 1:
			out=out[0]
		return out

	@staticmethod
	def randomUUID():
		return uuid.uuid4().hex

	@staticmethod
	def getPythonVersion(getTuple=False):
		version=sys.version_info
		version_tuple=(version.major,version.minor,version.micro)
		if getTuple:
			return version.major,version.minor,version.micro
		else:
			return '.'.join([str(el) for el in version_tuple])
	
	@staticmethod
	def getPythonExecName():
		version=Utils.getPythonVersion(getTuple=True)
		full_name='python{}.{}'.format(version[0],version[1])
		short_name='python{}'.format(version[0])
		default_name='python'
		if shutil.which(full_name) is not None:
			return full_name
		if shutil.which(short_name) is not None:
			return short_name
		return default_name

	@staticmethod
	def moveFile(src_path,dst_path):
		os.replace(src_path, dst_path)

	@staticmethod
	def createFolder(path):
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

	@staticmethod
	def getFolderPathsThatMatchesPattern(folder,pattern):
		paths=[]
		if os.path.exists(folder):
			for filename in os.listdir(folder):
				if re.match(pattern,filename):
					file_path = Utils.joinPath(folder,filename)
					paths.append(file_path)
		return paths

	@staticmethod
	def deleteFolderContents(folder,ignore=[]):
		if os.path.exists(folder):
			for filename in os.listdir(folder):
				if filename not in ignore:
					file_path = Utils.joinPath(folder,filename)
					Utils.deletePath(file_path)

	@staticmethod
	def deletePath(path):
		if os.path.isdir(path):
			Utils.deleteFolder(path)
		elif os.path.isfile(path) or os.path.islink(path):
			Utils.deleteFile(path)
		else:
			print('File {} is a special file.'.format(path))


	@staticmethod
	def deleteFile(path):
		if os.path.exists(path):
			os.remove(path)
		else:
			print('The file {} does not exist.'.format(path))


	@staticmethod
	def deleteFolder(path):
		if os.path.exists(path):
			shutil.rmtree(path)
		else:
			print('The folder {} does not exist.'.format(path))

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
	def timestampToStrDateTime(timestamp,include_time=False,date_format=DATE_FORMAT):
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
	def extractARegexGroup(string,pattern,group=1):
		re_result=re.search(pattern, string)
		return re_result.group(group)

	@staticmethod
	def filenameFromPath(path,get_extension=False):
		if get_extension :
			re_result=re.search('.*\\'+Utils.FILE_SEPARATOR+r'(.*\..+)', path)
			return re_result.group(1) if re_result is not None else path
		else:
			re_result=re.search('.*\\'+Utils.FILE_SEPARATOR+r'(.*)\..+', path)
			return re_result.group(1) if re_result is not None else path

	@staticmethod
	def copyFile(src_path,dst_path):
		shutil.copy(src_path, dst_path)

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
	def timestampByExtensive(timestamp,seconds=True):
		if seconds:
			timestamp_ms=timestamp*1000
		else:
			timestamp_ms=timestamp
		timestamp_ms=int(timestamp_ms)
		D=int(timestamp_ms/1000/60/60/24)
		H=int(timestamp_ms/1000/60/60%24)
		M=int(timestamp_ms/1000/60%60)
		S=int(timestamp_ms/1000%60)
		MS=int(timestamp_ms%1000)
		out='' if timestamp_ms > 0 else 'FINISHED'
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

	@staticmethod
	def calcMovingAverage(input_arr,window):
		i = 0
		moving_averages = []
		while i < len(input_arr) - window + 1:
			this_window = input_arr[i : i + window]
			window_average = sum(this_window) / window
			moving_averages.append(window_average)
			i += 1
		return moving_averages

	@staticmethod
	def calcDiffUp(input_arr):
		up = []
		for i in range(1,len(input_arr)):
			up.append(1 if input_arr[i]>input_arr[i-1] else 0)
		return up

	@staticmethod
	def createFolderIfNotExists(path):
		if not os.path.exists(path):
			os.makedirs(path)

	@staticmethod
	def getEnumBorder(enum,max_instead_of_min=False):
		if max_instead_of_min:
			return enum(list(enum._member_map_.items())[-1][1]).value
		else:
			return enum(list(enum._member_map_.items())[0][1]).value


	@staticmethod
	def objToJsonStr(obj,compress=False,b64=False):
		if obj is None:
			return None
		data=json.dumps(obj,cls=Utils.NumpyJsonEncoder)
		if compress:
			compressed=zlib.compress(data.encode('utf-8'))
			if b64:
				return base64.b64encode(compressed).decode()
			else:
				return compressed.decode('iso-8859-1')
		else:
			if b64:
				return base64.b64encode(data.encode('utf-8')).decode()
			else:
				return data 

	@staticmethod
	def jsonStrToObj(data_str,compress=False,b64=False):
		if data_str is None:
			return None
		if compress:
			if b64:
				data_to_load=zlib.decompress(base64.b64decode(data_str.encode('utf-8'))).decode('utf-8')
			else:
				data_to_load=zlib.decompress(data_str.encode('iso-8859-1')).decode('utf-8')
		else:
			if b64:
				data_to_load=base64.b64decode(data_str.encode('utf-8')).decode('utf-8')
			else:
				data_to_load=data_str
		if data_to_load is not None:
			data=json.loads(data_to_load,object_hook=Utils.NumpyJsonEncoder.dec)
		return data
	
	@staticmethod
	def base64ToBase65(base64,char_65='*'):
		if base64 is None:
			return None
		base65=''
		mark_char='&'
		last_char=mark_char
		min_lenght=4
		count=1
		for c in base64+last_char:
			if (last_char==mark_char):
				last_char=c
				count=1
			else:
				if (last_char!=c):
					if (count>min_lenght):
						base65+='{}{}{}{}'.format(last_char,char_65,count,char_65)
					else:
						base65+=last_char*count
					count=0
					last_char=c
				count+=1
		return base65

	@staticmethod
	def base65ToBase64(base65,char_65='*'):
		if base65 is None:
			return None
		if char_65 not in base65: # fast
			return base65
		base64=''
		mark_char='&'
		last_char=mark_char
		count=None
		for c in base65:
			if (c==char_65):
				if (count==None):
					count=-1
				else:
					base64+=last_char*(count-1)
					count=None
			if (count==None):
				if (c!=char_65):
					base64+=c
					last_char=c
			else:
				if (c!=char_65):
					if (count==-1):
						count=int(c)
					else:
						count*=10; 
						count+=int(c)
		return base64

	class NumpyJsonEncoder(json.JSONEncoder):
		def default(self, obj):
			if isinstance(obj, np.ndarray):
				data_b64 = base64.b64encode(obj.data).decode()
				return dict(__ndarray__=data_b64,
							dtype=str(obj.dtype),
							shape=obj.shape)
			return json.JSONEncoder.default(self, obj)

		def dec(dct):
			if isinstance(dct, dict) and '__ndarray__' in dct:
				data = base64.b64decode(dct['__ndarray__'].encode())
				return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
			return dct


	@staticmethod
	def logReturn(input_arr):
		l_return = []
		l = []
		for el in input_arr:
			l.append(math.log(el))
		for i in range(1,len(l)):
			l_return.append(l[i]-l[i-1])
		return l_return


	@staticmethod
	def calcExpMovingAverage(input_arr,window):
		ema = []
		j = 1
		# get window sma first and calculate the next window period ema
		sma = sum(input_arr[:window]) / window 
		multiplier = 2 / float(1 + window)
		ema.append(sma)
		# EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
		ema.append(( (input_arr[window] - sma) * multiplier) + sma)
		for i in input_arr[window+1:]:
			tmp = ( (i - ema[j]) * multiplier) + ema[j]
			j = j + 1
			ema.append(tmp)
		return ema
			