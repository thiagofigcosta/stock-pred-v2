#!/bin/python3
# -*- coding: utf-8 -*-

import re
import json
import time
import codecs
import urllib
import urllib.request
from Utils import Utils

class Crawler:
	DATASET_PATH='datasets/'

	def __init__(self):
		self.yahoo_api='https://query1.finance.yahoo.com/v7/finance'
		self.source='yahoo'
		Utils.createFolder(Crawler.DATASET_PATH)

	def filterNullLines(self,content):
		lines=content.split('\n')
		content=''
		for line in lines:
			if not re.match(r'((.*null|nan),.*|.*,(null|nan).*)',line, flags=re.IGNORECASE):
				content+=line+'\n'
		return content[:-1]

	def getDatasetPath(self,filename):
		path=filename
		if Crawler.DATASET_PATH+Utils.FILE_SEPARATOR not in path:
			path=Utils.joinPath(Crawler.DATASET_PATH,filename)
		return path

	def yahooCustomIntervalJsonToCSV(self,json_str):
		parsed_json=json.loads(json_str)
		timestamps=parsed_json['spark']['result'][0]['response'][0]['timestamp']
		close_values=parsed_json['spark']['result'][0]['response'][0]['indicators']['quote'][0]['close']
		if len(timestamps)!=len(close_values):
			raise Exception('Stock timestamp array({}) with different size from stock values array({})'.format(len(timestamps),len(close_values)))
		CSV='timestamp,Date,Close\n'
		for i in range(len(timestamps)):
			CSV+='{},{},{}\n'.format(timestamps[i],Utils.timestampToHumanReadable(timestamps[i],True),close_values[i])
		return CSV[:-1]

	def downloadStockDailyData(self,stock_name,filename,start_date=Utils.timestampToHumanReadable(0),end_date=Utils.timestampToHumanReadable(time.time())):
		if self.source=='yahoo':
			download_url='{}/download/{}?period1={}&period2={}&interval=1d&events=history&includeAdjustedClose=true'.format(self.yahoo_api,stock_name,Utils.dateToTimestamp(start_date),Utils.dateToTimestamp(end_date))
			print(download_url)
			with urllib.request.urlopen(download_url) as response:
				if response.code == 200:
					content=response.read().decode('utf-8')
					path=self.getDatasetPath(filename)
					content=self.filterNullLines(content)
					with codecs.open(path, "w", "utf-8") as file:
						file.write(content)
				else:
					raise Exception('Response code {}'.format(response.code))
		else:
			raise Exception('Unknown source {}'.format(self.source))

	def downloadStockDataCustomInterval(self,stock_name,filename,data_range='730d',start_timestamp='',end_timestamp='',interval='60m'):
		if self.source=='yahoo':
			# data_range maximum value is 730d
			# data_range minimum value is 1m
			if not start_timestamp or not end_timestamp:
				download_url='{}/spark?symbols={}&range={}&interval={}&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance'.format(self.yahoo_api,stock_name,data_range,interval)
			else:
				download_url='{}/spark?symbols={}&period1={}&period2={}&interval={}&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance'.format(self.yahoo_api,stock_name,start_timestamp,end_timestamp,interval)
			print(download_url)
			with urllib.request.urlopen(download_url) as response:
				if response.code == 200:
					content=response.read().decode('utf-8')
					path=self.getDatasetPath(filename)
					content=self.yahooCustomIntervalJsonToCSV(content)
					content=self.filterNullLines(content)
					with codecs.open(path, "w", "utf-8") as file:
						file.write(content)
				else:
					raise Exception('Response code {} - {}'.format(response.code))
		else:
			raise Exception('Unknown source {}'.format(self.source))
			