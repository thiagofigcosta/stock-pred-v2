#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils
from Dataset import Dataset


stock_value=[1,2,3,4,5,6,7,8,9,10]
features_values=[[10],[20],[30],[40],[50],[60],[70],[80],[90],100]
dates=Utils.getStrNextNWorkDays('17/06/2021',len(stock_value))


# -------------------
print('-------------------')
dataset=Dataset(name='OriginalGE')
dataset.addCompany(stock_value)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Values:',dataset.getValues())
# -------------------
print('-------------------')
dataset=Dataset(name='OriginalGE')
dataset.addCompany(stock_value,dates)
print('Len:',dataset.getSize())
dataset.printRawValues()
# -------------------
print('-------------------')
dataset=Dataset(name='OriginalGE')
dataset.addCompany(stock_value,dates,features_values)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
# -------------------
print('-------------------')
print('Train data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays()
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)
print()
print('Full data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(include_test_data=True)
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)
print()
print('To predict data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(only_test_data=True)
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)
# -------------------
print('-------------------')
stock_value_2=[-el for el in stock_value]
dataset=Dataset(name='OriginalGE and OriginalGE*-1')
dataset.addCompany(stock_value,dates,features_values)
dataset.addCompany(stock_value_2,dates,features_values)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
# -------------------
print('-------------------')
print('Train data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays()
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)
print()
print('Full data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(include_test_data=True)
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)
print()
print('To predict data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(only_test_data=True)
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)


# -------------------
print('-------------------')
dataset.revertFromTemporalValues()
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())