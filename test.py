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
print('Values Main:',dataset.getValues(only_main_value=True))
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
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset_reverted=dataset.copy()
dataset_reverted.name+=' copy'
dataset_reverted.revertFromTemporalValues()
print('Len:',dataset_reverted.getSize())
dataset_reverted.printRawValues()
print('Indexes:',dataset_reverted.getIndexes())
print('Values:',dataset_reverted.getValues())
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
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Len:',dataset.getSize())
dataset.printRawValues()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
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
print()
print('Train data:')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays()
print('Neural Network Start Index:',start_index)
print('Neural Network X Shape:',dataset_x.shape)
print('Neural Network Y Shape:',dataset_y.shape)
start_index_part_2,dataset_x_p1,dataset_x_p2=Dataset.splitNeuralNetworkArray(dataset_x,.7)
print('Train Data Splitted Start Index:',start_index_part_2)
print('Neural Network X Splitted P1 Shape:',dataset_x_p1.shape)
print('Neural Network X Splitted P1 Shape:',dataset_x_p2.shape)
print()
# -------------------
print('-------------------')
dataset_reverted=dataset.copy()
dataset_reverted.name+=' copy'
dataset_reverted.revertFromTemporalValues()
print('Len:',dataset_reverted.getSize())
dataset_reverted.printRawValues()
print('Indexes:',dataset_reverted.getIndexes())
print('Values:',dataset_reverted.getValues())





# TODO do
# -------------------
print('-------------------')
dataset.setNeuralNetworkResultArray(start_index,np.array([]))
