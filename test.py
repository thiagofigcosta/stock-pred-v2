#!/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
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
dataset.print()
print('Values:',dataset.getValues())
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset=Dataset(name='OriginalGE')
dataset.addCompany(stock_value,dates)
print('Len:',dataset.getSize())
dataset.print()
# -------------------
print('-------------------')
dataset=Dataset(name='OriginalGE')
dataset.addCompany(stock_value,dates,features_values)
print('Len:',dataset.getSize())
dataset.print()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Len:',dataset.getSize())
dataset.print()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset_reverted=dataset.copy()
dataset_reverted.name+=' copy'
dataset_reverted.revertFromTemporalValues()
print('Len:',dataset_reverted.getSize())
dataset_reverted.print()
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
dataset.print()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
print('Values Main:',dataset.getValues(only_main_value=True))
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Len:',dataset.getSize())
dataset.print()
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
start_index_to_pred,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(only_test_data=True)
print('Neural Network Start Index:',start_index_to_pred)
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
dataset_reverted.print()
print('Indexes:',dataset_reverted.getIndexes())
print('Values:',dataset_reverted.getValues())
# -------------------
print('-------------------')
correct_predictions=np.array([[[10, -10], [11, -11]],[[11, -11], [12, -12]]])
print('Correct preds shape',correct_predictions.shape)
dataset.setNeuralNetworkResultArray(start_index_to_pred,correct_predictions)
print('Len:',dataset.getSize())
dataset.print()
print('Indexes:',dataset.getIndexes())
print('Values:',dataset.getValues())
print('Indexes degree 1:',dataset.getIndexes(degree=1))
print('Values degree 1:',dataset.getValues(degree=1))
print('Indexes degree 2:',dataset.getIndexes(degree=2))
print('Values degree 2:',dataset.getValues(degree=2))
# -------------------
print('-------------------')
dataset_reverted=dataset.copy()
dataset_reverted.name+=' copy'
dataset_reverted.revertFromTemporalValues()
print('Len:',dataset_reverted.getSize())
dataset_reverted.print()
print('Indexes:',dataset_reverted.getIndexes())
print('Values:',dataset_reverted.getValues())
print()
indexes,preds=dataset_reverted.getDatesAndPredictions()
print('Pred Indexes:',indexes)
print('Pred Values:',preds)
print('\t*Inner dimmension = companies | After inner dimmension = multiple predictions')
# -------------------
print('-------------------')
dataset=Dataset(name='OriginalGE')
dataset.addCompany(stock_value,dates,features_values)
print('Values:',dataset.getValues())
print('Max:',dataset.getAbsMaxes())
# -------------------
print('-------------------')
dataset.convertToTemporalValues(3,2)
print('Max:',dataset.getAbsMaxes())
# -------------------
print('-------------------')
start_index,dataset_x,dataset_y=dataset.getNeuralNetworkArrays(include_test_data=True,normalization=Dataset.Normalization.NORMALIZE)
print('Neural Network X 0:',dataset_x[0].tolist())
print('Neural Network Y 0:',dataset_y[0].tolist())
# -------------------
print('-------------------')
correct_predictions=np.array([[[1, -1], [1.1, -1.1]],[[1.1, -1.1], [1.2, -1.2]]])
print('Correct preds shape',correct_predictions.shape)
dataset.setNeuralNetworkResultArray(start_index_to_pred,correct_predictions)
print('Len:',dataset.getSize())
dataset.print()