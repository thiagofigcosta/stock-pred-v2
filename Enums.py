#!/bin/python
# -*- coding: utf-8 -*-

from enum import Enum


class Features(Enum):
	CLOSE = 0
	OPEN = 1
	HIGH = 2
	LOW = 3
	ADJ_CLOSE = 4
	VOLUME = 5
	FAST_MOVING_AVG = 6
	SLOW_MOVING_AVG = 7
	LOG_RETURN = 8
	UP = 9
	OC = 10
	OH = 11
	OL = 12
	CH = 13
	CL = 14
	LH = 15
	FAST_EXP_MOVING_AVG = 16
	SLOW_EXP_MOVING_AVG = 17

	def toDatasetName(self):
		if self == Features.CLOSE:
			return 'Close'
		elif self == Features.OPEN:
			return 'Open'
		elif self == Features.HIGH:
			return 'High'
		elif self == Features.LOW:
			return 'Low'
		elif self == Features.ADJ_CLOSE:
			return 'Adj Close'
		elif self == Features.VOLUME:
			return 'Volume'
		elif self == Features.FAST_MOVING_AVG:
			return 'fast_moving_avg'
		elif self == Features.SLOW_MOVING_AVG:
			return 'slow_moving_avg'	
		elif self == Features.UP:
			return 'up'	
		elif self == Features.LOG_RETURN:
			return 'log_return'	
		elif self == Features.OC:
			return 'oc'	
		elif self == Features.OH:
			return 'oh'	
		elif self == Features.OL:
			return 'ol'	
		elif self == Features.CH:
			return 'ch'	
		elif self == Features.CL:
			return 'cl'	
		elif self == Features.LH:
			return 'lh'	
		elif self == Features.FAST_EXP_MOVING_AVG:
			return 'fast_exp_moving_avg'	
		elif self == Features.SLOW_EXP_MOVING_AVG:
			return 'slow_exp_moving_avg'	
		return None

class Metric(Enum):
	RAW_LOSS = 0
	ACCURACY = 1
	MSE=2
	MAE=3
	COSINE_SIM=4
	F1 = 5
	PRECISION = 6
	RECALL = 7
	RMSE = 8
	R2 = 9

	def toKerasName(self):
		if self == Metric.RAW_LOSS:
			return 'loss'
		elif self == Metric.F1:
			return 'f1_score'
		elif self == Metric.RECALL:
			return 'recall'
		elif self == Metric.ACCURACY:
			return 'accuracy'
		elif self == Metric.PRECISION:
			return 'precision'
		elif self == Metric.MSE:
			return 'mean_squared_error'
		elif self == Metric.MAE:
			return 'mean_absolute_error'
		elif self == Metric.COSINE_SIM:
			return 'cosine_similarity'		 
		elif self == Metric.RMSE:
			return 'root_mean_squared_error'
		elif self == Metric.R2:
			return 'R2'
		return None
		

class GeneticAlgorithmType(Enum):
	ENHANCED = 0
	STANDARD = 1


class NodeType(Enum):
	RELU = 0
	SIGMOID = 1
	TANH = 2
	EXPONENTIAL = 3
	LINEAR = 4
	HARD_SIGMOID = 5
	SOFTMAX = 6
	SOFTPLUS = 7
	SOFTSIGN = 8
	SELU = 9
	ELU = 10

	def toKerasName(self):
		if self == NodeType.RELU:
			return 'relu'
		elif self == NodeType.SOFTMAX:
			return 'softmax'
		elif self == NodeType.SIGMOID:
			return 'sigmoid'
		elif self == NodeType.HARD_SIGMOID:
			return 'hard_sigmoid'
		elif self == NodeType.TANH:
			return 'tanh'
		elif self == NodeType.SOFTPLUS:
			return 'softplus'
		elif self == NodeType.SOFTSIGN:
			return 'softsign'
		elif self == NodeType.SELU:
			return 'selu'
		elif self == NodeType.ELU:
			return 'elu'
		elif self == NodeType.EXPONENTIAL:
			return 'exponential'
		elif self == NodeType.LINEAR:
			return 'linear'
		return None

class GeneticRankType(Enum):
	RELATIVE = 0
	ABSOLUTE = 1
	INCREMENTAL = 1

class Loss(Enum):
	BINARY_CROSSENTROPY = 0
	CATEGORICAL_CROSSENTROPY = 1
	MEAN_SQUARED_ERROR = 2
	MEAN_ABSOLUTE_ERROR = 3
	ROOT_MEAN_SQUARED_ERROR = 4

	def toKerasName(self):
		if self == Loss.BINARY_CROSSENTROPY:
			return 'binary_crossentropy'
		elif self == Loss.CATEGORICAL_CROSSENTROPY:
			return 'categorical_crossentropy'
		elif self == Loss.MEAN_SQUARED_ERROR:
			return 'mean_squared_error'
		elif self == Loss.MEAN_ABSOLUTE_ERROR:
			return 'mean_absolute_error'
		elif self == Loss.ROOT_MEAN_SQUARED_ERROR:
			return 'root_mean_squared_error'
		return None

class Optimizers(Enum):
	SGD = 0
	ADAM = 1
	RMSPROP = 2

	def toKerasName(self):
		if self == Optimizers.SGD:
			return 'sgd'
		elif self == Optimizers.ADAM:
			return 'adam'
		elif self == Optimizers.RMSPROP:
			return 'rmsprop'
		return None