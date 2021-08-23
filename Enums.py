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
	UP = 8

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
		return None
		

class GeneticAlgorithmType(Enum):
	ENHANCED = 0
	STANDARD = 1


class NodeType(Enum):
	RELU = 0
	SIGMOID = 1
	TANH = 2
	HARD_SIGMOID = 3
	SOFTMAX = 4
	SOFTPLUS = 5
	SOFTSIGN = 6
	SELU = 7
	ELU = 8
	EXPONENTIAL = 9
	LINEAR = 10

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

	def toKerasName(self):
		if self == Loss.BINARY_CROSSENTROPY:
			return 'binary_crossentropy'
		elif self == Loss.CATEGORICAL_CROSSENTROPY:
			return 'categorical_crossentropy'
		elif self == Loss.MEAN_SQUARED_ERROR:
			return 'mean_squared_error'
		elif self == Loss.MEAN_ABSOLUTE_ERROR:
			return 'mean_absolute_error'
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