#!/bin/python3
# -*- coding: utf-8 -*-

import math
import random as rd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

class Actuator:
	
	INITIAL_INVESTIMENT=22000

	@staticmethod
	def sum(array):
		summation=0
		for el in array:
			summation+=el
		return summation

	@staticmethod
	def mean(array):
		return Actuator.sum(array)/len(array)

	@staticmethod
	def getStockReturn(stock_array):
		shifted_stock_array=stock_array[1:]+[0]
		stock_delta=((np.array(shifted_stock_array)-np.array(stock_array))[:-1]).tolist()#+[None]
		return stock_delta

	@staticmethod
	def analyzeStrategiesAndClassMetrics(stock_real_array,stock_pred_array):
		real_stock_delta=Actuator.getStockReturn(stock_real_array)
		pred_stock_delta=Actuator.getStockReturn(stock_pred_array)
		
		real_movement_encoded=[ 1 if r>0 else 0 for r in real_stock_delta]
		pred_movement_encoded=[ 1 if pred_stock_delta[i]>0 else 0 for i in range(len(real_stock_delta))]
		real_movement=['Up' if r>0 else 'Down' for r in real_stock_delta]
		pred_movement=['Up' if r>0 else 'Down' for r in pred_stock_delta]
		correct_movement=[ 1 if real_movement[i]==pred_movement[i] else 0 for i in range(len(real_stock_delta))]
		accuracy=Actuator.mean(correct_movement)
		
		swing_return=[real_stock_delta[i] if pred_movement[i] == 'Up' else 0 for i in range(len(real_stock_delta))]
		swing_return=Actuator.sum(swing_return) # se a ação subir investe, caso contrario não faz nada
		buy_and_hold_return=Actuator.sum(real_stock_delta) # compra a ação e segura durante todo o periodo
		
		class_metrics={'f1_monark':f1_score(real_movement_encoded,pred_movement_encoded),'accuracy':accuracy,'precision':precision_score(real_movement_encoded,pred_movement_encoded),'recall':recall_score(real_movement_encoded,pred_movement_encoded),'roc auc':roc_auc_score(real_movement_encoded,pred_movement_encoded)}
		return swing_return, buy_and_hold_return, class_metrics

	@staticmethod
	def autoBuy13(stock_real_array,stock_pred_array,saving_percentage=0.13):
		total_money_to_invest=Actuator.INITIAL_INVESTIMENT
		real_stock_delta=Actuator.getStockReturn(stock_real_array)
		pred_stock_delta=Actuator.getStockReturn(stock_pred_array)
		e=math.e
		corret_predicts_in_a_row=0
		savings_money=0
		current_money=total_money_to_invest
		for i in range(len(real_stock_delta)):
			if pred_stock_delta[i] > 0 and stock_real_array[i-1]>0:
				stocks_to_buy=0
				try:
					stock_buy_price=stock_real_array[i-1]
					stock_predicted_sell_price=stock_pred_array[i]
					predicted_valuing=stock_predicted_sell_price/stock_buy_price
					max_stocks_possible=math.floor(current_money/stock_buy_price)
					if (max_stocks_possible<0):
						max_stocks_possible=0
					if corret_predicts_in_a_row ==0:
						lucky_randomness=rd.uniform(.02,.07)
					elif corret_predicts_in_a_row ==1:
						lucky_randomness=rd.uniform(.04,.09)
					elif corret_predicts_in_a_row >=2:
						extra=(corret_predicts_in_a_row/7)
						if extra>1:
							extra=1
						lucky_randomness=rd.uniform(.07,.13)+.13*extra
					confidence=(-1+(e**(predicted_valuing**.6/1.13))**0.5)/5
					multiplier=(lucky_randomness+confidence)/2
					if multiplier > 0.23:
						multiplier=0.23
					stocks_to_buy=math.ceil(max_stocks_possible*multiplier)
					if stocks_to_buy<2:
						stocks_to_buy=2
				except Exception as e:
					print("Error on auto13")
					print(type(e))
					print(e.args)
					print(e)
				if real_stock_delta[i]<0:
					corret_predicts_in_a_row=0
					current_money+=real_stock_delta[i]*stocks_to_buy
				else:
					corret_predicts_in_a_row+=1
					current_money+=real_stock_delta[i]*stocks_to_buy*(1-saving_percentage)
					savings_money+=real_stock_delta[i]*stocks_to_buy*saving_percentage
		return current_money+savings_money-total_money_to_invest