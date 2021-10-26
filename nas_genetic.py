#!/bin/python3
# -*- coding: utf-8 -*-

import os
from Crawler import Crawler
from NeuralNetwork import NeuralNetwork
from Hyperparameters import Hyperparameters
from Utils import Utils
from Dataset import Dataset
from Enums import NodeType,Loss,Metric,Optimizers,Features
from StandardGeneticAlgorithm import StandardGeneticAlgorithm
from EnhancedGeneticAlgorithm import EnhancedGeneticAlgorithm
from PopulationManager import PopulationManager
from HallOfFame import HallOfFame
from Genome import Genome
from SearchSpace import SearchSpace

feature_group=0 #0-6
binary_classifier=False
stock='T'
use_enhanced=False

Genome.CACHE_WEIGHTS=False
search_maximum=True	
input_features=Hyperparameters.getFeatureGroups()

never_crawl=os.getenv('NEVER_CRAWL',default='False')
never_crawl=never_crawl.lower() in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade')

search_space=SearchSpace()
search_space.add(5,60,SearchSpace.Type.INT,'backwards_samples')
search_space.add(7,14,SearchSpace.Type.INT,'forward_samples')
search_space.add(1,5,SearchSpace.Type.INT,'lstm_layers')
search_space.add(500,5000,SearchSpace.Type.INT,'max_epochs')
search_space.add(100,5000,SearchSpace.Type.INT,'patience_epochs_stop')
search_space.add(0,1000,SearchSpace.Type.INT,'patience_epochs_reduce')
search_space.add(0.0,0.2,SearchSpace.Type.FLOAT,'reduce_factor')
search_space.add(0,128,SearchSpace.Type.INT,'batch_size')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'stateful')
search_space.add(True,True,SearchSpace.Type.BOOLEAN,'normalize')
search_space.add(Utils.getEnumBorder(Optimizers,False),Utils.getEnumBorder(Optimizers,True),SearchSpace.Type.INT,'optimizer')
search_space.add(NodeType.RELU,NodeType.LINEAR,SearchSpace.Type.INT,'activation_functions')
search_space.add(NodeType.RELU,NodeType.HARD_SIGMOID,SearchSpace.Type.INT,'recurrent_activation_functions')
search_space.add(False,False,SearchSpace.Type.BOOLEAN,'shuffle')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'use_dense_on_output')
search_space.add(10,200,SearchSpace.Type.INT,'layer_sizes')
search_space.add(0,0.3,SearchSpace.Type.FLOAT,'dropout_values')
search_space.add(0,0.3,SearchSpace.Type.FLOAT,'recurrent_dropout_values')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'bias')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'unit_forget_bias')
search_space.add(False,True,SearchSpace.Type.BOOLEAN,'go_backwards')
search_space=Genome.enrichSearchSpace(search_space)
amount_companies=1
train_percent=.8
val_percent=.3
if binary_classifier:
    input_features=[Features.UP]+input_features[feature_group]
    output_feature=Features.UP
    index_feature='Date'
    model_metrics=['accuracy','mean_squared_error']
    loss='categorical_crossentropy'
else:
    input_features=[Features.CLOSE]+input_features[feature_group]
    output_feature=Features.CLOSE
    index_feature='Date'
    model_metrics=['R2','mean_squared_error','mean_absolute_error','accuracy','cosine_similarity']
    loss='mean_squared_error'

start_date='01/01/2016' #Utils.FIRST_DATE
end_date='07/05/2021'
test_date='07/01/2021'
start_date_formated_for_file=''.join(Utils.extractNumbersFromDate(start_date,reverse=True))
end_date_formated_for_file=''.join(Utils.extractNumbersFromDate(end_date,reverse=True))
filename='{}_daily_{}-{}.csv'.format(stock,start_date_formated_for_file,end_date_formated_for_file)
crawler=Crawler()
filepath=crawler.getDatasetPath(filename)
if not Utils.checkIfPathExists(filepath) and not never_crawl:
    crawler.downloadStockDailyData(stock,filename,start_date=start_date,end_date=end_date)
    NeuralNetwork.enrichDataset(filepath)
    
def train_callback(genome):
    global stock,filepath,test_date,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier
    preserve_weights=False # TODO not implemented yet!
    hyperparameters=genome.toHyperparameters(input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)
    neuralNetwork=NeuralNetwork(hyperparameters,stock_name=stock,verbose=False)
    neuralNetwork.loadDataset(filepath,plot=False,blocking_plots=False,save_plots=True)
    neuralNetwork.buildModel(plot_model_to_file=True)
    neuralNetwork.train()
    neuralNetwork.restoreCheckpointWeights(delete_after=False)
    neuralNetwork.save()
    neuralNetwork.loadTestDataset(filepath,from_date=test_date,blocking_plots=False,save_plots=True)
    neuralNetwork.eval(plot=True,print_prediction=False,blocking_plots=False,save_plots=True)
    output=Utils.computeNNFitness(neuralNetwork.metrics,binary_classifier)
    neuralNetwork.destroy()
    return output

verbose_natural_selection=True
verbose_population_details=True
population_start_size_enh=100
max_gens=100
max_age=5
max_children=4
mutation_rate=0.1
recycle_rate=0.13
sex_rate=0.7
max_notables=5
elite=HallOfFame(max_notables, search_maximum)
if use_enhanced:
    ga=EnhancedGeneticAlgorithm(search_maximum,max_children,max_age,mutation_rate,sex_rate,recycle_rate)
else:
    ga=StandardGeneticAlgorithm(search_maximum,mutation_rate,sex_rate)
population=PopulationManager(ga,search_space,train_callback,population_start_size_enh,neural_genome=True,print_deltas=verbose_population_details)
population.hall_of_fame=elite
population.naturalSelection(max_gens,verbose_natural_selection,verbose_population_details)

for individual in elite.notables:
    print(str(individual))
Utils.printDict(elite.best,'Elite')
print('Evaluating best')

def test_callback(genome):
    global stock,filepath,test_date,input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier,start_date_formated_for_file,end_date_formated_for_file
    hyperparameters=genome.toHyperparameters(input_features,output_feature,index_feature,model_metrics,loss,train_percent,val_percent,amount_companies,binary_classifier)
    neuralNetwork=NeuralNetwork(hyperparameters,stock_name=stock,verbose=True)
    neuralNetwork.load()
    neuralNetwork.restoreCheckpointWeights(delete_after=False)
    neuralNetwork.loadTestDataset(filepath,from_date=test_date,blocking_plots=False,save_plots=True)
    neuralNetwork.eval(plot=True,print_prediction=True,blocking_plots=False,save_plots=True)
    neuralNetwork.destroy()
    NeuralNetwork.runPareto(use_ok_instead_of_f1=True,plot=True,blocking_plots=False,save_plots=True,label='{}-{}'.format(start_date_formated_for_file,end_date_formated_for_file))

test_callback(elite.getBestGenome())