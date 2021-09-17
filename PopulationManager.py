#!/bin/python
# -*- coding: utf-8 -*-

import time
from Genome import Genome
from Utils import Utils
from GeneticAlgorithm import GeneticAlgorithm
from StandardGeneticAlgorithm import StandardGeneticAlgorithm
from EnhancedGeneticAlgorithm import EnhancedGeneticAlgorithm


class PopulationManager(object):

	PRINT_REL_FREQUENCY=10
	MT_DNA_VALIDITY=15
	
	def __init__(self,genetic_algorithm,search_space,eval_callback,population_start_size,neural_genome=False,print_deltas=False,after_gen_callback=None):
		self.genetic_algorithm=genetic_algorithm
		self.space=self.genetic_algorithm.enrichSpace(search_space)
		has_age=False
		if type(self.genetic_algorithm) is EnhancedGeneticAlgorithm:
			self.genetic_algorithm.max_population=population_start_size*2
			has_age=True
		self.population=[]
		for i in range(population_start_size):
			self.population.append(Genome(search_space,eval_callback,neural_genome,has_age=has_age))
		self.print_deltas=print_deltas
		self.hall_of_fame=None
		self.after_gen_callback=after_gen_callback
		self.last_run_population_sizes=[]

	def __del__(self):
		self.genetic_algorithm=None
		self.space=None
		for individual in self.population:
			if GeneticAlgorithm.FREE_MEMORY_MANUALLY:
				del individual
		self.population=None
		self.print_deltas=None
		self.hall_of_fame=None
		self.after_gen_callback=None

	def naturalSelection(self, gens, verbose=False, verbose_generations=None):
		mean_delta=0.0
		self.last_run_population_sizes=[]
		for g in range(1,gens+1):
			t1=time.time()
			if self.genetic_algorithm.looking_highest_fitness:
				best_out=float('-inf')
			else:
				best_out=float('inf')
			
			if verbose:
				print('\tEvaluating individuals...')
			last_print=1
			for p,individual in enumerate(self.population):
				individual.evaluate()
				individual.gen=g
				output=individual.output
				if self.genetic_algorithm.looking_highest_fitness:
					if output>best_out:
						best_out=output	
				else:
					if output<best_out:
						best_out=output
				if verbose:
					percent=(p+1)/float(len(self.population))*100.0
					if  percent>=last_print*PopulationManager.PRINT_REL_FREQUENCY :
						last_print+=1
						print('\t\tprogress: {:2.2f}%'.format(percent))
			if verbose:
				print('\tEvaluated individuals...OK')
				print('\tCalculating fitness...')
			self.population=self.genetic_algorithm.fit(self.population)
			if verbose:
				print('\tCalculated fitness...OK')
			if self.hall_of_fame is not None:
				if verbose:
					print('\tSetting hall of fame...')
				self.hall_of_fame.update(self.population,g)
				if verbose:
					print('\tSetted hall of fame...OK')
			if g<gens:
				if verbose:
					print('\tSelecting and breeding individuals...')
				self.population=self.genetic_algorithm.select(self.population)
				if verbose:
					print('\tSelected and breed individuals...OK')
					enhanced_str=' and aging'
					if type(self.genetic_algorithm) is not EnhancedGeneticAlgorithm:
						enhanced_str=''
					print('\tMutating{} individuals...'.format(enhanced_str))
				self.population=self.genetic_algorithm.mutate(self.population)
				if g%PopulationManager.MT_DNA_VALIDITY==0:
					for individual in self.population:
						individual.resetMtDna()
				if verbose:
					enhanced_str=' and aged'
					if type(self.genetic_algorithm) is not EnhancedGeneticAlgorithm:
						enhanced_str=''
					print('\tMutated{} individuals...OK'.format(enhanced_str))
			else:
				self.population.sort()
			t2=time.time()
			delta=t2-t1
			mean_delta+=delta
			self.last_run_population_sizes.append(len(self.population))
			if self.after_gen_callback is not None:
				args_list=[len(self.population),g,best_out,delta,self.population,self.hall_of_fame]
				self.after_gen_callback(args_list)
			if len(self.population)<2:
				print('Early stopping generation {} due to its small size {}, this population died!'.format(g,len(self.population)))
				break
			if verbose_generations or self.print_deltas:
				print('Generation {} of {}, size: {} takes: {}'.format(g,gens,len(self.population),Utils.timestampByExtensive(delta)))
		return mean_delta/gens