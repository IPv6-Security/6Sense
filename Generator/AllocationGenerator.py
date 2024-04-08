"""
Copyright 2024 Georgia Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 """

import time
import random
import ipaddress
import math
import csv
from collections import Counter 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from Generator.cython_print_ips import print_ips
from iplibrary import AS_Processor

class NaiveWeightUpdateWithoutZerosEvenFirst():
    """
    Generate Allocations Weighted by the Hitrate. Even Sample the First Iteration. 
    """
    def __init__(self, dataset, threshold, ppi, filepath, *args):
        
        self.filepath = filepath
        # Get Allocation Strings
        self.allocation_strings_1 = dataset.allocation_proc_models.allocation_strings
        allocation_strings, allocation_indices = np.unique(np.array(self.allocation_strings_1), return_index=True)
        self.allocation_strings = list(allocation_strings)
        
        # Get Dataset
        self.dataset = dataset.allocation_dataset
        self.full_dataset = dataset
        self.unique_allocations = self.dataset[allocation_indices, :]
       
        # GET PROPER INDEXING:
        self.First = True
        self.threshold = threshold
        self.str_allocations = []
        self.ipv6_cidrs = {}
        self.indices = {}
        index = 0
        for a in self.allocation_strings:
            split = a.split("/")[0]
            self.str_allocations.append(split)
            self.ipv6_cidrs[a] = ipaddress.IPv6Network(a)
            self.indices[split] = index
            index += 1
            
        print("Number of Allocations: ", len(self.str_allocations))        
  
        # INITIALIZE WEIGHTS:
        self.weights = np.ones((self.unique_allocations.shape[0]))/self.unique_allocations.shape[0]
        
        # History:
        self.history_length = 3
        self.current_history = 0
        self.history = np.zeros((self.history_length, self.unique_allocations.shape[0]))
        self.historical_weights = np.zeros((self.history_length, self.unique_allocations.shape[0]))
        
        # Set Filenames:
        self.allocation_weights_csv = self.filepath + "/allocation_weights"
        self.allocation_hitrate_csv = self.filepath + "/allocation_hitrate"
        self.allocation_hits_csv = self.filepath + "/allocation_hits"
        self.allocation_generated_csv = self.filepath + "/allocation_generated"
        self.allocation_top_ten = self.filepath + "/allocation_top_ten"
        self.allocation_top_ten_hits = self.filepath + "/allocation_top_ten_hits"
        self.allocation_top_ten_generated = self.filepath + "/allocation_top_ten_generated"

        # Allocation Strings -> Allocation Indexes
        self.allocation_mapping = {}
        for x in range(0, len(self.str_allocations)):
            self.allocation_mapping[self.str_allocations[x]] = x
            
        self.aliased = np.ones((self.unique_allocations.shape[0]))
        self.allocation_times = []
        
                
    def sample(self, numberToGenerate):
        np.random.seed(123)
        t1 = time.time()
        if self.First == True:
            number_of_allocations = self.unique_allocations.shape[0]
            sampled_1 = np.tile(self.unique_allocations, (math.ceil((numberToGenerate/number_of_allocations)), 1)) 
            sampled = np.zeros((numberToGenerate, 16))
            sampled[0:sampled_1.shape[0]] = sampled_1[:numberToGenerate]
            self.sampled_allocations = list(np.tile(np.array(self.str_allocations), math.ceil((numberToGenerate/number_of_allocations))))[:numberToGenerate]
            t2 = time.time()
            self.allocation_times.append(t2 - t1)
            return sampled
        else:
            random_indices = np.random.choice(self.weights.shape[0], 
                                              size=numberToGenerate, 
                                              replace=True, 
                                              p=self.weights)
            allocations = self.unique_allocations[random_indices, :]
            self.sampled_allocations = list(np.array(self.str_allocations)[random_indices])

            t2 = time.time()
            self.allocation_times.append(t2 - t1)
            return allocations
        
    def update(self, input_ips, allocation_dictionary, allocation_list_generated_from, aliases, *args):
        """
        Get the Allocation with the maximum hitrate. Set this to be generated Threshold% of the time. 
        """
        # RECONSTRUCT ALLOCATIONS FROM HITS and ALIASES
        # Hits
        allocations = []
        for a in input_ips:
            if a in allocation_dictionary:
                allocations.append(allocation_dictionary[a])

        # Aliases
        aliased_allocations = []
        for a in aliases:
            if a in allocation_dictionary:
                aliased_allocations.append(allocation_dictionary[a])
        
        # GET TOP ALLOCATION
        word_counts = Counter(allocations)

        # GET COUNT OF EACH ACTIVE ALLOCATION
        hitrates = np.zeros((self.unique_allocations.shape[0]))
        word_count_dict = dict(word_counts)
        for key in word_count_dict:
            hitrates[self.indices[key]] = word_count_dict[key] #allocation_list.append(key)
            
        # GET COUNT OF EACH ALIASED ALLOCATION
        hitrates_aliased = np.zeros((self.unique_allocations.shape[0]))
        if len(aliased_allocations) > 0:
            word_counts_aliased = Counter(aliased_allocations)
            word_count_dict_aliased = dict(word_counts_aliased)
            for key in word_count_dict_aliased:
                hitrates_aliased[self.indices[key]] = word_count_dict_aliased[key] #allocation_list.append(key)
        
        self.aliased[hitrates_aliased>hitrates] = 0 # Set As Aliases

        # GET GENRATED PER ALLOCATION
        total_generated = np.zeros((len(self.str_allocations)))

        if len(allocation_list_generated_from) > 1:
            unique_als = allocation_list_generated_from[0]
            al_freqs = allocation_list_generated_from[1]
        else:
            unique_als, al_freqs = np.unique(allocation_list_generated_from[0], return_counts=True)

        for y in range(0, unique_als.shape[0]):
            total_generated[self.indices[unique_als[y]]] = al_freqs[y]

        # SET HISTORY
        self.history[self.current_history] = hitrates
        self.historical_weights[self.current_history] = total_generated
        self.current_history = (self.current_history + 1) % self.history_length
        
        summed_hitrates = np.sum(self.history, axis=0)
        summed_weights = np.sum(self.historical_weights, axis=0)

        # DIVIDE BY WEIGHTS TO ESTIMATE MOST COMMON ADDRESSES
        scaled_hitrates = summed_hitrates/summed_weights
        scaled_hitrates[np.isnan(scaled_hitrates)] = 0
        
        if self.First == True:
            self.First = False
        else:
            pass

        self.weights = self.aliased * scaled_hitrates
        self.weights = self.weights/np.sum(self.weights)
       





class OnlySampleNonHitAllocations():
    """
    Generate Allocations, sampling Evenly but removing Allocations where we see hits. 
    """
    def __init__(self, dataset, threshold, ppi, filepath, *args):
        
        self.filepath = filepath
        # Get Allocation Strings
        self.allocation_strings_1 = dataset.allocation_proc_models.allocation_strings
        allocation_strings, allocation_indices = np.unique(np.array(self.allocation_strings_1), return_index=True)
        self.allocation_strings = list(allocation_strings)
        
        # Get Dataset
        self.dataset = dataset.allocation_dataset
        self.full_dataset = dataset
        self.unique_allocations = self.dataset[allocation_indices, :]
       
        # GET PROPER INDEXING:
        self.First = True
        self.threshold = threshold
        self.str_allocations = []
        self.ipv6_cidrs = {}
        self.indices = {}
        index = 0
        for a in self.allocation_strings:
            split = a.split("/")[0]
            self.str_allocations.append(split)
            self.ipv6_cidrs[a] = ipaddress.IPv6Network(a)
            self.indices[split] = index
            index += 1
            
        print("Number of Allocations: ", len(self.str_allocations))        
  
        # INITIALIZE WEIGHTS:
        self.weights = np.ones((self.unique_allocations.shape[0]))/self.unique_allocations.shape[0]
        
        # History:
        self.history_weights = np.zeros((self.unique_allocations.shape[0]))
        
        # Set Filenames:
        self.allocation_weights_csv = self.filepath + "/allocation_weights"
        self.allocation_hitrate_csv = self.filepath + "/allocation_hitrate"
        self.allocation_hits_csv = self.filepath + "/allocation_hits"
        self.allocation_generated_csv = self.filepath + "/allocation_generated"
        self.allocation_top_ten = self.filepath + "/allocation_top_ten"
        self.allocation_top_ten_hits = self.filepath + "/allocation_top_ten_hits"
        self.allocation_top_ten_generated = self.filepath + "/allocation_top_ten_generated"
            
        # Allocation Strings -> Allocation Indexes
        self.allocation_mapping = {}
        for x in range(0, len(self.str_allocations)):
            self.allocation_mapping[self.str_allocations[x]] = x
            
        self.aliased = np.ones((self.unique_allocations.shape[0]))
        self.allocation_times = []
        
                
    def sample(self, numberToGenerate):
        
        t1 = time.time()
        if self.First == True:
            number_of_allocations = self.unique_allocations.shape[0]
            sampled_1 = np.tile(self.unique_allocations, (math.ceil((numberToGenerate/number_of_allocations)), 1)) 
            sampled = np.zeros((numberToGenerate, 16))
            sampled[0:sampled_1.shape[0]] = sampled_1[:numberToGenerate]
            self.sampled_allocations = list(np.tile(np.array(self.str_allocations), math.ceil((numberToGenerate/number_of_allocations))))[:numberToGenerate]
            t2 = time.time()
            self.allocation_times.append(t2 - t1)
            return sampled
        else:
            random_indices = np.random.choice(self.weights.shape[0], 
                                              size=numberToGenerate, 
                                              replace=True, 
                                              p=self.weights)
            allocations = self.unique_allocations[random_indices, :]
            self.sampled_allocations = list(np.array(self.str_allocations)[random_indices])
            t2 = time.time()
            self.allocation_times.append(t2 - t1)
            return allocations
        
    def update(self, input_ips, allocation_dictionary, allocation_list_generated_from, aliases, *args):
        """
        Get the Allocation with the maximum hitrate. Set this to be generated Threshold% of the time. 
        """
        # RECONSTRUCT ALLOCATIONS FROM HITS and ALIASES
        # Hits
        print("BEGIN ALLOCATION WEIGHT UPDATE")

        allocations = []
        for a in input_ips:
            if a in allocation_dictionary:
                allocations.append(allocation_dictionary[a])

        for a in aliases:
            if a in allocation_dictionary:
                allocations.append(allocation_dictionary[a])


        # GET IF EACH ALLOCATION HAS A HIT OR NOT
        hitrates = np.zeros((self.unique_allocations.shape[0]))
        for key in allocations:
            hitrates[self.indices[key]] = 1 #allocation_list.append(key)
        self.history_weights= self.history_weights + hitrates

        
        # REMOVE WEIGHTS THAT HAVE SEEN ACTIVE ALLOCATIONS
        self.weights = (self.history_weights == 0).astype(int)
        weight_length = np.sum(self.weights)
        self.weights = self.weights/weight_length

        print("TOTAL ALLOCATIONS TO SAMPLE FROM: ", weight_length)


        if self.First == True:
            self.First = False
        else:
            pass




class BalancedAllocationSampling():
    """
    Generate Allocations Weighted by the Hitrate. Even Sample the First Iteration. 
    """
    def __init__(self, dataset, threshold, ppi, filepath, *args):
        
        self.filepath = filepath
        # Get Allocation Strings
        self.allocation_strings_1 = dataset.allocation_proc_models.allocation_strings
        allocation_strings, allocation_indices = np.unique(np.array(self.allocation_strings_1), return_index=True)
        self.allocation_strings = list(allocation_strings)
        
        # Get Dataset
        self.dataset = dataset.allocation_dataset
        self.full_dataset = dataset
        self.unique_allocations = self.dataset[allocation_indices, :]
       
        # GET PROPER INDEXING:
        self.First = True
        self.threshold = threshold
        self.str_allocations = []
        self.ipv6_cidrs = {}
        self.indices = {}
        index = 0
        for a in self.allocation_strings:
            split = a.split("/")[0]
            self.str_allocations.append(split)
            self.ipv6_cidrs[a] = ipaddress.IPv6Network(a)
            self.indices[split] = index
            index += 1
            
        print("Number of Allocations: ", len(self.str_allocations))        
  
        # INITIALIZE WEIGHTS:
        self.weights = np.ones((self.unique_allocations.shape[0]))/self.unique_allocations.shape[0]
        self.weights_unseen = np.ones((self.unique_allocations.shape[0]))/self.unique_allocations.shape[0]
        self.weights_discarded = np.ones((self.unique_allocations.shape[0]))/self.unique_allocations.shape[0]
        self.any_discarded_allocations = False

        
        # History:
        self.history_length = 3
        self.current_history = 0
        self.history = np.zeros((self.history_length, self.unique_allocations.shape[0]))
        self.historical_weights = np.zeros((self.history_length, self.unique_allocations.shape[0]))

        self.historical_unseen_weights = np.zeros((self.unique_allocations.shape[0]))
        self.alias_tracker =  np.ones((self.unique_allocations.shape[0]))

        self.discarded_allocation_list = set([])

        
        # Set Filenames:
        self.allocation_weights_csv = self.filepath + "/allocation_weights"
        self.allocation_hitrate_csv = self.filepath + "/allocation_hitrate"
        self.allocation_hits_csv = self.filepath + "/allocation_hits"
        self.allocation_generated_csv = self.filepath + "/allocation_generated"
        self.allocation_top_ten = self.filepath + "/allocation_top_ten"
        self.allocation_top_ten_hits = self.filepath + "/allocation_top_ten_hits"
        self.allocation_top_ten_generated = self.filepath + "/allocation_top_ten_generated"
            
            
        # Allocation Strings -> Allocation Indexes
        self.allocation_mapping = {}
        for x in range(0, len(self.str_allocations)):
            self.allocation_mapping[self.str_allocations[x]] = x
            
        self.aliased = np.ones((self.unique_allocations.shape[0]))
        self.allocation_times = []
        
                
    def sample(self, numberToGenerate):
        np.random.seed(123)
        t1 = time.time()
        if self.First == True:
            number_of_allocations = self.unique_allocations.shape[0]
            sampled_1 = np.tile(self.unique_allocations, (math.ceil((numberToGenerate/number_of_allocations)), 1)) 
            sampled = np.zeros((numberToGenerate, 16))
            sampled[0:sampled_1.shape[0]] = sampled_1[:numberToGenerate]
            self.sampled_allocations = list(np.tile(np.array(self.str_allocations), math.ceil((numberToGenerate/number_of_allocations))))[:numberToGenerate]
            t2 = time.time()
            self.allocation_times.append(t2 - t1)
            return sampled
        else:
            allocations = np.zeros((numberToGenerate, self.unique_allocations.shape[1]))

            to_generate_1 = int(numberToGenerate*0.6)
            to_generate_2 = int(numberToGenerate*0.2)
            to_generate_3 = numberToGenerate - to_generate_1 - to_generate_2


            # CHOICE #1 - X% of the budget goes to primary weighted sampling:
            random_indices = np.random.choice(self.weights.shape[0], 
                                              size=to_generate_1, 
                                              replace=True, 
                                              p=self.weights)
            allocations[0:random_indices.shape[0],:] = self.unique_allocations[random_indices, :]
            self.sampled_allocations = list(np.array(self.str_allocations)[random_indices])
            
            # CHOICE #2 - Y% goes to previously unseen allocations
            random_indices2 = np.random.choice(self.weights_unseen.shape[0], 
                                    size=to_generate_2, 
                                    replace=True, 
                                    p=self.weights_unseen)
            allocations[random_indices.shape[0]:random_indices.shape[0] + random_indices2.shape[0],:] = self.unique_allocations[random_indices2, :]
            self.sampled_allocations += list(np.array(self.str_allocations)[random_indices2])

            # CHOICE #3 - Z% goes to previously discarded allocations or to regular weights if no discarded allocations yet.
            
            if self.any_discarded_allocations == True:
                random_indices3 = np.random.choice(self.weights_discarded.shape[0], 
                            size=to_generate_3, 
                            replace=True, 
                            p=self.weights_discarded)
            else:
                random_indices3 = np.random.choice(self.weights.shape[0], 
                            size=to_generate_3, 
                            replace=True, 
                            p=self.weights)

            allocations[random_indices2.shape[0] + random_indices.shape[0]:,:] = self.unique_allocations[random_indices3, :]
            self.sampled_allocations += list(np.array(self.str_allocations)[random_indices3])


            t2 = time.time()
            self.allocation_times.append(t2 - t1)
            return allocations
        
    def update(self, input_ips, allocation_dictionary, allocation_list_generated_from, aliases, *args):
        """
        Get the Allocation with the maximum hitrate. Set this to be generated Threshold% of the time. 
        """
        # RECONSTRUCT ALLOCATIONS FROM HITS and ALIASES
        # Hits
        print("BEGIN ALLOCATION UPDATE")

        allocations = []
        unseen_allocations = []
        discarded_hits = 0
        for a in input_ips:
            if a in allocation_dictionary:
                unseen_allocations.append(allocation_dictionary[a])
                if allocation_dictionary[a] not in self.discarded_allocation_list:
                    allocations.append(allocation_dictionary[a])
                else:
                    discarded_hits += 1

        print("Hits from Discarded Allocations: ", discarded_hits)

        
        # Aliases
        aliased_allocations = []
        for a in aliases:
            if a in allocation_dictionary:
                aliased_allocations.append(allocation_dictionary[a])


        # GET UNSEEN WEIGHTS
        unseen_allocations = set(unseen_allocations + aliased_allocations)
        hitrates_unseen = np.zeros((self.unique_allocations.shape[0]))
        for key in unseen_allocations:
            hitrates_unseen[self.indices[key]] = 1
        self.historical_unseen_weights= self.historical_unseen_weights + hitrates_unseen

        self.weights_unseen = (self.historical_unseen_weights == 0).astype(int)
        weight_length = np.sum(self.weights_unseen)
        self.weights_unseen = self.weights_unseen/weight_length

        print("TOTAL UNSEEN ALLOCATIONS TO SAMPLE FROM: ", weight_length)
        
        # GET TOP ALLOCATION
        word_counts = Counter(allocations)
        
        # GET COUNT OF EACH ACTIVE ALLOCATION
        hitrates = np.zeros((self.unique_allocations.shape[0]))
        word_count_dict = dict(word_counts)
        for key in word_count_dict:
            hitrates[self.indices[key]] = word_count_dict[key] 
            
        # GET COUNT OF EACH ALIASED ALLOCATION
        hitrates_aliased = np.zeros((self.unique_allocations.shape[0]))
        if len(aliased_allocations) > 0:
            word_counts_aliased = Counter(aliased_allocations)
            word_count_dict_aliased = dict(word_counts_aliased)
            for key in word_count_dict_aliased:
                hitrates_aliased[self.indices[key]] = word_count_dict_aliased[key] 
        
        self.aliased[hitrates_aliased>hitrates] = 0 # Set As Aliases

        # GET GENERATED PER ALLOCATION
        total_generated = np.zeros((len(self.str_allocations)))

        if len(allocation_list_generated_from) > 1:
            unique_als = allocation_list_generated_from[0]
            al_freqs = allocation_list_generated_from[1]
        else:
            unique_als, al_freqs = np.unique(allocation_list_generated_from[0], return_counts=True)

        for y in range(0, unique_als.shape[0]):
            total_generated[self.indices[unique_als[y]]] = al_freqs[y]

        # SET HISTORY
        self.history[self.current_history] = hitrates
        self.historical_weights[self.current_history] = total_generated
        self.current_history = (self.current_history + 1) % self.history_length

        print("Hitrates: ", np.sum(hitrates))
        print("Generated: ", np.sum(total_generated))
        print("History: ", np.sum(self.history))
        print("Generated History: ", np.sum(self.historical_weights))

        summed_hitrates = np.sum(self.history, axis=0)
        summed_weights = np.sum(self.historical_weights, axis=0)
        print("Summed Weights: ", summed_weights)

        # DIVIDE BY WEIGHTS TO ESTIMATE MOST COMMON ADDRESSES
        scaled_hitrates = summed_hitrates/summed_weights
        scaled_hitrates[np.isnan(scaled_hitrates)] = 0

        print("Total Hitrates: ", np.sum(scaled_hitrates))

        self.weights = self.aliased * scaled_hitrates
        print("Unscaled Weights: ", np.sum(self.weights))
        self.weights = self.weights/np.sum(self.weights)

        # COMPUTE EVEN SAMPLING WEIGHTS FOR DISCARDED ALLOCATIONS
        self.alias_tracker = self.alias_tracker * self.aliased

        current_allocations_sampled_from = (self.weights != 0).astype(int) + (self.weights_unseen!=0).astype(int) + (self.alias_tracker == 0).astype(int)
        print("CURRENT ALLOCATIONS SAMPLED FROM: ", np.sum(current_allocations_sampled_from))
        print("UNSEEN ALLOCATIONS: ", np.sum((self.weights_unseen!=0).astype(int)))
        print("CURRENTLY SAMPLING ALLOCATIONS: ", np.sum((self.weights != 0).astype(int)))
        print("ALL ALLOCATIONS: ", self.weights.shape[0])
        print("ESTIMATED DISCARDED ALLOCATIONS: ", self.weights.shape[0] - np.sum(current_allocations_sampled_from))
        print("ALIASES: ", np.sum((self.alias_tracker == 0).astype(int)))

        if self.any_discarded_allocations == True or np.sum(current_allocations_sampled_from) < self.unique_allocations.shape[0]:
            print("Setting Discarded Allocations")
            self.any_discarded_allocations = True
            self.weights_discarded = self.alias_tracker * (current_allocations_sampled_from == 0).astype(int)
            weight_length = np.sum(self.weights_discarded)
            self.weights_discarded = self.weights_discarded/weight_length

            for w in range(0, self.weights_discarded.shape[0]):
                if self.weights_discarded[w] != 0:
                    self.discarded_allocation_list.add(self.str_allocations[w])

            print("TOTAL DISCARDED ALLOCATIONS: ", len(self.discarded_allocation_list))


        if self.First == True:
            self.First = False
        else:
            pass
        