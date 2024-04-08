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
from icecream import ic

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from iplibrary import AS_Processor
from Generator.cython_print_ips import print_ips
import config as conf



class FixedLowBit():
    def __init__(self, dataset, PriorLevel, hps, allocation_gen, **kwargs):
        self.string_ending = ":0000:0000:0000:0001"
        if "string" in hps:
            self.string_ending = hps["string"]
        
        self.upper_generator = PriorLevel
    
    def sample(self, number_to_generate, *args):
        sampled = self.upper_generator(number_to_generate, "all_ips")
        addresses = (16* sampled).astype(int)
        addresses = print_ips(addresses)
        
        final_addresses = []
        for ip in addresses:
            if ip[-2:] == "::":
                final_addresses.append(ip[:-2] + self.string_ending)
            else:
                final_addresses.append(ip + self.string_ending)
                
        return final_addresses
    
    def update(hits, *args):
        pass


class IterativeOnPriorPatternsLowerGeneratorFaster():
    """
    1. Generate from Known Patterns.
    2. Incremement index by 1
    3. Generate from Known Patterns + index
    4. Repeat 
    """
    def __init__(self, dataset, PriorLevel, hps, allocation_gen, **kwargs):

        self.upper_times = []
        self.lower_times = []

        self.upper_generator = PriorLevel
        self.allocation_gen = allocation_gen
        
        self.allocation_list = list(set(hps['Allocations']))
                
        als = np.array(hps['Allocations'])
        lowers = dataset['all_ips'][:, 16:]
        uppers = dataset['all_ips'][:, :16]
        
        al_inds = np.argsort(als)
        sorted_als = als[al_inds]
        sorted_lowers = lowers[al_inds, :]
        sorted_uppers = print_ips(uppers[al_inds, :].astype(int))
        
        self.low_bits = {}
        current_ind = 0
        if_one_exists = 0
        no_one = 0
        
        self.seen_upper = {}
        self.seen = {}
        
        # Loop through lowers and add them to each allocation
        for x in range(0, len(sorted_als)):
            hashed_representation = integer_extraction(sorted_lowers[x])
            
            if sorted_uppers[x] not in self.seen_upper:
                self.seen_upper[sorted_uppers[x]] = set([])
            self.seen_upper[sorted_uppers[x]].add(hashed_representation)
            
            formatted_al = sorted_als[x].split('/')[0]
            if formatted_al not in self.seen:
                self.seen[formatted_al] = set([])
            self.seen[formatted_al].add(hashed_representation) 
            
            if x == len(sorted_als)-1 or sorted_als[x] != sorted_als[x+1]:
                current_val = sorted_als[x]
                current_lower, counts =  np.unique(sorted_lowers[current_ind:x+1, :], return_counts=True, axis=0)
                cuargs = np.argsort(counts)
                current_lower_sorted = current_lower[cuargs, :][::-1] 
                value = np.equal(current_lower_sorted,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all(1)
                if any(value):
                    if_one_exists += 1
                else:
                    new_lower = np.zeros((current_lower_sorted.shape[0]+1, current_lower_sorted.shape[1]))
                    new_lower[:-1, :] = current_lower_sorted
                    new_lower[-1, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    current_lower_sorted = new_lower
                    no_one += 1
                    hashed_representation_1 = integer_extraction(current_lower_sorted[-1, :])
                    self.seen[formatted_al].add(hashed_representation_1) 
                
                allocation_calculated = current_val.split('/')[0]
                self.low_bits[allocation_calculated] = current_lower_sorted                
                current_ind = x+1
           
        self.upper_bits = {} # Upper-64s We've Seen Before
        self.upper_bit_numbers = {}
        
        self.original_allocations = {}
        self.current_iter = {} # What to add to next iteration
        self.current_original_index = {}
        self.current_low_bit_number = {}
        for al in self.low_bits:
            self.original_allocations[al] = len(self.low_bits[al])
            self.current_iter[al] = 0
            self.current_original_index[al] = 0
            self.current_low_bit_number[al] = self.low_bits[al].shape[0]
            low_bit_np = np.zeros((self.current_low_bit_number[al]*10, self.low_bits[al].shape[1]))
            low_bit_np[:self.current_low_bit_number[al], :] = self.low_bits[al]
            self.low_bits[al] = np.copy(low_bit_np)
            
    
    
    def sample(self, number_to_generate, total=None):
        print("################")
        print("Begin Sampling")
        # GET UPPER-64s
        t1 = time.time()
        sampled = self.upper_generator(number_to_generate, "all_ips", None) # Numpy
        t111 = time.time()
        self.upper_times.append(t111-t1)
        print("Upper 64 Sampling Time: ", t111-t1)
        addresses = (16* sampled).astype(int)
        addresses_final = np.zeros((addresses.shape[0], 32))
        addresses_printed = print_ips(addresses) # String Upper-64s


        # CHECK INDEX OF UPPER-BITS
        inds = [] # Index per Upper-64 within the allocation list
        allocation_number = 0   
        
        seen_iter = 0
        not_seen_iter = 0
        
        for i in range(0, len(addresses_printed)):
            # GET LOWER-64 INDICES
            a = addresses_printed[i]
            allocation = self.allocation_gen.sampled_allocations[i]
            
            
            if a in self.upper_bits:
                inds.append(self.upper_bits[a])
                self.upper_bits[a] += 1
            else:
                inds.append(0)
                self.upper_bits[a] = 1
                # Prevent Repeats of Upper-64s
                self.upper_bit_numbers[a] = 1
                
            if a not in self.seen_upper:
                self.seen_upper[a] = set([])


            # GENERATE LOWER-64
            addresses_final[i, :16] = addresses[i, :16]
            numerical_representation = 1
            matched = False
            new = False            

            while matched == False:
                if inds[i] < self.current_low_bit_number[allocation]:
                    # Loop through if in seed
                    addresses_final[i, 16:] = self.low_bits[allocation][inds[i]]
                    # Set as having already seen Lower-64
                    numerical_representation = integer_extraction(addresses_final[i, 16:]) #int("".join(["{:01x}".format(int(l)) for l in addresses_final[i, 16:]]), 16)
                    # Update Indices
                    self.upper_bits[a] += 1
                    inds[i] += 1
                    not_seen_in_allocations = True
                    seen_iter += 1
                else:
                    # Loop through if in allocations seen before, if not add to self.low_bits[alllocation]
                    new = True
                    addresses_final[i, 16:] = self.low_bits[allocation][self.current_original_index[allocation]]
                    str_number = hex(self.current_iter[allocation]).lstrip("0x").rstrip("L")

                    list1 = [int(s, 16) for s in str_number]

                    list1.reverse()

                    for l in range(1, len(list1)+1):
                        addresses_final[i, len(addresses_final[i]) - l] = (list1[l-1] + addresses_final[i, len(addresses_final[i]) - l]) % 16
                    numerical_representation = integer_extraction(addresses_final[i, 16:]) #int("".join(["{:01x}".format(int(l)) for l in addresses_final[i, 16:]]), 16)
                    
                    # Update Indices
                    self.upper_bits[a] += 1
                    inds[i] += 1

                    # Update How Many of Each Allocation have been seen so far. 
                    self.current_original_index[allocation] = (self.current_original_index[allocation] + 1) % (self.original_allocations[allocation])
                    if self.current_original_index[allocation] == 0:
                        self.current_iter[allocation] += 1

                    # Add to Current Allocation List
                    if numerical_representation not in self.seen[allocation]:
                        not_seen_in_allocations = True
                        if self.current_low_bit_number[allocation] < self.low_bits[allocation].shape[0]:
                            self.low_bits[allocation][self.current_low_bit_number[allocation]] = addresses_final[i, 16:]
                        else:
                            low_bit_np = np.zeros((self.current_low_bit_number[allocation]*10, self.low_bits[allocation].shape[1]))
                            low_bit_np[:self.current_low_bit_number[allocation], :] = self.low_bits[allocation]
                            self.low_bits[allocation] = np.copy(low_bit_np)
                        self.current_low_bit_number[allocation] += 1
                        
                        self.seen[allocation].add(numerical_representation)
                        
                    else:
                        not_seen_in_allocations = False
                        
                    not_seen_iter += 1 
                
                if not_seen_in_allocations and numerical_representation not in self.seen_upper[a]:
                    matched = True
                    

            # Offset for the last update
            self.upper_bits[a] -= 1
            inds[i] -= 1


        print("Iterations of Previously Seen Lower-64s: ", seen_iter)
        print("Iterations of Previously UnSeen Lower-64s: ", not_seen_iter)

        addresses_to_return = print_ips(addresses_final.astype(int))
        t2 = time.time()
        print("Lower Sampling Time: ", t2-t111)
        self.lower_times.append(t2-t111)
        
        return addresses_to_return
    
    def update(self, *args):
        pass   
    
def integer_extraction(input_integers):
    return int("".join(["{:01x}".format(int(l)) for l in input_integers]), 16)

def powers_hash(input_integers, qs):
    return np.sum(np.power(input_integers, qs))
    
    
    
class UpperSeedSampler():
    """
    1. Generate from Known Patterns.
    2. Incremement index by 1
    3. Generate from Known Patterns + index
    4. Repeat (indefinitely?)
    5. Generate Duplicate?? But probably never reach this point. Could we even?
    """
    def __init__(self, dataset, PriorLevel, hps, **kwargs):
        print("Begin Parsing Upper-64s")        
        self.allocation_generator = PriorLevel

        ip_list = print_ips(dataset['all_ips'].astype(int))
        asproc = AS_Processor(ip_list, prefix_filename=conf.UPDATED_PFX_AS_FILE)
        self.allocation_object = hps["AllocationObject"]
        allocation_list = asproc.allocation_strings
        self.allocation_list = []
        for a in allocation_list:
            self.allocation_list.append(a.split('/')[0])

        print("Number of Total Allocations: ", len(self.allocation_list))

        uppers = dataset['all_ips'][:, :16]/16

        print("Number of Total Upper-64s: ", uppers.shape)

        self.uppers_dictionary = {}
        self.allocation_index = {}

        for index in range(0, len(self.allocation_list)):
            allocation = self.allocation_list[index]
            upper64 = uppers[index, :]
            if allocation not in self.uppers_dictionary:
                self.uppers_dictionary[allocation] = []
                self.allocation_index[allocation] = 0

            self.uppers_dictionary[allocation].append(upper64)

        print("Total Examined Allocations: ", len(self.uppers_dictionary.keys()))

        for al in self.uppers_dictionary:
            numpy_list = self.uppers_dictionary[al]
            numpy_array = np.array(numpy_list)
            self.uppers_dictionary[al] = numpy_array


    def sample(self, number_to_generate, total=None):
        numpy_allocations = self.allocation_generator(number_to_generate, "")
        allocations = self.allocation_object.sampled_allocations

        upper64s = np.zeros((number_to_generate, 16))

        missed = 0
        for index in range(0, len(allocations)):
            if allocations[index] in self.uppers_dictionary:
                upper64s[index, :] = self.uppers_dictionary[allocations[index]][self.allocation_index[allocations[index]]]
                total_upper64s = self.uppers_dictionary[allocations[index]].shape[0]
                self.allocation_index[allocations[index]] = (self.allocation_index[allocations[index]] + 1) % total_upper64s
            else:
                upper64s[index, :] = numpy_allocations[index, :] - 1/16
                missed += 1

        print("Missed Allocations: ", missed)

        return upper64s

    def update(self, *args):
        pass   
    
    