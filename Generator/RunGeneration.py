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

import os
import time
import math
import csv
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from iplibrary import AS_Processor, ExpandIPs
from Generator.cython_print_ips import print_ips

from Generator.ScanDeduplicateDealias import ScanDeduplicateDealiasObject, ThreadingScanDeduplicateDealiasClientSide

import config as conf


def ComparisonModel_online(numberToGenerate, comparisonName, comparisonFunction, seedObject, ppi=5000000, per_iteration=100000, allocation_gradient_amount=1000000, allocation_gradient_threshold=0.005, **kwargs):    
    total_generated = 0
    IP_Model = seedObject.allocation_proc_models 
    comparison_object = comparisonFunction(seedObject, ppi, **kwargs)
    stats_file = seedObject.ALLOCATION_FILEPATH + "/stats"
    allocations_file = seedObject.ALLOCATION_FILEPATH + "/allocations"
    allocation_to_ip_file = seedObject.ALLOCATION_FILEPATH + "/allocations_to_ips"
    
    with open(stats_file, 'w') as f:
        pass
    
    hits_set = set([])
    sddo = ThreadingScanDeduplicateDealiasClientSide(IP_Model.IPs, scanner_log_filepath=seedObject.ALLOCATION_FILEPATH + "/scanner.log")
    
    iteration = 0
    start = True
    
    time_start = time.time()
    current_duplicates = 0
    FirstIteration = True
    threshold_met = False
    
    total_hits = 0
    
    iteration_distribution = []
    
    combined_hits = []
    combined_allocation_dictionary = {}
    combined_allocation_list_generated_from = []
    combined_ip_list = []
    combined_aliases = []
    all_allocations_thus_far = set([])
    prior_all_allocations = 0

    send_queue = mp.Manager().Queue()
    recieve_queue = mp.Manager().Queue()
    
    
    while total_generated < numberToGenerate:
        t_start_iteration = time.time()
        print("########################################################################")
        print("Iteration: ", iteration)
        print("Generated: ", total_generated, "of", numberToGenerate)
        iteration += 1

        # Generate 1M on first round. 
        if numberToGenerate-total_generated < ppi:
            to_generate_this_round = numberToGenerate-total_generated
        elif FirstIteration == True:
            to_generate_this_round = allocation_gradient_amount 
        else: 
            to_generate_this_round = ppi

        # Generate:
        iteration_amount = per_iteration
        generated_so_far = 0
        sddo.CreateNewScanner()
        
        allocation_dictionary = {}
        allocation_list_generated_from = []
        ip_list = []

        add_batch_time = []
        generation_time = []
        allocation_setup_time = []
        filewriting_time = []
        allocation_dictionary_time = []

        list_of_iteration_dictionaries = []

        
        while generated_so_far < to_generate_this_round:
            if iteration_amount + generated_so_far > to_generate_this_round:
                iteration_amount = to_generate_this_round - generated_so_far
                
            t1 = time.time()

            addresses = comparison_object.sample(iteration_amount, random_sample_als = False, total_to_gen=to_generate_this_round)
            t2 = time.time()
            print("Time: ", t2-t1)
            generation_time.append(t2-t1)
            
            addresses_to_scan = addresses
            
            # Stats
            ip_list += addresses_to_scan
            generated_so_far += iteration_amount

            # SEND TO SCANNER
            sddo.AddBatch(addresses_to_scan, expanded=True)
            
            # ALLOCATION SAMPLING COMPUTATIONS
            
            single_iteration_allocation_dict = {}
            allocations_from_comparison = comparison_object.list_of_allocations
            for a in range(0, len(allocations_from_comparison)):
                single_iteration_allocation_dict[addresses_to_scan[a]] = allocations_from_comparison[a]
            
            allocation_list_generated_from += allocations_from_comparison
            list_of_iteration_dictionaries.append(single_iteration_allocation_dict)
            
            print("--------------")
        

        # Spawn subprocess
        sddo.StartGetResults()

        # -------------------------
        # ALLOCATION PARSING BLOCK
        # Create Dictionary
        time1 = time.time()
        allocation_dictionary = {}
        for single_iteration_allocation_dict in list_of_iteration_dictionaries:
            allocation_dictionary.update(single_iteration_allocation_dict)

        
        # Get Unique Addresses
        unique_als, al_freqs = np.unique(allocation_list_generated_from, return_counts=True)
        allocation_gen_tuple = (unique_als, al_freqs)
        t2 = time.time()
        with open(os.path.join(comparison_object.filepath, "generated"), 'a') as f:
            f.write('\n'.join(ip_list))
            f.write('\n')
        t23 = time.time()
        print(t23-time1)
        # --------------------------

        # Wait for scanner results
        hits, aliases = sddo.GetResults(send_wait=False)

        # Get results from allocation parsing

        t1 = time.time()
        with open(os.path.join(comparison_object.filepath, "dealiased"), 'a') as f:
            f.write('\n'.join(hits))
            f.write('\n')
        
        with open(os.path.join(comparison_object.filepath, "aliased"), 'a') as f:
            f.write('\n'.join(aliases))
            f.write('\n')

        with open(allocations_file, 'a') as f:
            f.write('\n'.join(allocation_list_generated_from))
            f.write('\n')

        with open(allocation_to_ip_file, 'a') as f:
            for n in range(0, len(allocation_list_generated_from)):
                f.write(allocation_list_generated_from[n] + '\t' + ip_list[n] + '\n')


        t2 = time.time()
        print(t2-t1)

        total_hits += len(hits)
            
        total_generated += to_generate_this_round
        
        with open(stats_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([iteration-1, len(hits), "None", total_hits, unique_als.shape[0], total_generated])

        if FirstIteration == True:
            # Get Diff
            for a in hits:
                all_allocations_thus_far.add(allocation_dictionary[a])
                
            diff = (len(all_allocations_thus_far) - prior_all_allocations)/len(all_allocations_thus_far)
            prior_all_allocations = len(all_allocations_thus_far)
            
            if diff < allocation_gradient_threshold:
                threshold_met = True

        # Account for Gradient Updates when updating allocations
        if FirstIteration == True and threshold_met == True:
            print("######## THRESHOLD MET #############")
            t1 = time.time()
            comparison_object.update(combined_hits, 
                                     combined_allocation_dictionary, 
                                     (combined_allocation_list_generated_from,), 
                                     combined_ip_list, 
                                     combined_aliases)
            t2 = time.time()
            print("Allocation Training Time: ", t2-t1)
        elif FirstIteration == False:
            t1 = time.time()
            comparison_object.update(hits, 
                                     allocation_dictionary, 
                                     allocation_gen_tuple, 
                                     ip_list, 
                                     aliases)
            t2 = time.time()
            print("Allocation Training Time: ", t2-t1)
        elif FirstIteration == True:
            t1 = time.time()
            combined_hits = combined_hits + hits
            combined_allocation_dictionary.update(allocation_dictionary)
            combined_allocation_list_generated_from = combined_allocation_list_generated_from + allocation_list_generated_from
            combined_ip_list = combined_ip_list + ip_list
            combined_aliases = combined_aliases + aliases
            t2 = time.time()
            print("Offline Allocation Sampling: ", t2 - t1)
        
        if FirstIteration == True and threshold_met == True:
            FirstIteration = False

        
        t_end_iteration = time.time()
        print("Iteration Time: ", t_end_iteration - t_start_iteration)
        print("Total Allocation Sampling Time: ", sum(comparison_object.allocation_gen.allocation_times))
        print("Total Upper-64 Sampling Time: ", sum(comparison_object.iidGenerator.upper_times))
        print("Total Lower-64 Sampling Time: ", sum(comparison_object.iidGenerator.lower_times))
        
        comparison_object.allocation_gen.allocation_times = []
        comparison_object.iidGenerator.lower_times  = []
        comparison_object.iidGenerator.upper_times = []

    sddo.EndProcess()    
  
    time_end = time.time()
    
    print("Total Generated: ", numberToGenerate)
    print("Total Scanning Time: ", time_end-time_start)
    

    
def Training_Evaluation(epochs, numberToGenerate, comparisonName, comparisonFunction, seedObject, ppi=5000000, asn_chart=True, asn_threshold_dups=10000, asn_threshold_hits=100, **kwargs):    
    total_generated = 0
    IP_Model = seedObject.allocation_proc_models 
    
    # CREATE MODEL
    comparison_object = comparisonFunction(seedObject, ppi, **kwargs)
    
    # TRAIN MODEL

    t1 = time.time()
    comparison_object.sidGenerator.TrainLSTM(epochs, plot=True, name=comparisonName)
    with open(os.path.join(comparison_object.filepath, comparison_object.checkpoint + "_loss"), 'w') as f:
        for i in comparison_object.sidGenerator.GENERATOR_LOSS["all_ips"]:
            f.write(str(i))
            f.write('\n')
    t2 = time.time()
    print("Train Time: ", t2-t1)






    

    