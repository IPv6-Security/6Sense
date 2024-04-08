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

import numpy as np
import os, pwd
import time
import multiprocessing as mp
import random

from Generator.cython_print_ips import print_ips



from Generator.AdditionalGenerators import IterativeOnPriorPatternsLowerGeneratorFaster, integer_extraction, powers_hash



class MultiProcessIIDGenerator():
    """
    """
    def __init__(self, dataset, PriorLevel, hps, allocation_gen, **kwargs):
        self.upper_times = []
        self.lower_times = []
        self.upper_generator = PriorLevel
        self.allocation_gen = allocation_gen

        als = np.array(hps['Allocations'])
        self.subprocesses = hps['subprocesses']

        # Get Allocations
        self.allocation_list = list(set(hps['Allocations']))
        self.allocation_mapping = {}
        
        index = 0
        for al in self.allocation_list:
            self.allocation_mapping[al.split('/')[0]] = index % self.subprocesses 
            index += 1
        
        # Split by Allocation
        allocation_of_seed_allocations = []
        for al in hps['Allocations']:
            allocation_of_seed_allocations.append(self.allocation_mapping[al.split('/')[0]])
        
        allocation_of_seed_allocations = np.array(allocation_of_seed_allocations)


        datasets = {}
        new_hps = {}
        for x in range(0, self.subprocesses):
            subprocess_specific_ips = dataset['all_ips'][allocation_of_seed_allocations==x, :]
            datasets[x] = {'all_ips':subprocess_specific_ips}
            subprocess_specific_allocations = np.array(hps['Allocations'])[allocation_of_seed_allocations==x]
            new_hps[x] = {'Allocations':subprocess_specific_allocations}
        
         # Trigger Generators
        self.output_queue = []
        self.commands_queue = []
        self.generated_queue = []
        self.threads = []
        for x in range(0, self.subprocesses):
            self.output_queue.append(mp.Manager().Queue())
            self.commands_queue.append(mp.Manager().Queue())
            self.generated_queue.append(mp.Manager().Queue())
            self.threads.append(mp.Process(target=MultiProcessIIDClient, args=(datasets[x], PriorLevel, new_hps[x], allocation_gen, self.commands_queue[x], self.generated_queue[x], self.output_queue[x])))
            self.threads[-1].start() 

        # Wait for everyone to be setup
        setup_models = 0
        while setup_models < self.subprocesses:
            cmd = self.output_queue[setup_models].get()
            if cmd == "setup":
                print("Subprocess", setup_models, "Setup")
                setup_models += 1
    
        

    
    def sample(self, number_to_generate, total=None):

        # GET UPPER-64s
        t1 = time.time()
        sampled = self.upper_generator(number_to_generate, "all_ips", total) # Numpy
        t111 = time.time()
        self.upper_times.append(t111-t1)
        allocation_list = self.allocation_gen.sampled_allocations
 
        upper64_to_subprocess_mapping = []
        a_not_in_mapping = 0
        nnn = -1
        for a in allocation_list:
            nnn += 1
            upper64_to_subprocess_mapping.append(self.allocation_mapping[a])

        upper64_to_subprocess_mapping = np.array(upper64_to_subprocess_mapping)


        sampled_mapping = {}
        allocation_list_mapping = {}
        allocation_list_np = np.array(allocation_list)
        for x in range(0, self.subprocesses):
            sampled_mapping[x] = sampled[upper64_to_subprocess_mapping==x, :]
            allocation_list_mapping[x] = allocation_list_np[upper64_to_subprocess_mapping==x]


        # Run Multiple IID Samplers
        for sampler in range(0, self.subprocesses):
            self.commands_queue[sampler].put("sample")
            self.generated_queue[sampler].put(sampled_mapping[sampler])
            self.generated_queue[sampler].put(allocation_list_mapping[sampler])
            
        addresses_to_return = []
        for sampler in range(0,  self.subprocesses):
            while True:
                cmd = self.output_queue[sampler].get()
                if cmd == "returned":
                    returned_gen_addresses = self.output_queue[sampler].get()
                    addresses_to_return.append(returned_gen_addresses)
                    break

        addresses = []
        current_indices = {}
        for x in range(0, self.subprocesses):
            current_indices[x] = 0
        for n in range(0, number_to_generate):
            addresses.append(addresses_to_return[upper64_to_subprocess_mapping[n]][current_indices[upper64_to_subprocess_mapping[n]]])
            current_indices[upper64_to_subprocess_mapping[n]] += 1

        t2 = time.time()
        self.lower_times.append(t2-t111)

        return addresses
    
    def update(self, *args):
        pass   



def MultiProcessIIDClient(dataset, PriorLevel, hps, allocation_gen, commands_queue, generated_queue, output_queue):
        
        generator = MultiProcessSamplingIIDGenerator(dataset, PriorLevel, hps, allocation_gen)

        output_queue.put("setup")

        while True:
            cmd = commands_queue.get()
            if cmd == "end":
                break
            elif cmd == "sample":

                t = time.time()
                sampled = generated_queue.get()
                allocation_list = generated_queue.get()

                addresses = generator.iid_sampling(sampled, allocation_list)
                
                output_queue.put("returned")
                output_queue.put(addresses)

                t2 = time.time()







class MultiProcessSamplingIIDGenerator(IterativeOnPriorPatternsLowerGeneratorFaster):
    """
    1. Generate from Known Patterns.
    2. Incremement index by 1
    3. Generate from Known Patterns + index
    4. Repeat (indefinitely?)
    5. Generate Duplicate?? But probably never reach this point. Could we even?
    """
    def __init__(self, dataset, PriorLevel, hps, allocation_gen, **kwargs):
       super().__init__(dataset, PriorLevel, hps, allocation_gen, **kwargs)
        
    
    def sample(self, number_to_generate, total=None):
        # GET UPPER-64s
        t1 = time.time()
        sampled = self.upper_generator(number_to_generate, "all_ips", None) # Numpy
        t111 = time.time()
        print("Upper 64 Sampling Time: ", t111-t1)
        allocation_list = np.copy(self.allocation_gen.sampled_allocations)

        addresses_to_return = self.iid_sampling(sampled, allocation_list)

        t2 = time.time()
        print("Lower Sampling Time: ", t2-t111)

        return addresses_to_return

    def iid_sampling(self, sampled, allocation_list):
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
            allocation = allocation_list[i]
            
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

        addresses_to_return = print_ips(addresses_final.astype(int))


        
        return addresses_to_return
    
    def update(self, *args):
        pass   