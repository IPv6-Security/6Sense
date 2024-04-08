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
import math

from Generator.ModelWrapper import ModelBase
from Generator.AdditionalGenerators import IterativeOnPriorPatternsLowerGeneratorFaster
from Generator.AllocationGenerator import NaiveWeightUpdateWithoutZerosEvenFirst
from Generator.NN_models import GeneratorMaskedLSTM
from Generator.DatasetEncoders import AllocationEncoder

from Generator.cython_print_ips import print_ips


class Generator_AUL():
    """
    Generator with the Following Components
    Fixed (Random) Allocation
    LSTM Upper
    LSTM Lower
    """
    def __init__(self, 
                 si, 
                 ppi,
                 Lower64=IterativeOnPriorPatternsLowerGeneratorFaster,
                 Upper64=ModelBase,
                 Allocations=NaiveWeightUpdateWithoutZerosEvenFirst,
                 Upper64_HPs={"model":GeneratorMaskedLSTM, "lr":1e-3, "dataset_encoder":AllocationEncoder,},
                 Lower64_HPs={"string":"::1"},
                 Allocation_HPs={"threshold":0.5},):
        
        # Create Allocation Generator
        t1 = time.time()
        self.allocation_gen = Allocations(si, Allocation_HPs["threshold"], ppi, si.ALLOCATION_FILEPATH, Allocation_HPs["observation_window"])
        self.list_of_allocations = []
        t2 = time.time()
        print("Allocation Setup Time: ", t2-t1)
        
        print("Sampled Allocations: ")
        for a in self.allocation_gen.sample(10):
            print(a)

        t3 = time.time()
        # Create Upper-64 bit Generator
        Upper64_HPs["seq_length"] = 8
        Upper64_HPs["max_length"] = 16
        Upper64_HPs["checkpoint"] = si.sid_generator_checkpoints
        Upper64_HPs["AllocationObject"] = self.allocation_gen
        
        allocation_function = lambda x, c : self.allocation_gen.sample(x) #[0] #, c)[0]

        self.sidGenerator = Upper64(dataset=si.sid_datasets, PriorLevel=allocation_function, hps=Upper64_HPs) #, filepath=si.SSID_FILEPATH, allocation_gen = self.allocation_gen)
        t4 = time.time()
        print("Upper-64 Setup Time: ", t4-t3)
        
        t5 = time.time()
        # Create Lower-64 bit Generator
        Lower64_HPs["seq_length"] = 16
        Lower64_HPs["max_length"] = 32
        Lower64_HPs["checkpoint"] = si.iid_generator_checkpoints,
        
        upper_function = lambda x, c, t : self.sidGenerator.sample(x, total=t)
        self.iidGenerator = Lower64(dataset=si.iid_datasets, PriorLevel=upper_function, hps=Lower64_HPs, allocation_gen=self.allocation_gen)
        
        t6 = time.time()
        print("Lower 64 Setup Time: ", t6-t5)

        
        self.checkpoint = si.sid_generator_checkpoints["all_ips"]
        self.filepath = si.SSID_FILEPATH
        
    def update(self, hits, allocation_dictionary, allocation_list_generated_from, ip_list, aliases):
        self.allocation_gen.update(hits, allocation_dictionary, allocation_list_generated_from, aliases, ip_list)
        self.sidGenerator.update(hits, ip_list)
        self.iidGenerator.update(hits)
                
    def sample(self, to_generate_this_round, randomness=False, random_sample_als = True, total_to_gen=None):
        if random_sample_als == True:
            self.allocation_gen.First = random_sample_als
            
        addresses = self.iidGenerator.sample(to_generate_this_round, total=total_to_gen)
        self.list_of_allocations = self.allocation_gen.sampled_allocations

        return addresses
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    