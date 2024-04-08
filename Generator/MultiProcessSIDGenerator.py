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
import math


from Generator.ModelWrapper import ModelBase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras


class MultiProcessSIDGenerator():
    def __init__(self, 
                 dataset, 
                 PriorLevel, 
                 hps, 
                 *args):
        """
        Multiprocess Upper-64 Sampling. Currently only configured to work with Allocation Sampler beforehand. 

        Args:
            dataset (_type_): _description_
            PriorLevel (_type_): _description_
            hps (_type_): _description_
        """

        self.First = True
        self.generated_thus_far = 0
        self.prior_allocations = []

        self.number_of_gpus = 1
        if "gpus" in hps:
            self.number_of_gpus = hps["gpus"]
        # HPs:
        self.SEQUENCE_LENGTH = 16
        if "seq_length" in hps:
            self.SEQUENCE_LENGTH = hps["seq_length"]
        self.START_COLUMN = self.SEQUENCE_LENGTH

        # Maximum Length to keep of each dataset row:
        self.MAXIMUM_LENGTH = 32
        if "max_length" in hps:
            self.MAXIMUM_LENGTH = hps["max_length"]
        
        self.UPPER_SEQUENCE_GENERATION_FUNCTION = PriorLevel

        self.allocation_gen = {}
        if "AllocationObject" in hps:
            self.allocation_gen = hps["AllocationObject"]

        self.all_allocations = []

        # Trigger Generators
        self.output_queue = []
        self.commands_queue = []
        self.generated_queue = []
        self.threads = []
        for gpu_number in  range(0,  self.number_of_gpus):
            input_gpu_number = gpu_number % 8
            self.output_queue.append(mp.Manager().Queue())
            self.commands_queue.append(mp.Manager().Queue())
            self.generated_queue.append(mp.Manager().Queue())
            self.threads.append(mp.Process(target=MultiProcessSIDClient, args=(dataset, 
                                    PriorLevel, 
                                    hps, 
                                    self.commands_queue[-1], 
                                    self.generated_queue[-1], 
                                    self.output_queue[-1],
                                    input_gpu_number)))
            time.sleep(9)
            self.threads[-1].start()
        
        setup_models = 0
        while setup_models < self.number_of_gpus:
            cmd = self.output_queue[setup_models].get()
            if cmd == "setup":
                print("GPU ", setup_models, "Setup")
                setup_models += 1


    def sample(self, numberToGenerate, plot=False, total=None):
        """
        Sample from model.
        """

        if self.First == True:
            ##############
            # Sample from Allocation Sampler and Setup Padding - A1
            ##############
            sequences = np.zeros((numberToGenerate, self.MAXIMUM_LENGTH - self.START_COLUMN + self.SEQUENCE_LENGTH))

            upper = self.UPPER_SEQUENCE_GENERATION_FUNCTION(numberToGenerate, "None")

            if self.SEQUENCE_LENGTH != 16:
                mask = (upper==0)
                current_upper_sequence_lengths = np.where(mask.any(1), mask.argmax(1), upper.shape[1])
            else:
                current_upper_sequence_lengths = (np.ones((sequences.shape[0]))*self.SEQUENCE_LENGTH).astype(int)

            sequences[:, 0:upper.shape[1]] = upper


            ####################
            # Sample Upper64s - U1
            ####################

            block_size = int(sequences.shape[0]/self.number_of_gpus)
            # Trigger Sampling
            starting_point = 0
            for sampler in range(0, self.number_of_gpus):
                self.commands_queue[sampler].put("sample")
                if sampler <  self.number_of_gpus-1:
                    t = time.time()
                    self.generated_queue[sampler].put(sequences[starting_point:starting_point+block_size,:])
                    self.generated_queue[sampler].put(current_upper_sequence_lengths[starting_point:starting_point+block_size])
                    starting_point = starting_point + block_size
                else:
                    self.generated_queue[sampler].put(sequences[starting_point:,:])
                    self.generated_queue[sampler].put(current_upper_sequence_lengths[starting_point:])
            
            self.generated_thus_far += numberToGenerate

            if total is not None:
                self.First = False

            self.prior_allocations = self.allocation_gen.sampled_allocations


        if total is not None and self.generated_thus_far < total:
            # Generate All Allocations in the meantime
            to_gen = numberToGenerate
            if total - self.generated_thus_far < numberToGenerate:
                to_gen = total - self.generated_thus_far

            # Gen Next set of allocations
            initial_sampling_allocations = self.prior_allocations
            second_uppers = self.UPPER_SEQUENCE_GENERATION_FUNCTION(to_gen, "None")
            self.prior_allocations = self.allocation_gen.sampled_allocations
            self.allocation_gen.sampled_allocations = initial_sampling_allocations

            second_sequences_full = np.zeros((to_gen, self.MAXIMUM_LENGTH - self.START_COLUMN + self.SEQUENCE_LENGTH))
            if self.SEQUENCE_LENGTH != 16:
                mask = (second_uppers==0)
                second_upper_sequence_lengths_full = np.where(mask.any(1), mask.argmax(1), second_uppers.shape[1])
            else:
                second_upper_sequence_lengths_full = (np.ones((second_sequences_full.shape[0]))*self.SEQUENCE_LENGTH).astype(int)

            second_sequences_full[:, 0:second_uppers.shape[1]] = second_uppers

            # Send to Upper-64 Generator
            block_size = int(second_sequences_full.shape[0]/self.number_of_gpus)
            # Trigger Sampling
            starting_point = 0
            for sampler in range(0, self.number_of_gpus):
                self.commands_queue[sampler].put("sample")
                if sampler <  self.number_of_gpus-1:
                    t = time.time()
                    self.generated_queue[sampler].put(second_sequences_full[starting_point:starting_point+block_size,:])
                    self.generated_queue[sampler].put(second_upper_sequence_lengths_full[starting_point:starting_point+block_size])
                    starting_point = starting_point + block_size
                else:
                    self.generated_queue[sampler].put(second_sequences_full[starting_point:,:])
                    self.generated_queue[sampler].put(second_upper_sequence_lengths_full[starting_point:])
            
            self.generated_thus_far += to_gen
        elif total is not None and self.generated_thus_far >= total:
            self.generated_thus_far = 0
            self.First = True




        # Get Next Block   
        single_sequence_list = []
        for sampler in range(0,  self.number_of_gpus):
            while True:
                cmd = self.output_queue[sampler].get()
                if cmd == "returned":
                    single_sequences = self.output_queue[sampler].get()
                    single_sequence_list.append(single_sequences)
                    break

        single_sequences = single_sequence_list[0]
        for s in range(1, len(single_sequence_list)):
            single_sequences = np.concatenate((single_sequences, single_sequence_list[s]))

        return single_sequences
            
    
    def update(self, hits, *args):
        """
        Spot for updating the model during training is needed. 
        """
        pass



def MultiProcessSIDClient(dataset, 
                 PriorLevel, 
                 hps, commands_queue, generated_queue, output_queue, gpu_number):
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        for p in physical_devices:
            tf.config.experimental.set_memory_growth(p, enable=True)            
        os.environ['TF_ GPU_ALLOCATOR'] = 'cuda_malloc_async'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        
        generator = ModelBase(dataset=dataset, PriorLevel=PriorLevel, hps=hps)

        output_queue.put("setup")

        while True:
            cmd = commands_queue.get()
            if cmd == "end":
                break
            elif cmd == "sample":
                # read from gen queue until end of line
                # Send to AddBatch
                sequences = generated_queue.get()
                current_upper_sequence_lengths = generated_queue.get()
                single_sequences = generator.generate_from_input(sequences, current_upper_sequence_lengths)                
                output_queue.put("returned")
                output_queue.put(single_sequences)


