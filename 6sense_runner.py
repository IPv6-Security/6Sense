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
import sys
import numpy as np
from math import log
import math
from scipy.stats import entropy

import matplotlib.pyplot as plt
from iplibrary import AS_Processor
from Generator.SeedObject import SeedObject
from Generator.RunGeneration import Training_Evaluation, ComparisonModel_online
from Generator.Generator import Generator_AUL
from Generator.ModelWrapper import ModelBase
from Generator.AdditionalGenerators import  IterativeOnPriorPatternsLowerGeneratorFaster, FixedLowBit
from Generator.AllocationGenerator import NaiveWeightUpdateWithoutZerosEvenFirst
from Generator.NN_models import GeneratorMaskedLSTM
from Generator.DatasetEncoders import AllocationEncoder
from Generator.Sampling import predict_base_function, predict_base_function_faster
from Generator.MultiProcessSIDGenerator import MultiProcessSIDGenerator
from Generator.MultiProcessIIDGenerator import MultiProcessIIDGenerator, MultiProcessSamplingIIDGenerator

import config as conf


def loadData(Comparison_name):
    t1 = time.time()
    seedDatasetObject = SeedObject(Comparison_name, 
                                   sid_checkpoint=conf.CHECKPOINT_LSTM, 
                                   dataset_prefix=conf.DATASET_FILE, 
                                   lower_names_to_use=["all_ips"], 
                                   lower=True, 
                                   prefix_filename=conf.UPDATED_PFX_AS_FILE)
    t2 = time.time()
    print("Seed Dataset Time: ", t2-t1)
    
    return seedDatasetObject

def run6Sense(Comparison_name, seedDatasetObject, budget, ppi, gradient_threshold):
    t1 = time.time()
    
    # Positive Allocation Sampling
    c = ComparisonModel_online(budget, 
                                Comparison_name, 
                                Generator_AUL,
                                seedDatasetObject, 
                                ppi=ppi,
                                per_iteration=100000,
                                allocation_gradient_threshold=gradient_threshold,
                                Lower64=MultiProcessIIDGenerator,
                                Upper64=MultiProcessSIDGenerator,
                                Allocations=NaiveWeightUpdateWithoutZerosEvenFirst,
                                Upper64_HPs={"sampler":predict_base_function_faster, 
                                            "model":GeneratorMaskedLSTM,
                                            "sampling_batch_size":10500,
                                            "gpus":4,
                                            "lr":1e-3, 
                                            "dropout":0.2, 
                                            "layers":[512, 256], 
                                            "encoder":AllocationEncoder, 
                                            "preload":True, 
                                            "validation_split":0.15},                         
                                Lower64_HPs={"Allocations":seedDatasetObject.allocation_proc_models.allocation_strings,
                                            "subprocesses":40},                            
                                Allocation_HPs={"threshold":0.5, 
                                                "observation_window":3},
                                )

    t2 = time.time()
    print("Time: ", t2-t1)
    print("Time per IP:", (t2-t1)/100000000)

    
    
if __name__=="__main__":
    
    print(sys.argv)
    
    if len(sys.argv) < 2:
        raise ValueError("Please provide a scan budget, ppi, and gradient threshold.")
        
    try:
        int(sys.argv[1])
    except ValueError:
        raise ValueError("Please provide a valid integer scan budget")

    try:
        int(sys.argv[2])
    except ValueError:
        raise ValueError("Please provide a valid integer ppi")

    try:
        float(sys.argv[3])
    except ValueError:
        raise ValueError("Please provide a valid gradient threshold")
    
    
    print("Begin")
    Comparison_name = "6SENSE_SCAN"

    seedDatasetObject = loadData(Comparison_name)
    print("completed data loading")
    run6Sense(Comparison_name, seedDatasetObject, int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
    