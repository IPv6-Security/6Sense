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
import time as time
import datetime
import ipaddress
import pickle

import numpy as np

from iplibrary import Faster_IP_Processor, AS_Processor
from Generator.cython_print_ips import print_ips

import config as conf

def CreateFileStructure(prefix, name):
    evaluation_path = os.path.join(prefix, 'evaluation')
    if not os.path.isdir(evaluation_path):
        os.mkdir(evaluation_path)
        
    date_timestamp = str(datetime.datetime.now())
    fn1 = name + ' ' + date_timestamp
    fn2 = fn1.replace(" ","")
    folder_name = fn2.replace(":","")
    folder_path = os.path.join(evaluation_path, folder_name)
    os.mkdir(folder_path)
    
    final_names = []
    for subname in ["Allocation", "SSID", "SSID_Sampling", "IID", "FULL_ADDRESS"]:
        final_names.append(os.path.join(folder_path, subname))
        os.mkdir(final_names[-1])

    return final_names[0], final_names[1], final_names[2], final_names[3], final_names[4], folder_path


class SeedObject():
    def __init__(self, name, prefix = conf.HOME_DIRECTORY, dataset_prefix="new_split/", allocation_masking_length = 16, lower_names_to_use =  ['ipv4_embedding', 'unknown', 'MAC', 'lower_pattern', 'low_zeros',  'low_bit'], upper_names_to_use=["all_ips"], full_names_to_use=["all_ips"], full_address=False, upper=True, lower=False, iid_checkpoint=None, expand=False,sid_checkpoint=None, full_checkpoint=None, checkpoint_folder=conf.MODEL_CHECKPOINTS,prefix_filename=conf.UPDATED_PFX_AS_FILE, readall = True, recreate=False):
        """
        Load All seed Datasets
        """
        t1 = time.time()
        self.name = name
        self.prefix = prefix
        # CREATE EVAL FILEPATH
        self.ALLOCATION_FILEPATH, self.SSID_FILEPATH, self.SSID_SAMPLING_FILEPATH, self.IID_FILEPATH, self.FULL_ADDRESS_FILEPATH, self.FOLDER = CreateFileStructure(self.prefix, self.name)
        print(self.ALLOCATION_FILEPATH)
        print(self.SSID_FILEPATH)
        print(self.SSID_SAMPLING_FILEPATH)
        print(self.IID_FILEPATH)
        print(self.FOLDER)
        
        # CREATE SAVE/LOAD and DATASET Filepaths
        lower_checkpoints = prefix + checkpoint_folder
        
        if sid_checkpoint != None:
            lower_checkpoints = lower_checkpoints + sid_checkpoint + "/"
            if not os.path.exists(lower_checkpoints):
                os.mkdir(lower_checkpoints)
        
        lower_path = prefix + dataset_prefix
        self.all_lower_names = lower_names_to_use
        self.all_upper_names = upper_names_to_use
        self.all_full_names = full_names_to_use
        t2 = time.time()
    
        # ALLOCATION DATASET
        t1 = time.time()
        print("Processing: ALLOCATIONS", self.all_upper_names[0])

        # Load Allocations and IPs
        self.allocation_proc_models = Faster_IP_Processor(lower_path + self.all_upper_names[0], recreate=recreate, deduplicate=False,expand=expand,prefix_filename=prefix_filename)

        if readall == False or os.path.isfile(conf.HOME_DIRECTORY + conf.DATASET_FILE + "all_ips.allocations.masked.npy") == False:
            scaled_allocation_lengths = np.floor(np.array(self.allocation_proc_models.allocation_lengths)/4).astype(int)

            scaled_allocation_lengths[scaled_allocation_lengths < 8] = 8
            scaled_allocation_lengths[scaled_allocation_lengths > 16] = 16
            
            self.allocation_dataset = np.zeros((self.allocation_proc_models.datasets["int_full_data"].shape[0], allocation_masking_length))
            
            for a in range(0, self.allocation_proc_models.datasets["int_full_data"].shape[0]):
                length = scaled_allocation_lengths[a]
                self.allocation_dataset[a, :length] = self.allocation_proc_models.datasets["int_full_data"][a, :length]+1
            
            self.allocation_dataset /= 16
            with open(conf.HOME_DIRECTORY + conf.DATASET_FILE + "all_ips.allocations.masked.npy", 'wb') as f:
                np.save(f, self.allocation_dataset)
        else:
            self.allocation_dataset = np.load(conf.HOME_DIRECTORY + conf.DATASET_FILE + "all_ips.allocations.masked.npy")

        t2 = time.time()
        print("Allocation Time: ", t2-t1)
                
        # SID DATASET
        t1 = time.time()
        self.sid_datasets = {}
        self.sid_generator_checkpoints = {}
        self.sid_files = self.all_upper_names
        self.sid_sets = {}
        if upper == True:
            if sid_checkpoint is None:
                generator_checkpoint = lower_checkpoints + "lstm_sid_allocation_base_100_" 
            else:
                generator_checkpoint = lower_checkpoints + sid_checkpoint

            discriminator_checkpoint = lower_checkpoints + "cnn_sid_allocation_base_100_"
            self.sid_discriminator_checkpoint = discriminator_checkpoint

            for l in self.all_upper_names:
                print("Processing: SIDs", l)
                sid_proc_model = self.allocation_proc_models
                dataset = sid_proc_model.datasets['int_full_data'][:, 0:16]
                
                upper_ips = set([])
                index = []
                ind_iter = 0
                for i in sid_proc_model.IPs:
                    ip_to_add_to_set = i[:19] + "::"
                    if ip_to_add_to_set not in upper_ips:
                        index.append(ind_iter)
                    upper_ips.add(ip_to_add_to_set)
                    ind_iter += 1
                self.sid_sets[l] = upper_ips 

                
                unique_dataset = dataset[index, :]
                self.sid_datasets[l] = unique_dataset
                print("Unique SSID Dataset Shape: ", unique_dataset.shape)
                self.sid_generator_checkpoints[l] = generator_checkpoint + l 
                
        t2 = time.time()
        print("SSID Time: ", t2 - t1)        
        
        # IID DATSET
        self.iid_datasets = {}
        self.iid_generator_checkpoints = {}
        self.iid_files = self.all_lower_names
        self.iid_sets = {}
        self.iid_proc_models = {}
        if lower== True:
            if iid_checkpoint is None:
                generator_checkpoint = lower_checkpoints + "lstm_lower_full_encoding_ORGAN_100_"
            else:
                generator_checkpoint = lower_checkpoints + iid_checkpoint

            discriminator_checkpoint = lower_checkpoints + "cnn_lower_full_encoding_ORGAN_100"

            self.iid_discriminator_checkpoint = discriminator_checkpoint
            if len(self.all_lower_names) == 1 and self.all_lower_names[0] == "all_ips":
                t1 = time.time()
                l = self.all_lower_names[0]
                print("Processing: IIDs", l)
                # Single Lower-64 File
                iid_proc_model = self.allocation_proc_models
                dataset = iid_proc_model.datasets['int_full_data']
                self.iid_datasets[l] = dataset
                self.iid_generator_checkpoints[l] = generator_checkpoint + l 
                self.iid_sets[l] = set(iid_proc_model.IPs)
                self.iid_proc_models[l] = iid_proc_model
                t2 = time.time()
                print("Lower Timing: ", t2-t1)
                
            else:
                # Multiple Separate Lower-64 Files
                for l in self.all_lower_names:
                    print("Processing: IIDs", l)
                    iid_proc_model = Faster_IP_Processor(lower_path + l, recreate=recreate, deduplicate=False,prefix_filename=prefix_filename)
                    dataset = iid_proc_model.datasets['int_full_data']
                    self.iid_datasets[l] = dataset
                    self.iid_generator_checkpoints[l] = generator_checkpoint + l 
                    self.iid_sets[l] = set(iid_proc_model.IPs)
                    self.iid_proc_models[l] = iid_proc_model
        

                