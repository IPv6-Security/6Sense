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

# important
import os
from os.path import exists
import time
import math
from math import log
import ipaddress
import random
import multiprocessing as mp



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import hamming
import pandas as pd
import pyasn
import pytricia
import pickle

import config as conf



"""
Simple Little Helper Function
"""

def ExpandIPs(ips):
    expanded_ips = []
    for d in ips:
        try:
            ip = ipaddress.ip_address(d)
            ip_exploded = ip.exploded
            expanded_ips.append(ip_exploded)
        except ValueError:
            print("Invalid IP: ", d)
        
    return expanded_ips

"""
AS_Processor:
A Class for processing AS Numbers of IP addresses
"""

# small helper function for .dat files to pytricia
def subnetTreeFromDat(fpath):
    tree = pytricia.PyTricia(128)
    with open(fpath, 'rt') as f:
        for l in f:
            pts = l.split(' ')
            pfx = pts[0].strip()
            val = pts[1].strip()
            try:
                tree[pfx] = val
            except ValueError as e:
                print("skipped " + pfx, file=sys.stderr)

    return tree

class AS_Processor:

    def __init__(self, inputIPs,prefix_filename=conf.UPDATED_PFX_AS_FILE):
        self.AS_numbers_Labels = []
        self.Allocation_Exists = []
        self.frequency_sorted = []
        self.number_sorted = []
        self.total_times_repeated = []
        self.total_number = []
        self.all_labels = []
       
        print(prefix_filename)
        asndb = pyasn.pyasn(prefix_filename)
        
        if type(inputIPs) == list:
            # input data is an IP list
            data = inputIPs
        else:
            # input data is an IP file
            with open(inputIPs, 'r') as f:
                data = f.read().splitlines()
        
        self.inputIPsForASN = inputIPs

        AS_numbers = np.zeros((len(data)))
        Allocations = []
        Allocation_strings = []
        Allocation_lengths = []
        for a in range(0,len(data)):
            try:
                lookup = asndb.lookup(data[a])
            except:
                print(data[a])
                raise ValueError
            AS_numbers[a] = lookup[0]
            if AS_numbers[a] != None and  AS_numbers[a] != np.nan:
                try:
                    AS_numbers[a] = int(AS_numbers[a])
                    Allocations.append((lookup[1], lookup[0]))
                    Allocation_strings.append(lookup[1])
                    Allocation_lengths.append(int(lookup[1].split("/")[1]))
                    
                    self.Allocation_Exists.append(True)

                    
                    
                except ValueError:
                    AS_numbers[a] = 0
                    Allocations.append((data[a][:10] + ":/32", 0))
                    Allocation_strings.append(data[a][:10] + ":/32")
                    Allocation_lengths.append(32)
                    self.Allocation_Exists.append(False)
            else:
                AS_numbers[a] = 0
                Allocations.append((data[a][:10] + ":/32", 0))
                Allocation_strings.append(data[a][:10] + ":/32")
                Allocation_lengths.append(32)
                self.Allocation_Exists.append(False)

        self.AS_numbers_int = AS_numbers.astype(int) 
        self.AS_numbers_norm = self.AS_numbers_int/200000
        
        self.full_allocations = Allocations
        self.allocations = list(set(Allocations))
        self.allocation_strings = Allocation_strings
        self.allocation_lengths = Allocation_lengths
        

def split_int_ips(IPs, q1):
    integer_ips = []
    for a in IPs:
        str_ip = a
        str_ip = str_ip.replace(':', '')
        list_ip = list(str_ip)
        str_ip = [int(i, 16) for i in list_ip]
        integer_ips.append(str_ip)
        
    int_IP_data = np.zeros((len(integer_ips), 32))
    for i in range(0, len(integer_ips)):
        int_IP_data[i] = integer_ips[i]

    q1.put(int_IP_data)


"""
IP_Processor:
A Class for processing/manipulating IP addresses
"""
class Faster_IP_Processor(AS_Processor):

    def __init__(self, inputIPs = None, recreate=False, filename = None, save_to_file = False, deduplicate=True, debug=False,expand=False,prefix_filename=conf.UPDATED_PFX_AS_FILE):
        """
        The Multipurpose visualization and base library for IP address manipulation.
        
        PARAMETERS:
            inputIPs: Either a string of the filename of IPs, or another IP_Processor object (in which case all variables are copied in), or a list of IP addresses.  
            recreate: Should the integer, binary, and hex files be recreated if they already exist and we want to store IPs to a file?. (Default False)
            filename: If the input IPs are a list, a filename is needed. 
            save_to_file: Whether IP addresses shold be saved to a file to speed up future load time (set to True only when you plan to reuse a file many times to look at the same IPs). This will also work if inputIPs is a filename (it will just save the IPs to a new file specified in filename, or only save the integer and binary representatiosn if no filename is given).
            deduplicate: Whether the IPs should be deduplicated.
            debug: Whether to print debugging information
        """

       #print("START")
        
        
        # GET INPUT TYPES AND SET OVERALL NAME
        if type(inputIPs) == str and os.path.isfile(inputIPs):
            self.input_type = "file"
            self.filename_original = inputIPs
            if debug:
                print("IP FILE: ", self.filename_original)
        elif type(inputIPs) == str and not os.path.isfile(inputIPs):
            raise ValueError('Invalid Filepath. File not found. You\'re not in Kansas anymore :(', inputIPs)
        else:
            raise ValueError('Invalid input type. Accepted Types: "IP_Processor" object, "list" of IPs, or "str" filename. Times New Roman is highly discouraged.')
        
        # SET FILE NAMES
        self.int_filename = self.filename_original + "_integer_values"
        
        # PARSE FILES (IF THEY EXIST)
        # Read IP Addresses from file
        start = time.time()
        with open(inputIPs, 'r') as f:
            InitialIPs = f.read().splitlines()

        if debug:
            print("Dataset Length:", len(InitialIPs))
            end = time.time()
            print("Time: ", end-start)
            
            
        # EXPLODE ALL STRING IPS
        if expand == True:
            expandedIPs = []
            self.compressedIPs = []
            self.incorrect_ips = 0

            counter = 0
            expand_time = time.time()

            for ip_a in InitialIPs:
                if debug:
                    counter += 1
                    if counter % 100000 == 0:
                        print("Current: ", counter)
                        print("Time: ", time.time() - expand_time)
                try:
                    ip = ipaddress.ip_address(ip_a)
                    expandedIPs.append(ip.exploded)
                    self.compressedIPs.append(ip.compressed)
                except ValueError:
                    self.incorrect_ips += 1
                    #print("Not an IP address: " + ip_a)
                
            self.IPs = expandedIPs
        else:
            self.IPs = InitialIPs
        
        # INTEGER IP PROCESSING
        if self.input_type == "file" and recreate == False and os.path.isfile(self.int_filename):
            int_IP_data = np.load(self.int_filename) #np.genfromtxt(self.int_filename,delimiter=',')
        else:
            start = time.time()
            #q1 = mp.Manager().Queue()
            #p1 = []
            #subprocesses = 20

            #for x in range(0, subprocesses - 1):
            #    beginning = int(x*len(self.IPs)/subprocesses)
            #    end = int((x+1)*len(self.IPs)/subprocesses)
            #    p1.append(mp.Process(target=split_int_ips, args=(self.IPs[beginning:end], q1)))
            #    p1[-1].start()

            #p1.append(mp.Process(target=split_int_ips, args=(self.IPs[end:], q1)))
            #p1[-1].start()
            
            #ip_arrays = []
            #for x in range(0, subprocesses):
            #    ip_arrays.append(q1.get())
            #    p1[x].join()

            #int_IP_data = np.concatenate(ip_arrays)
            
            integer_ips = []
            for a in self.IPs:
                str_ip = a
                str_ip = str_ip.replace(':', '')
                list_ip = list(str_ip)
                str_ip = [int(i, 16) for i in list_ip]
                integer_ips.append(str_ip)
            
            int_IP_data = np.zeros((len(integer_ips), 32))
            for i in range(0, len(integer_ips)):
                int_IP_data[i] = integer_ips[i]
                
            end = time.time()
            if debug:
                print("Integer Parse Time: ", end-start)

            with open(self.int_filename, 'wb') as f:
                np.save(f, int_IP_data)
            #np.savetxt(self.int_filename, int_IP_data, delimiter = ",")
            #print("Saved to File")

            

        # MAKE THE UPPER AND LOWER SPLITS
        # integer
        middle = 16
        int_upper_IP_data = int_IP_data[:, :int(middle)]
        int_lower_IP_data = int_IP_data[:, int(middle):]
        
        
        # INITIALIZE THE AS PROCESSOR
        names_to_write = [".Allocation_Exists", ".AS_numbers_int", ".full_allocations", ".allocations", ".allocation_strings", ".allocation_lengths"]
        files_exist = 0
        for n in names_to_write:
            if os.path.isfile(self.filename_original + n):
                files_exist += 1
        
        if recreate == False and files_exist == len(names_to_write):
            print("Loading Allocations from File")
            with open(self.filename_original + names_to_write[0], 'rb') as g:
                self.Allocation_Exists = pickle.load(g)
            with open(self.filename_original + names_to_write[1], 'rb') as g:
                self.AS_numbers_int = pickle.load(g)
            with open(self.filename_original + names_to_write[2], 'rb') as g:
                self.full_allocations = pickle.load(g)
            with open(self.filename_original + names_to_write[3], 'rb') as g:
                self.allocations = pickle.load(g)
            with open(self.filename_original + names_to_write[4], 'rb') as g:
                self.allocation_strings = pickle.load(g)
            with open(self.filename_original + names_to_write[5], 'rb') as g:
                self.allocation_lengths = pickle.load(g)
        else:
            AS_Processor.__init__(self, self.IPs,prefix_filename = prefix_filename)
            values_to_write = [self.Allocation_Exists, 
                            self.AS_numbers_int, 
                            self.full_allocations, 
                            self.allocations, 
                            self.allocation_strings, 
                            self.allocation_lengths]
            for v in range(0, len(values_to_write)):
                with open(self.filename_original + names_to_write[v], 'wb') as f:
                    pickle.dump(values_to_write[v], f)
        
        # CREATE OBJECT OF COMMONLY USED PERMUTATIONS. (all can be obtained from int_full_data + AS Numbers) 
        self.datasets = {
            "int_full_data": int_IP_data,
        }
        
        # Set any variables to be used later:
        self.aliases = []
        self.alias_labels = np.zeros((len(self.IPs)))
        self.total_aliased = 0
        self.ips_embedded = {}
        self.debug = debug