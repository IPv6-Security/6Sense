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
 
import os, pwd
import time
import multiprocessing as mp
import queue
from threading import Thread


from OnlineDealiaserSocketless import SocketLessDealiaserSystem

from Generator.cython_print_ips import print_ips
from DealiaserPrefixBased import DealiaserSystem
from iplibrary import AS_Processor, ExpandIPs

import config as conf

class ScanDeduplicateDealiasObject:
    def __init__(self, seed_addresses, hits_set=None, count_gen_in_dups=True, alias_list=conf.ALIAS_PREFIX_FILE, scanner_log_filepath='/dev/null'):
        """
        Get Scanned, deduplicated, and dealiased IPs from a list of generated addresses (counting duplicates of the seeds as duplicates).

        """
        
        # Deduplicate:
        self.known_addresses = set(seed_addresses)
        self.seed_addresses = set(seed_addresses)
        if hits_set is not None:
            self.known_addresses = self.known_addresses.union(hits_set)
        
        self.duplicates = []
        self.DUPLICATE_SCORE = []
        self.total_duplicates = 0
        self.count_gen_in_dups = count_gen_in_dups
        self.currently_aliased = []
        self.scanner_log_filepath = scanner_log_filepath
        
        # Prefix Based Dealiaser
        self.DEALIASER = DealiaserSystem(aliases = conf.ALIAS_PREFIX_FILE)
            
    def CreateNewScanner(self):
        self.SCANNER = SocketLessDealiaserSystem(reset_aliases=False) #scanner_log_filepath=self.scanner_log_filepath)


    def AddBatch(self, input_addresses, expanded=False):
        if expanded == False:
            expanded_inputs = ExpandIPs(input_addresses)
        else:
            expanded_inputs = input_addresses
        
        
        dups = 0
        seeds = 0

        deduplicated_addresses = input_addresses
        self.deduped = deduplicated_addresses        
        
        # Dealias
        td1 = time.time()

        ### Online Dealiaser
        dealiased, aliased = self.DEALIASER.dealias(deduplicated_addresses, sorted_ips=False) # Used to be to_dealias
        self.currently_aliased += aliased
        
        td2 = time.time()
        print("Offline Dealias Time: ", td2-td1)
        
        # Scan
        ts1 = time.time()
        self.SCANNER.add([dealiased], expanded = True)
        ts2 = time.time()
        print("Scanner Time: ", ts2-ts1)
    
    def GetResults(self, debug=False):
        
        print("Begin Closing Connection:")
        ts1 = time.time()
        self.SCANNER.closeConnection()
        ts2 = time.time()
        print("Connection Closing Time: ", ts2-ts1)

        tgr1 = time.time()
        self.SCANNING_SCORE, ACTIVE_IPS, ALIASING_SCORE, ALIASED_IPS = self.SCANNER.getResults(expanded=True)

        TOTAL_ALIASED_IPS = []
        for a in ALIASED_IPS:
            TOTAL_ALIASED_IPS += a
        
        TOTAL_ALIASED_IPS += self.currently_aliased
        self.currently_aliased = []
        
        
        TOTAL_ACTIVE_IPS = []
        for a in ACTIVE_IPS:
            TOTAL_ACTIVE_IPS += a
            
        self.RawHits = TOTAL_ACTIVE_IPS
            
        tgr2 = time.time()
        print("Parse Time: ", tgr2-tgr1)

        return TOTAL_ACTIVE_IPS, TOTAL_ALIASED_IPS

    
    
def ThreadingScanDeduplicateDealiasThreadSide(alias_queue, hits_queue, commands_queue, client_commands, generated_queue, seed_addresses, scanner_log_filepath):
    
    sddo = ScanDeduplicateDealiasObject(seed_addresses, scanner_log_filepath=scanner_log_filepath)

    while True:
        cmd = commands_queue.get()
        if cmd == "end":
            break
        elif cmd == "create-scanner":
            sddo.CreateNewScanner()
        elif cmd == "add-batch":
            # read from gen queue until end of line
            # Send to AddBatch
            addresses_to_scan = []
            while True:
                ip = generated_queue.get()
                if ip == "end-batch":
                    break
                else:
                    addresses_to_scan += ip
            sddo.AddBatch(addresses_to_scan, expanded=True)
        elif cmd == "get-results":
            # Get Results
            active, aliased = sddo.GetResults()
            
            # Tell Client it's time to read results. 
            client_commands.put("begin-read")
            
            # Write to Hits Queue
            hits_queue.put(active)
            hits_queue.put("end-line")
                
            # Write to Aliases Queue
            alias_queue.put(aliased)
            alias_queue.put("end-line")

            # Update Offline Dealiaser with Results
            
            sddo.DEALIASER.update(aliased)

            

         
    
class ThreadingScanDeduplicateDealiasClientSide():
    def __init__(self, seed_addresses, hits_set=None, count_gen_in_dups=True, scanner_log_filepath='/dev/null'):
        """
        Get Scanned, deduplicated, and dealiased IPs from a list of generated addresses (counting duplicates of the seeds as duplicates).

        Essentially do all prerpocessing to get actual hits. 
        """
        self.alias_queue = mp.Manager().Queue()
        self.hits_queue = mp.Manager().Queue()
        self.commands_queue = mp.Manager().Queue()
        self.client_commands = mp.Manager().Queue()
        self.generated_queue = mp.Manager().Queue()

        self.sddo_thread = mp.Process(target=ThreadingScanDeduplicateDealiasThreadSide, args=(self.alias_queue, self.hits_queue, self.commands_queue, self.client_commands, self.generated_queue, seed_addresses, scanner_log_filepath))
        self.sddo_thread.daemon = False
        self.sddo_thread.start()
        
        self.total_duplicates = 0
        
            
    def CreateNewScanner(self):
        self.commands_queue.put("create-scanner")


    def AddBatch(self, input_addresses, expanded=False):
        # Send Comand Saying Batch is coming
        self.commands_queue.put("add-batch")
        
        # Send Batch
        self.generated_queue.put(input_addresses)
        self.generated_queue.put("end-batch")


    def StartGetResults(self, debug=False):
        self.commands_queue.put("get-results")
    
    def GetResults(self, debug=False, send_wait=True):
        if send_wait == True:
            self.commands_queue.put("get-results")

        t_1 = time.time()
        # Wait For Completion of Scanning
        while True:
            msg = self.client_commands.get()
            if msg == "begin-read":
                break
        
        TOTAL_ACTIVE_IPS = []
        TOTAL_ALIASED_IPS = []
        
        t_2 = time.time()
        print("Wait Time: ", t_2 - t_1)
        
        # Get Hits
        while True:
            ip = self.hits_queue.get()
            if ip != "end-line":
                TOTAL_ACTIVE_IPS += ip
            else:
                break
        
        # Get Aliases
        while True:
            ip = self.alias_queue.get()
            if ip != "end-line":
                TOTAL_ALIASED_IPS += ip
            else:
                break

        return TOTAL_ACTIVE_IPS, TOTAL_ALIASED_IPS
    
    
    def EndProcess(self):
        self.commands_queue.put("end")
        self.sddo_thread.join()

        
    