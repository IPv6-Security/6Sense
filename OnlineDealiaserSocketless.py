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
import sys
import time
import json
import random
import ipaddress
import threading
import pickle
from threading import Thread
import subprocess
from subprocess import DEVNULL, PIPE
from multiprocessing import Process, Pipe, Queue
import multiprocessing as mp
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
import select

import numpy as np

from client_sender import run_sending_function, create_response_listener, create_alias_listener
import config as conf


# -------- GLOBALS for Multithreading ------------
# STORAGE
SENT_ADDRESSES = set([]) 
ALIAS = {}
ALIAS_TEST_ADDRESS_MAPPING = {}
ALIAS_TEST_IPS = {}
PREFIX_TO_IP = {}
PROC = {}

# PARAMETERS
ALIAS_TEST_CASES = 3 # Number to generate
ALIAS_REPEATS = 3
ALIAS_TEST_CASES_ACTIVE = 2 # Number that have to be active to call this an alias.
PACKETS_SENT = 0 # Aliased Packets Sent
PACKETS_SENT_WHEN_LAST_CHECKED=0
PACKETS_RECIEVED = 0
INPUT_SENT = 0


# MULTI PROCESSOR FUNCTIONS
def listen_on_pipe(output_queue, reciever_queue):    
    """
    Read output of subprocess, and write alias detection addresses to Queue
    
    Output Queue: Queue to Write New Scans to.
    """
    # Create Aliasing Datastructure to Record Results
    # - If Not seen in alias dictionary, presume new.
    # - If Seen in alias dictionary, presume 
    # Create Hits Datastructure
    # - If not in alias dictionary, 
    global SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP, PROC, PACKETS_SENT, ALIAS_TEST_IPS
    print("BEGIN DEALIASING")

    while True:
        line = reciever_queue.get() 
        if len(line) > 0:
            # If Line in aliases - increment alias dictionary
            merged_ips = ""
            for str_line in line.split('\n'):
                str_line = str_line + '\n'
                if str_line in SENT_ADDRESSES:
                    addr_prefix = str_line[:30]
                    if addr_prefix not in ALIAS:
                        ALIAS[addr_prefix] = set([])
                        ALIAS_TEST_IPS[addr_prefix] = []
                        PREFIX_TO_IP[addr_prefix] = [str_line]
                        # Run Aliasing generation -> aliased_test_addresses
                        for alias_number in range(0, ALIAS_TEST_CASES):
                            # Random 16 bits
                            random_half_1 = "{:04x}".format(random.randint(0, 65535)) # "aaaa"
                            random_half_2 = "{:04x}".format(random.randint(0, 65535)) # "bbb" + str(alias_number)
                            alias_mapped_address = str_line[:30] + random_half_1 + ":" + random_half_2 + "\n"
                            ALIAS_TEST_ADDRESS_MAPPING[alias_mapped_address] = addr_prefix
                            ALIAS_TEST_IPS[addr_prefix].append(alias_mapped_address)
                            for _ in range(0, ALIAS_REPEATS): 
                                merged_ips += alias_mapped_address
                            PACKETS_SENT += ALIAS_REPEATS
                    else:
                        if addr_prefix not in PREFIX_TO_IP:
                            PREFIX_TO_IP[addr_prefix] = []
                        PREFIX_TO_IP[addr_prefix].append(str_line)
                    
                if str_line in ALIAS_TEST_ADDRESS_MAPPING:
                    # If Address is one of our alias tests
                    original_address = ALIAS_TEST_ADDRESS_MAPPING[str_line]
                    ALIAS[original_address].add(str_line) 
            if len(merged_ips) > 0:
                output_queue.put(merged_ips)
        else:
            print("Finished Reading Output - Multiprocess")
            break
            
def run_subprocess(listener_queue):
    """
    COMPLETE
    Send Addresses to Subprocess from Queue
    - listener_queue - Queue to Scan From
    - out_pipe - Output pipe for stdout
    """
    global SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP, PROC
    print("BEGIN SENDING")
    count2 = 0
    while True:
        line = listener_queue.get()
        if len(line) > 0:
            count2 += 1
            try:
                PROC.stdin.write(line)
                PROC.stdin.flush()
            except Exception as e:
                print(e)
        else:
            print("Finished Sending To Scanner")
            break    
    PROC.stdin.close()

def accept_subprocess(reciever_queue):
    global SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP, PROC, PACKETS_RECIEVED
    print("BEGIN LISTENING FOR RESPONSES")
    prior_line = ""
    current_count = 0
    last_count = 0
    start_time = time.time()
    while True:
        if select.select([PROC.stdout, ], [], [], 0.3)[0]:
            line = PROC.stdout.readline()
            if len(line) > 0:
                prior_line += line.strip() + '\n'
                PACKETS_RECIEVED += 1
                current_count += 1
                if (current_count - last_count) >= 1000:
                    last_count = current_count
                    reciever_queue.put(prior_line)
                    prior_line = ""
                
            else:
                if len(prior_line) > 0:
                    reciever_queue.put(prior_line)
                    print("Prior Line to Place: ", len(prior_line))

                print("Recieving Finished")
                break
        else:
            if len(prior_line) > 0:
                reciever_queue.put(prior_line)
                prior_line = ""
                last_count = current_count

def get_log_line(scanner_log_filepath):
    log_line = subprocess.check_output(['tail', '-1', scanner_log_filepath])
    while ("[INPUT HANDLER]" in str(log_line)) or ("probes sent" not in str(log_line)):
        time.sleep(1)
        log_line = subprocess.check_output(['tail', '-1', scanner_log_filepath])
    
    return log_line
    



class SocketLessDealiaserSystem():
    
    def __init__(self, scanner_log_filepath=None, config_filepath=None, duplicates=True, return_hits=True, port=None, protocol=None, reset_aliases=True, dest_unreach_filepath=None):
        """
        ScannerSystem: Asynchronous scanner using sockets to convey IP addresses. 
            duplicates: No Longer Used. Returns all IPs found active (kept for legacy support)
            return_hits: Not Longer Used: Always returns IPs foudn as hits for parsing. 
            reset_aliases: Should we reset the list of Aliased and Dealiased Addresses every time we run orkeep them across iterations. 
        """
        # Output Logging
        if scanner_log_filepath is None:
            self.scanner_log_filepath = conf.SCANNER_LOG_FILEPATH
            print("SCANNER FILENAME: ", self.scanner_log_filepath)
            self.scanner_log_object = open(self.scanner_log_filepath, 'a')
        else:
            self.scanner_log_filepath = scanner_log_filepath
            print("SCANNER FILENAME: ", self.scanner_log_filepath)
            self.scanner_log_object = open(self.scanner_log_filepath, 'a')
            
        # Output Destination Unreachables:
        if dest_unreach_filepath is None:
            self.dest_unreach_filepath = conf.icmp_dest_unreach_output_file
            print("DEST UNREACH FILENAME: ", self.dest_unreach_filepath)
        else:
            self.dest_unreach_filepath = dest_unreach_filepath
            print("DEST UNREACH FILENAME: ", self.dest_unreach_filepath)

        if port is None:
            self.port = conf.target_port
        else:
            self.port = port

        if protocol is None:
            self.protocol = conf.PORT_TO_SCAN
        else:
            self.protocol = protocol

        self.cooldown = conf.cooldown_time

        # Write Config File:
        if config_filepath is None:
            if os.path.exists(conf.SCANNER_CONFIG_FILE) != True or conf.RESET_CONFIG == True:
                conf_string = '[Application Options]\n'
                conf_string += 'icmp-dest-unreach-output-file="' + conf.icmp_dest_unreach_output_file + '"\n'
                conf_string += 'blocklist-file="' + conf.blocklist_file + '"\n'
                conf_string += 'source-ip="' + conf.source_ip + '"\n'
                conf_string += 'interface="' + conf.interface + '"\n'
                conf_string += 'gateway-mac="' + conf.gateway_mac + '"\n'
                conf_string += 'cooldown-time=' + str(0) + '\n'
                conf_string += 'senders=' + str(conf.senders) + '\n'
                conf_string += 'receivers=' + str(conf.receivers) + '\n'
                conf_string += 'receiver-buffer-size=' + str(conf.receiver_buffer_size) + '\n'
                conf_string += 'rate=' +str(conf.rate) + '\n'
                conf_string += 'probes=' + str(conf.probes) + '\n'
                conf_string += 'expanded=' + conf.expanded + '\n'
                conf_string += 'flush=' + conf.flush + '\n'
                conf_string += 'simulation=' + conf.simulation + '\n'
                conf_string += 'simulation-hitrate=' + str(conf.simulation_hitrate) + '\n'
                conf_string += 'target-port=' + str(self.port) + '\n'
                if conf.PORT_TO_SCAN == "udp6_dnsscan":
                    conf_string += '[udp6_dnsscan]\n'
                    conf_string += 'udp-query-domain="' + conf.udp_query_domain + '"\n'
                    conf_string += 'udp-query-type="' + conf.udp_query_type + '"\n'

                with open(conf.SCANNER_CONFIG_FILE, 'w') as g:
                    g.write(conf_string)
            
            self.config_filepath = conf.SCANNER_CONFIG_FILE
        else:
            self.config_filepath = config_filepath


        self.first_time = time.time()
        self.reset_aliases = reset_aliases
        self.initialize_scanner()
        self.time_tracker = time.time()
        self.send_rate = conf.rate

    
    def initialize_scanner(self):
        """
        Start all necessary sockets/listeners.
        """
        global PROC, PACKETS_SENT, PACKETS_RECIEVED, PACKETS_SENT_WHEN_LAST_CHECKED, SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP
        # Initialize Variables
        SENT_ADDRESSES = set([])
        if self.reset_aliases == True:
            ALIAS = {}
            ALIAS_TEST_IPS = {}
            print("ALIASES RESET: ", len(list(ALIAS.keys())))
        ALIAS_TEST_ADDRESS_MAPPING = {}
        PREFIX_TO_IP = {}
        PACKETS_SENT = 0
        PACKETS_SENT_WHEN_LAST_CHECKED = 0
        PACKETS_RECIEVED = 0

        # Create Scanner Queue
        self.to_scan = Queue()
        self.reciever_queue = Queue()

        commands = ["sudo", conf.SCANNER_PATH, self.protocol, "--config-file", self.config_filepath]
        print("Scanner Commands: ", ' '.join(commands))
        print("Logging Name: ", self.scanner_log_object.name)
    
        PROC = subprocess.Popen(commands, stdin=PIPE, stdout=PIPE, stderr=self.scanner_log_object, bufsize=0, universal_newlines=True)

        # Call MultiProcesser to Create Alias Detection of Addresses
        self.reciever1 = Thread(target=listen_on_pipe, args=(self.to_scan, self.reciever_queue, ))
        self.reciever1.start()

        # Call MultiProcessor to Send to Alias Detection
        self.reciever3 = Thread(target=accept_subprocess, args=(self.reciever_queue,))
        self.reciever3.start()

        # Call MultiProcessor to Send to Scanner
        self.reciever2 = Thread(target=run_subprocess, args=(self.to_scan, ))
        self.reciever2.start()
        time.sleep(1)
    
    def add(self, ipLists, expanded=False):
        global SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP, PROC, PACKETS_SENT, PACKETS_SENT_WHEN_LAST_CHECKED, INPUT_SENT
        
        t1 = time.time()
        ip_string = ""
        total_sent = 0
        currently_known_alias_packets = PACKETS_SENT

        for e in range(0, len(ipLists)):
            count = 0
            list_to_scan = ipLists[e]
            
            for num_ips in range(0, len(list_to_scan)):
                total_sent += 1
                count += 1
                try:
                    ip_address = ""
                    if expanded == False:
                        ip = ipaddress.ip_address(list_to_scan[num_ips])
                        ip_address = ip.exploded
                    else:
                        ip_address = list_to_scan[num_ips]
                    
                    ip_address += '\n'
                    SENT_ADDRESSES.add(ip_address)
                    ip_string += ip_address    

                    if count % 1000 == 0:
                        self.to_scan.put(ip_string)
                        ip_string = ""

                except ValueError:
                    print("ValueError Adding IP: ", num_ips)
                    pass

            if len(ip_string) > 0:
                self.to_scan.put(ip_string)
                ip_string = ""
            print("BATCH SENDING COMPLETE: ", count) 
            
        INPUT_SENT += total_sent
        # Set Time to Send all of this at the current rate minus the time that has already passed
        time_to_send = total_sent/self.send_rate - (time.time()-t1)
        self.time_tracker += (currently_known_alias_packets - PACKETS_SENT_WHEN_LAST_CHECKED)/self.send_rate
        if time.time() >  self.time_tracker:
            self.time_tracker = time.time() + time_to_send 
        else:
            self.time_tracker = self.time_tracker + time_to_send

        PACKETS_SENT_WHEN_LAST_CHECKED = currently_known_alias_packets
                    
    def getResults(self, expanded = False):
        """
        Process Hits and Aliases of Results
        1. Process Hits from Hits File: <current working directory>/"scan_results"
        2. Process Aliases from Aliases File: <current working directory>/"alias_results"
        """ 
        global SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP, PROC, ALIAS_TEST_IPS

        print("Number of /96 Prefixes: ", len(ALIAS.keys()))
        aliases = 0
        aliased_ips = []
        dealiased_ips = []
        t1 = time.time()
        try:
            for line in PREFIX_TO_IP:
                if len(ALIAS[line]) >= ALIAS_TEST_CASES_ACTIVE:
                    for ip_addr in PREFIX_TO_IP[line]:
                        aliased_ips.append(ip_addr[:-1])
                        aliases += 1
                else:
                    for ip_addr in PREFIX_TO_IP[line]:
                        dealiased_ips.append(ip_addr[:-1])
        except Exception as e:
            print(e)

        aliased_ips = list(set(aliased_ips))
        dealiased_ips = list(set(dealiased_ips))

        t2 = time.time()
        print("Parse Time 1: ", t2 -t1)
        print("Aliased IPs: ", aliases)
        print("Alias Dictionary Length: ", len(ALIAS.keys()))

        # Write ALIAS to a file
        """
        t1 = time.time()
        if self.reset_aliases:
            with open(self.scanner_log_filepath  + ".alias_objects", 'ab+') as g:
                pickle.dump(ALIAS, g)
                pickle.dump(ALIAS_TEST_IPS, g)
        else:
            with open(self.scanner_log_filepath  + ".alias_objects", 'wb') as g:
                pickle.dump(ALIAS, g)
                pickle.dump(ALIAS_TEST_IPS, g)
        t2 = time.time()
        print("Pickle Time: ", t2-t1)
        """

        SENT_ADDRESSES = set([])
        if self.reset_aliases == True:
            ALIAS = {}
            ALIAS_TEST_IPS = {}
        ALIAS_TEST_ADDRESS_MAPPING = {}
        PREFIX_TO_IP = {}

        print("Time Until Parsing: ", time.time() - self.first_time)

        # Legacy return format to support older versions of scanner/dealiaser
        return [], [dealiased_ips], [], [aliased_ips]

    
    def reset(self):
        pass

    def closeConnection(self):
        """
        Close Connection to Scanner. 
        """
        global SENT_ADDRESSES, ALIAS, ALIAS_TEST_ADDRESS_MAPPING, PREFIX_TO_IP, PROC, PACKETS_SENT, PACKETS_SENT_WHEN_LAST_CHECKED, PACKETS_RECIEVED, INPUT_SENT
        
        t1 = time.time()
        
        #try:
        # Check Dropped Packets
        log_line = get_log_line(self.scanner_log_filepath)
        split_line = str(log_line).split('(m+: ')
        first_buffer = int(split_line[1].split(')')[0])
        second_buffer = int(split_line[2].split(')')[0])
        print(log_line, first_buffer, second_buffer)


        # Wait until it estimates packets should have been sent 
        t1 = time.time()
        print("Waiting to Close Connection")
        self.time_tracker += (PACKETS_SENT - PACKETS_SENT_WHEN_LAST_CHECKED)/self.send_rate
        PACKETS_SENT_WHEN_LAST_CHECKED = PACKETS_SENT
        time_diff =  self.time_tracker - time.time()
        print("Time Remaining: ", time_diff)
        while time.time() < self.time_tracker:
            time.sleep(time_diff)
            self.time_tracker += (PACKETS_SENT - PACKETS_SENT_WHEN_LAST_CHECKED)/self.send_rate
            PACKETS_SENT_WHEN_LAST_CHECKED = PACKETS_SENT
            time_diff = self.time_tracker - time.time()
            print("Time Remaining: ", time_diff)
        t2 = time.time()
        
        # Make Sure Everything is Sent
        t3 = time.time()
        total_sent = 0
        while total_sent < (INPUT_SENT+PACKETS_SENT-1):
            time.sleep(1)
            log_line = get_log_line(self.scanner_log_filepath)
            total_sent = int(str(log_line).split(" probes sent")[0].split(" ")[-1])
            blocklisted = int(str(log_line).split(" addresses blocked")[0].split(" ")[-1])
            total_sent = blocklisted + total_sent
            
            print("To Send: ", total_sent, "v.", INPUT_SENT+PACKETS_SENT)
        
        
            
        t4 = time.time()
        print("Time Waiting for Sending: ", t3-t4)
        print("Time Spent Waiting for Everything to be sent: ", t2-t1)
        
        # Make Sure We aren't still getting hits after five seconds. 
        still_ips = True
        while still_ips:
            print("Loop Continuing: ", still_ips)
            first_buffers = 0
            second_buffers = 0
            log_line = get_log_line(self.scanner_log_filepath)

            first_responses = int(str(log_line).split(" hits received")[0].split(' ')[-1])
            first_sent = int(str(log_line).split(" probes sent")[0].split(' ')[-1])
            blocklisted = int(str(log_line).split(" addresses blocked")[0].split(" ")[-1])
            first_sent = first_sent + blocklisted


            #for x in range(0, 10):
            #    log_line = get_log_line(self.scanner_log_filepath)                
            #    split_line = str(log_line).split('(m+: ')
            #    first_buffers = int(split_line[1].split(')')[0])
            #    second_buffers = int(split_line[2].split(')')[0])
            #    time.sleep(1)
            time.sleep(5)

            log_line = get_log_line(self.scanner_log_filepath)
            responses = int(str(log_line).split(' hits received')[0].split(' ')[-1])
            sent = int(str(log_line).split(' probes sent')[0].split(' ')[-1])
            sent = blocklisted + sent
            print("Responses: ", first_responses, "->", responses)
             

            if sent > first_sent:
                still_ips = True
            elif responses > first_responses+50:
                still_ips = True
            elif first_buffers != 0 or second_buffers > 10:
                still_ips = True
            else:
                still_ips = False
                
            

        t2 = time.time()
        print("Total Wait Time for Sending: ", t2-t1)

        # Terminate threads and subprocess
        self.to_scan.put('')
        self.reciever2.join()
        self.reciever3.join()
        self.reciever_queue.put('')
        self.reciever1.join()
        print("Final UnSent Size: ", self.to_scan.qsize())
        print("Final Reciever Size: ", self.reciever_queue.qsize())

        # Close Scanner Queue
        s_lines = 0
        while self.to_scan.qsize() != 0:
            line = self.to_scan.get()
            print("Missed Sending Line: ", len(line))
            s_lines += 1
        print("Missed Sender lines: ", s_lines)
        print("Final Sender Queue Size: ", self.to_scan.qsize())
        self.to_scan.close()
        self.to_scan.join_thread()

        # CLose Reciever Queue
        r_lines = 0
        while self.reciever_queue.qsize() != 0:
            line = self.reciever_queue.get()
            print("Missed Reciever line: ", len(line))
            r_lines += 1
        print("Missed Reciever lines: ", r_lines)
        print("Final Reciever Queue Size: ", self.reciever_queue.qsize())
        #try:
        self.reciever_queue.close()
        self.reciever_queue.join_thread()
        self.scanner_log_object.close()



        INPUT_SENT = 0
        PACKETS_SENT = 0
        PACKETS_SENT_WHEN_LAST_CHECKED = 0
        PACKETS_RECIEVED = 0

        print("Completed All Subprocesses")


