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

from subprocess import Popen, PIPE
import json
import time
import ipaddress
import config as conf
import multiprocessing as mp


class DealiaserSystem():
        
    def __init__(self, aliases=conf.ALIAS_PREFIX_FILE, home_dir=""):
        """
        ScannerSystem: Asynchronous Dealiaser
        """
        self.dealiaser_file = aliases
        self.initialize_dealiaser()

    
    def initialize_dealiaser(self):
        """
        Start all necessary sockets/listeners.
        """
        self.processes = 6
        commands = {}
        for p in range(0, self.processes):
            commands[p] = [conf.OFFLINE_DEALIASER_PATH + "/aliasv6", "--flush", "--expanded", "-c", self.dealiaser_file, "-m", conf.OFFLINE_DEALIASER_PATH +"/dealiaser" + str(p) + ".meta", "-l", conf.OFFLINE_DEALIASER_PATH + "/dealiaser" + str(p) + ".log"]
        
        self.process_list = []
        for p in range(0, self.processes):
            self.process_list.append(Popen(commands[p], stdin=PIPE, stdout=PIPE))

        self.dealiased_queue = mp.Queue()
        self.aliased_queue = mp.Queue()

        time.sleep(0.2)
        
    def create_command(self, dtype, data=None):
        if dtype not in ["lookup", "insert", "quit"]:
            return {}
        command = {"Type": dtype}
        if dtype in ["lookup", "insert"]:
            if not data:
                return {}
            command["Data"] = data
        return command
    
    def update(self, aliased_ips):
        for a in aliased_ips:
            aliased_prefix = a[:29] + "::/96"
            data = self.create_command("insert", aliased_prefix)
            if not data:
                print("Error: ", line)
                continue
            for p in self.process_list:
                p.stdin.write(bytes("{}\n".format(json.dumps(data)), encoding='utf-8'))
                p.stdin.flush()
        
    
    def dealias(self, ips_to_dealias, sorted_ips=False):
        args = []
        for p in range(0, self.processes-1):
            args.append((self.process_list[p], ips_to_dealias[p*int(len(ips_to_dealias)/self.processes):(p+1)*int(len(ips_to_dealias)/self.processes)], self.dealiased_queue, self.aliased_queue))
        
        args.append((self.process_list[-1], ips_to_dealias[(self.processes-1)*int(len(ips_to_dealias)/self.processes):], self.dealiased_queue, self.aliased_queue))

        spawned = []
        for p in range(0, len(args)):
            p0 = mp.Process(target=sub_dealias, args=args[p])
            p0.start()
        

        dealiased_IPs = []
        aliased_IPs = []

        for x in range(0, self.processes):
            dealiased_IPs +=  self.dealiased_queue.get()
            aliased_IPs +=  self.aliased_queue.get()
        
        return dealiased_IPs, aliased_IPs

    
    def end(self):
        """
        Close Connection to Scanner. 
        """
        for process in self.process_list:
            data = self.create_command("quit")
            if not data:
                print("something is wrong")
            process.stdin.write(bytes("{}\n".format(json.dumps(data)), encoding='utf-8'))
            process.stdin.flush()

            process.stdin.close()
            process.stdout.close()


def sub_dealias(process, ips_to_dealias, dealiased, aliased):
    dealiased_IPs = []
    aliased_IPs = []
    reader_counter = 0
    t1 = time.time()
    for line in ips_to_dealias:

        process.stdin.write(bytes(f"{line}\n", encoding='utf-8')) 
        reader_counter += 1

        if reader_counter % 1000 == 0:
            process.stdin.flush()
        
        if reader_counter % 1000 == 0:
            for _ in range(0, 1000):
                results = json.loads(process.stdout.readline())
                ip = results["ip"] # the ip we sent
                res = results["result"]["aliased"] # true or false
                if res and "metadata" in results["result"]:
                    alias_prefix = results["result"]["metadata"]
                    aliased_IPs.append(ip)
                else:
                    dealiased_IPs.append(ip)

    process.stdin.flush()
    for _ in range(0, reader_counter % 1000):
        results = json.loads(process.stdout.readline())
        ip = results["ip"] # the ip we sent
        res = results["result"]["aliased"] # true or false
        if res and "metadata" in results["result"]:
            alias_prefix = results["result"]["metadata"]
            aliased_IPs.append(ip)
        else:
            dealiased_IPs.append(ip)
    
    t2 = time.time()

    dealiased.put(dealiased_IPs)
    aliased.put(aliased_IPs)    