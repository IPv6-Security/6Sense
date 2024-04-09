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

import sys
import ipaddress

#Function to return contents of a file as a list

def fn_file_to_list(filename):
    ips = []
    with open(filename, 'r') as f:
       lines = f.read().splitlines()
    print("Lines not containing Allocations in the file:")
    for line in lines:
        if line == '::\t0':
            print("\t",line)
        elif (line[0]==';'):
            print("\t",line)
        else:
            parts = line.strip().split('\t')
            alloc = parts[0] + '/' + parts[1]
            asn = parts[2]
            ips.append(alloc+'\t'+asn)
    print("Number of Allocations in the file : ",len(ips))
    set_ips = set(ips)
    list_ips = list(set_ips)
    print("Unique count of Allocations in the file : ",len(list_ips))
    return list_ips


#Function to write a list to a file

def fn_list_to_file(filename,alloc_list):
    with open(filename, 'w') as f:
        f.write('\n'.join(alloc_list))


#Prefixes below /32 rounded to /32

def fn_round2(alloc_list):

    to_round_list = []
    round_list = []
    temp = []
    rounding = []
    rounding_list = []

    for i in alloc_list:
        v = i.split('\t')[0]
        val = int(v.split('/')[-1])
        if val<32:
            to_round_list.append(i)
        else:
            round_list.append(i)

    print("Allocation with rounded prefix lower than /32 : ",len(to_round_list))
    print("Allocation with rounded prefix higher than /32 (including /32) : ",len(round_list))

    for i in to_round_list:
        v = i.split('\t')[0]
        v1 = i.split('\t')[-1]
        temp = list(ipaddress.ip_network(v).subnets(new_prefix=32))
        for j in temp:
            j = str(j) + "\t" + v1
            j = [j]
            rounding = rounding + j
    for i in rounding:
        j = str(i)
        rounding_list.append(j)
    round_list = round_list + rounding_list
    
    print("Allocations with rounded Prefixes lower than /32 after rounding to /32: ",len(rounding_list))
    print("Allocations with Prefixes rounded to /32 or higher nybbles: ",len(round_list))

    return round_list

#Round to closest 4s or nybble

def fn_round1(alloc_list):
    
    to_round_list = []
    round_list = []
    rounding = []
    rounding_list = []
    temp = []
    
    for i in alloc_list:
        v = i.split('\t')[0]
        val = int(v.split('/')[-1])
        if val%4 != 0:
            to_round_list.append(i)
        elif val%4 == 0:
            round_list.append(i)
    print("Allocations with Prefixes to round : ",len(to_round_list))
    print("Allocations with rounded Prefixes : ",len(round_list))

    for i in to_round_list:
        v = i.split('\t')[0]
        v1 = i.split('\t')[-1]
        val = int(v.split('/')[-1])
        if val%4 == 1:
            temp = list(ipaddress.ip_network(v).subnets(prefixlen_diff=3))
            for j in temp:
                j = str(j) + "\t" + v1
                j = [j]
                rounding = rounding + j
        elif val%4 == 2:
            temp = list(ipaddress.ip_network(v).subnets(prefixlen_diff=2))
            for j in temp:
                j = str(j) + "\t" + v1
                j = [j]
                rounding = rounding + j
        elif val%4 == 3:
            temp = list(ipaddress.ip_network(v).subnets(prefixlen_diff=1))
            for j in temp:
                j = str(j) + "\t" + v1
                j = [j]
                rounding = rounding + j
    
    for i in rounding:
        j = str(i)
        rounding_list.append(j)
    round_list = round_list + rounding_list

    print("Allocations after rounding the prefix to the closest nybble: ",len(round_list))
    final_list = fn_round2(round_list)
    
    return final_list

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 rounding.py <input_filename> <output_filename>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    print("Rounding allocations to the closest nybble with allocations being /32 and above : ")
    ip_list = fn_file_to_list(input_filename)
    alloc_round = fn_round1(ip_list)
    fn_list_to_file(output_filename, alloc_round)