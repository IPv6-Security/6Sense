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

def print_ips(multi_addresses):
    """
    Return a list of formatted IP addresses.
    multi_addresses: A numpy array where the rows are nybbles (starting from bit 0).
    """
    ip_address_list = []
    for formatted_data in multi_addresses:
        ip_string = ""
        i = 0
        for a in formatted_data:
            i += 1
            str_ip = hex(a)[2:]
            #str_ip = a[2:]
            if len(str_ip) > 1:
                str_ip = "f"
            ip_string += str_ip
            if (i)%4 == 0 and i != 32:
                ip_string += ":"
                
        if len(ip_string) < 39:
            while (i)%4 != 0:
                i += 1
                ip_string += "0"
            if ip_string[-1] == ":" and i != 32:
                ip_string += ":"
            elif i != 32:
                ip_string += "::"
            
        ip_address_list.append(ip_string)    
    return ip_address_list