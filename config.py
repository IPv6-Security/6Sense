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

#### File Paths
HOME_DIRECTORY = 'YOUR/HOME/DIRECTORY/HERE' 
DATASET_FILE = '' # Path to Dataset with repsect to your HOME_DIRECTORY. This should be a folder containing an "all_ips" file which is your seed dataset. 
OFFLINE_DEALIASER_PATH = '' # Path to aliasv6 directory
UPDATED_PFX_AS_FILE = '' # PFX2AS File (Rounded)
ALIAS_PREFIX_FILE = "" # Aliased prefixes filepath
MODEL_CHECKPOINTS = "" # Directory of LSTM checkpoints with respect to your HOME_DIRECTORY
CHECKPOINT_LSTM = "" # LSTM Checkpoint name with respect to the MODEL_CHECKPOINTS file.

PORT_TO_SCAN = "icmp6_echoscan"
SCANNER_LOG_FILEPATH = HOME_DIRECTORY + "/scanner.log.2"
SCANNER_CONFIG_FILE = HOME_DIRECTORY + "/6sense_config.ini"
RESET_CONFIG = True # WARNING!!!!!!!!!!!!!!!!!!!!! SETTING THIS TO FALSE WILL MEAN THE CONFIG FILE WILL NOT BE OVERWRITTEN EACH EXECUTION. BEST TO BE ON THE SAFE SIDE AND KEEP THIS TRUE AT ALL TIMES.
SCANNER_PATH = ""


#### Scanner Config
icmp_dest_unreach_output_file=HOME_DIRECTORY + "destination_unreachable_filepath.json"
blocklist_file="/blocklist/file/path"
source_ip=""
interface=""
gateway_mac=""
cooldown_time=5
senders=1
receivers=256
receiver_buffer_size=17179869184 #1073741824
rate=10000
probes=1
expanded="true"
flush="true"

# Whether to Simulate Scans
simulation="false"
simulation_hitrate=10

# TCP and UDP scans
target_port=80

# [udp6_dnsscan]
udp_query_domain="example.com"
udp_query_type="AAAA"

