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

from OnlineDealiaserSocketless import SocketLessDealiaserSystem


if __name__ == "__main__":
    with open("/dataset/path", 'r') as f:
        dealiased_IPs = f.read().splitlines()
    de = SocketLessDealiaserSystem(scanner_log_filepath="log/path", port=80, protocol="icmp6_echoscan")
    de.add([dealiased_IPs])
    de.closeConnection()
    _, active_dealiased_ips, _, active_aliased_ips = de.getResults()