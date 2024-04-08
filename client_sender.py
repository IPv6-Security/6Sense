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

from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
import sys
import threading
import subprocess
import time

# Receiver
def create_response_listener():
    r = socket(AF_INET, SOCK_STREAM)
    r.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  
    r.bind(('',6002))
    myaddr, myport = r.getsockname()
    r.listen(1)
    conn, remoteaddr = r.accept()
    dar_input = conn.makefile('rb', 0)
    
    with open('scan_results', 'w+') as file_output:
        proc = subprocess.Popen(
            ['tee'],
            stdin=dar_input,
            stdout=file_output, shell=True
        )
    proc.wait()

def create_alias_listener():
    r = socket(AF_INET, SOCK_STREAM)
    r.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  
    r.bind(('',6003))
    myaddr, myport = r.getsockname()
    r.listen(1)
    conn, remoteaddr = r.accept()
    dar_input = conn.makefile('rb', 0)
    
    with open('alias_results', 'w+') as file_output:
        proc = subprocess.Popen(
            ['tee'],
            stdin=dar_input,
            stdout=file_output, shell=True
        )
    proc.wait()


def run_sending_function():
    background_thread = threading.Thread(target=create_response_listener)
    background_thread.daemon = True
    background_thread.start()


    # Sender
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(('localhost', 6001))
    # Confirm setup first
    data = s.recv(1024).decode()

    dar_output = s.makefile('w',1024)
    with open("ips_to_be_scanned", 'r') as g:
        while True:
            line = g.readline()
            if not line:
                break
            dar_output.write(line)
            dar_output.flush()

    dar_output.close()
    s.close()
    background_thread.join()

    
if __name__ == "__main__":
    run_sending_function()