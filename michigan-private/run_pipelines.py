#!/usr/bin/env python3

"""
Run all pipelines in parallel. Reads commands from "run_pipeline_cmds.txt" and prints any failed commands to
"failed_cmds.txt".
"""

import os
import queue
import subprocess
import threading

NUM_WORKERS = 8

# Ensure that this script operates in the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
os.chdir(PROJ_DIR)

# Load the commands to run, stored in the file below
with open('run_pipeline_cmds.txt', 'r') as f:
    pipeline_cmds = [line.strip() for line in f.readlines()]

# Set up file to store commands that failed
failed_cmds_file = open('failed_cmds.txt', 'w')
failed_cmds_file_lock = threading.Lock()

def do_work(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        with failed_cmds_file_lock:
            failed_cmds_file.write('{}\n'.format(cmd))
            failed_cmds_file.flush()

def worker():
    while True:
        cmd = q.get()
        if cmd is None:
            break
        do_work(cmd)
        q.task_done()

q = queue.Queue()
threads = []
for i in range(NUM_WORKERS):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for cmd in pipeline_cmds:
    q.put(cmd)

# block until all tasks are done
q.join()

# stop workers
for i in range(NUM_WORKERS):
    q.put(None)
for t in threads:
    t.join()

failed_cmds_file.close()
