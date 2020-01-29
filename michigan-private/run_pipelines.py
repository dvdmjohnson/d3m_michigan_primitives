#!/usr/bin/env python3

"""
Run all pipelines in parallel. Reads commands from "run_pipeline_cmds.txt" and prints any failed commands to
"failed_cmds.txt".
"""

import os

from utils import BashCommandWorkerPool


def main():

    # Ensure that this script operates in the project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    os.chdir(PROJ_DIR)

    # Load the commands to run, stored in the file below
    with open('run_pipeline_cmds.txt', 'r') as f:
        pipeline_cmds = [line.strip() for line in f.readlines()]

    # Run pipeline commands in parallel
    pool = BashCommandWorkerPool(8, failed_cmds_file_path='failed_cmds.txt')
    for cmd in pipeline_cmds:
        pool.add_work(cmd)
    pool.join()


if __name__ == '__main__':
    main()
