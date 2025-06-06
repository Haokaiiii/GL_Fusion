#!/bin/bash

# Wrapper script to ensure conda environment is activated on each node
# This script will be run by each MPI process

set -e

# Activate conda environment on this node
source ~/.bashrc
condapbs_ex gl

# Add project directory to Python path
export PYTHONPATH="/g/data/hn98/ht5059/GL:$PYTHONPATH"

# Run the Python script
python3 /g/data/hn98/ht5059/GL/src/test_multinode.py 