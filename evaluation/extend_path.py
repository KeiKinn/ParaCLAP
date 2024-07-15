import sys
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, parent_dir)