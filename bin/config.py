import os
import sys

# Get the absolute path of the directory containing the script
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..'))  # Adds the lama/ directory to the path
sys.path.append(os.path.join(base_dir, '../saicinpainting'))  # Adds saicinpainting/
