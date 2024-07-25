import os
import shutil
import json
import csv
#from argparse import ArgumentParser
from collections import deque

# Directories
dirname = "crossvalidation"
LAS_DATA_TRAIN = f"tests/data/{dirname}/train/"
LAS_DATA_TEST = f"tests/data/{dirname}/test/"
DATASET_HDF5_PATH = f"tests/data/{dirname}.hdf5"

# Configuration log file
config_log = f"tests/data/{dirname}/state.json"
csv_file = f"tests/data/{dirname}/{dirname}.csv"

# Function to write csv file of the current config
def write_csv(train_files, test_files):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['basename', 'split'])
        for path in train_files:
            writer.writerow([os.path.basename(path), 'train'])
        for path in test_files:
            writer.writerow([os.path.basename(path), 'test'])

# Function to read configurations log
def read_previous_state():
    if os.path.exists(config_log):
        with open(config_log, 'r') as file:
            config = json.load(file)
            files = [LAS_DATA_TRAIN + file for file in config["train"]] + [LAS_DATA_TEST + file for file in config["test"]]
            return files
    else:
        return []

# Function to write configurations log
def write_current_state(config):
    config["train"] = [os.path.basename(file) for file in config["train"]]
    config["test"] = [os.path.basename(file) for file in config["test"]]
    with open(config_log, 'w') as file:
        json.dump(config, file, indent=4)

def shift_cross(): 
    all_files = read_previous_state()  

    if not all_files:
        # Gather all .las files
        all_files = []
        for folder in [LAS_DATA_TRAIN, LAS_DATA_TEST]:
            for file in os.listdir(folder):
                if file.endswith('.las'):
                    all_files.append(os.path.join(folder, file))

    items = deque(all_files)
    items.rotate(9) # Ogni gruppo di hold-out è composto da 9 elementi
    rotated_files = list(items)
    new_train = rotated_files[:80]
    new_test = rotated_files[80:]

    # Move files to new locations
    out_dirs = [LAS_DATA_TRAIN, LAS_DATA_TEST]
    for i,input_dir in enumerate([new_train, new_test]):
        #print("INPUT DIR: ", input_dir)
        for file_path in input_dir:
             try:
                 shutil.move(file_path, out_dirs[i])
             except shutil.Error as e:
                 continue

    # Save the new configuration and write to csv
    write_csv(new_train, new_test)

    new_config = {
        'train': new_train,
        'test': new_test
    }
    write_current_state(new_config)

if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser.add_argument("num_shift", type=int)
    #args = parser.parse_args()
    #shift_cross(args.num_shift)
    shift_cross()
