import os
import random
import shutil
import json
import csv

# Directories
dirname = "crossvalidation"
LAS_DATA_TRAIN = f"tests/data/{dirname}/train/"
LAS_DATA_VAL = f"tests/data/{dirname}/val/"
LAS_DATA_TEST = f"tests/data/{dirname}/test/"
DATASET_HDF5_PATH = f"tests/data/{dirname}.hdf5"

# Configuration log file
config_log = f"tests/data/{dirname}/permutation.json"
csv_file = f"tests/data/{dirname}/{dirname}.csv"

# Function to read configurations log
def read_configurations():
    if os.path.exists(config_log):
        with open(config_log, 'r') as file:
            return json.load(file)
    else:
        return []

# Function to write configurations log
def write_configuration(config):
    configs = read_configurations()
    configs.append(config)
    with open(config_log, 'w') as file:
        json.dump(configs, file, indent=4)

# Function to write csv file of the current config
def write_csv(train_files, val_files, test_files):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['basename', 'split'])
        for path in train_files:
            writer.writerow([os.path.basename(path), 'train'])
        for path in val_files:
            writer.writerow([os.path.basename(path), 'val'])
        for path in test_files:
            writer.writerow([os.path.basename(path), 'test'])

# Gather all .las files
all_files = []
for folder in [LAS_DATA_TRAIN, LAS_DATA_VAL, LAS_DATA_TEST]:
    for file in os.listdir(folder):
        if file.endswith('.las'):
            all_files.append(os.path.join(folder, file))

def distribute_files():
    existing_configs = read_configurations()
    
    while True:
        random.shuffle(all_files)
        new_train = sorted(all_files[:71])
        new_val = sorted(all_files[71:80])
        new_test = sorted(all_files[80:])

        new_config = {
            'train': new_train,
            'val': new_val,
            'test': new_test
        }

        # Check if the configuration is new
        if new_config not in existing_configs:
            break

    # Move files to new locations
    out_dirs = [LAS_DATA_TRAIN, LAS_DATA_VAL, LAS_DATA_TEST]
    for i,input_dir in enumerate([new_train, new_val, new_test]):
        for file_path in input_dir:
             try:
                 shutil.move(file_path, out_dirs[i])
             except shutil.Error as e:
                 continue

    # Save the new configuration and write to csv
    write_configuration(new_config)
    write_csv(new_train, new_val, new_test)

if __name__ == '__main__':
    distribute_files()
