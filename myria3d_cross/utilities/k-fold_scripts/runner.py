import os, shlex, subprocess, time

# Path to the scripts you want to run multiple times
shift_cross_script = 'utilities/shift_cross.py'
shuffle_dataset_script = 'utilities/shuffle_dataset.py'
generate_hdf5_script = 'myria3d/pctl/dataset/shrec18_dataset.py'
main_command_line = 'python run.py logger.comet.experiment_name='

# Number of times to execute the script
num_crossvalidations = 5

for c in range(num_crossvalidations):
    print(f"Crossvalidazione numero {c}")

    # Shuffle the dataset
    process_0 = subprocess.run(['python', shuffle_dataset_script], text=True)
    print(f"Subprocess 0 completed with exit code {process_0.returncode}")

    for r in range(0,10): # 10-fold crossvalidation

        # Shift the dataset 
        process_1 = subprocess.run(['python', shift_cross_script], text=True)
        print(f"Subprocess 1 completed with exit code {process_1.returncode}")
        
        # Generate HDF5 file
        process_2 = subprocess.run(['python', generate_hdf5_script], text=True)
        print(f"Subprocess 2 completed with exit code {process_2.returncode}")

        time.sleep(1)

        # Run experiment with new name on comet
        cmd = main_command_line + "cross_" + str(c) + "_run_" + str(r)
        args = shlex.split(cmd)
        process_3 = subprocess.run(args)
        print(f"Subprocess 3 completed with exit code {process_3.returncode}")
    
    # Shift one last time to restore the original dataset
    #process_1 = subprocess.run(['python', shift_cross_script], text=True)
    #print(f"Dataset structure restored, exit code {process_1.returncode}")

    # Delete state.json file
    os.remove("tests/data/crossvalidation/state.json")
