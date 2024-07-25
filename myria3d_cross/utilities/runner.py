import shlex, subprocess

# Path to the scripts you want to run multiple times
generate_dirs_script = 'utilities/generate_dirs.py'
generate_hdf5_script = 'myria3d/pctl/dataset/shrec18_dataset.py'
main_command_line = 'python run.py logger.comet.experiment_name=run_'

# Number of times to execute the script
number_of_executions = 16

for num in range(number_of_executions):
    print(f"Esperimento numero {num+84}")

    # Generate new dataset using Python
    process_1 = subprocess.run(['python', generate_dirs_script], text=True)
    print(f"Subprocess 1 completed with exit code {process_1.returncode}")
    
    # Generate HDF5 file
    process_2 = subprocess.run(['python', generate_hdf5_script], text=True)
    print(f"Subprocess 2 completed with exit code {process_2.returncode}")

    # Run experiment with new name on comet
    cmd = main_command_line + str(num+84)
    args = shlex.split(cmd)
    process_3 = subprocess.run(args)
    print(f"Subprocess 3 completed with exit code {process_3.returncode}")
