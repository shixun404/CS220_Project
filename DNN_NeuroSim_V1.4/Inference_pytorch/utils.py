import subprocess
import os

def compile_Neurosim():
    command = "make clean; make -j; cd -"
    directory = "./NeuroSIM"
    original_directory = os.getcwd()
    # Change to the specified directory
    os.chdir(directory)

    # Run the command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    os.chdir(original_directory)
    print(stdout, stderr)