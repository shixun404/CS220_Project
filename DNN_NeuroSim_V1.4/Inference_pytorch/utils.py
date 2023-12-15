import subprocess
import os
import re

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
    # print(stdout, stderr)

def output_post_process(output):
    metrics = { 'Energy_Efficiency_TOPS/W':"Energy Efficiency TOPS/W (Pipelined Process):",
               'Throughput_TOPS':"Throughput TOPS (Pipelined Process):",
               'Throughput_FPS':"Throughput FPS (Pipelined Process):",
               'Compute_Efficiency_TOPS/mm^2':"Compute efficiency TOPS/mm^2 (Pipelined Process):",
                'ChipArea_um^2': "ChipArea :",
    }
    result = {}
    for line in output.splitlines():
        print(line)
        for key in metrics.keys():
            if metrics[key] in line:
                result[key] = line.split(':')[-1]
                result[key] = result[key].lstrip()
                result[key] = float(re.match(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', result[key])[0])
    print(result)
    return result
    # assert 0

def param_config(speedUpDegree, technode, cellBit):
    param_keyword = {
        "speedUpDegree": "int speedUpDegree_ =",
        "technode": "int technode_ = ",
        "cellBit": "int cellBit_ = ",
    }

    params = {
        "speedUpDegree": speedUpDegree,
        "technode": technode,
        "cellBit": cellBit,
    }
    param_path = "NeuroSIM/Param.cpp"
    with open(param_path, 'r') as f:
        lines = f.readlines()

    # Process each line
    modified_lines = []
    for line in lines:
        for key in param_keyword.keys():
            if param_keyword[key] in line:
                line = param_keyword[key] + f"{params[key]};\n"
                print(line)
        modified_lines.append(line)
    # print(modified_lines)
    # Write the modified lines back to the file
    with open(param_path, 'w') as f:
        for line in modified_lines:
            f.write(line)
    # assert 0