import argparse
import os
import time
from utils import *
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
import pandas as pd
from models import dataset
import torchvision.models as models
from minirocket import fit, transform
from utee import hook
#from IPython import embed
from datetime import datetime
from subprocess import call
from subprocess import run
import subprocess
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', type=str, nargs="*", default=['InsectSound', 'FruitFlies', 'FordA', 'FordB'])
parser.add_argument('--model', default='Rocket')
parser.add_argument('--dim', type=int, nargs="*", default=[8192, 4096, 2048, 1024])
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', type=int, default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

ucr_dataset_list = ['InsectSound']

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))
logger("========================================")


rocket_name_list = ['Rocket']
# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

# data loader and model

    
ucr_args = \
{
    "score"      : True,
    "chunk_size" : 2 ** 12,
    "test_size"  : None
}

neurosim_params = {
     "speedUpDegree": [8, 4, 1],
     "technode": [22, 32],
     "cellBit": [1 , 3, 6],
}

result_lines = []

for name_dataset in args.dataset:
    ucr_info_path = "/global/homes/s/swu264/rocket/ucr_info.csv"
    ucr_df = pd.read_csv(ucr_info_path)
    ucr_df.set_index('Unnamed: 0', inplace=True)
    num_classes = int(ucr_df.at[name_dataset, 'nc_test'])
    for dim in args.dim:
             
        from models import Rocket
        model_path = f'./log/{name_dataset}/{dim}/Rocket_{name_dataset}.pth'     # WAGE mode pretrained model
        parameters_path = f'./log/{name_dataset}/{dim}/Rocket_parameters_{name_dataset}.pkl'

        
        with open("NeuroSIM/NetWork_Rocket.csv", 'w') as f:
            f.write(f"1,1,{dim},1,1,{num_classes},0,1\n")
        modelCF = Rocket.RocketNet(args=args, logger=logger, num_classes=num_classes,
                                    num_features=dim, pretrained=model_path, 
                                    parameters=parameters_path)

        for speedUpDegree in neurosim_params["speedUpDegree"]:
            for technode in neurosim_params["technode"]:
                for cellBit in neurosim_params["cellBit"]:
                    param_config(speedUpDegree, technode, cellBit)
                    # assert 0
                    try:
                        compile_Neurosim()
                        
                        if args.cuda:
                            modelCF.cuda()

                        best_acc, old_file = 0, None
                        t_begin = time.time()
                        # ready to go
                        modelCF.eval()

                        test_loss = 0
                        correct = 0
                        trained_with_quantization = True

                        criterion = torch.nn.CrossEntropyLoss()
                        # criterion = wage_util.SSE()

                        # NOTE: Parallel read is not supported in inference accuracy simulation with multi-level cells yet
                        if args.parallelRead < args.subArray and args.cellBit > 1:
                            logger('\n=====================================================================================')
                            logger('ERROR: Partial parallel read is not supported for multi-level cells yet!')
                            logger('Please make sure parallelRead == subArray when cellBit > 1.')
                            logger('We will support partial parallel for multi-level cells in DNN_NeuroSim V1.5 (to be released in Spring 2024).')
                            logger('Thank you for using NeuroSim! We appreciate your patience.')
                            logger('=====================================================================================\n')
                            exit()


                        rocket_test_file_nrows = 0
                        train_loader, test_loader = dataset.get_ucr(ucr_args, test_path=f'/global/cfs/cdirs/m4271/swu264/ucr_archive/{name_dataset}/{name_dataset}_TEST.tsv')
                        for chunk_index, chunk in enumerate(test_loader):

                            print(f"Chunk = {chunk_index + 1}...".ljust(80, " "), end = "\r")

                            # gotcha: copy() is essential to avoid competition for memory access with read_csv(...)
                            test_data = chunk.values.copy()
                            target, data = test_data[:, 0], test_data[:, 1:].astype(np.float32)
                            data = transform(data, modelCF.conv_parameters)
                            data = (data - modelCF.f_mean) / modelCF.f_std
                            data = torch.FloatTensor(data)[:, :dim]
                            target = torch.LongTensor(target)
                            
                            rocket_test_file_nrows += target.shape[0]

                            if chunk_index == 0:
                                hook_handle_list = hook.hardware_evaluation(modelCF,args.wl_weight,args.wl_activate,args.subArray,args.parallelRead,args.model,args.mode)
                            indx_target = target.clone()
                            if args.cuda:
                                data, target = data.cuda(), target.cuda()
                            with torch.no_grad():
                                data, target = Variable(data), Variable(target)
                                output = modelCF(data)
                                test_loss_i = criterion(output, target)
                                test_loss += test_loss_i.data
                                pred = output.data.max(1)[1]  # get the index of the max log-probability
                                correct += pred.cpu().eq(indx_target).sum()
                            if chunk_index == 0:
                                hook.remove_hook_list(hook_handle_list)
                        test_loss = test_loss / chunk_index  # average over number of mini-batch
                        acc = 100. * correct / rocket_test_file_nrows

                        accuracy = acc.cpu().data.numpy()

                        if args.inference:
                            print(" --- Hardware Properties --- ")
                            print("subArray size: ")
                            print(args.subArray)
                            print("parallel read: ")
                            print(args.parallelRead)
                            print("ADC precision: ")
                            print(args.ADCprecision)
                            print("cell precision: ")
                            print(args.cellBit)
                            print("on/off ratio: ")
                            print(args.onoffratio)
                            print("variation: ")
                            print(args.vari)

                        if args.model not in rocket_name_list:
                            logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                                test_loss, correct, len(test_loader.dataset), acc))
                        else:
                            logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                                test_loss, correct, rocket_test_file_nrows, acc))
                            print(rocket_test_file_nrows)



                        # call(["/bin/bash", './layer_record_'+str(args.model)+'/trace_command.sh'])
                        command = ["/bin/bash", './layer_record_'+str(args.model)+'/trace_command.sh']
                        result = run(command, stdout=subprocess.PIPE, text=True)
                        result_post_process = output_post_process(result.stdout)
                        if len(result_lines) == 0:
                            title_line = "dataset, dim, speedUpDegree, technode, cellBit, Accuracy, ChipArea_um^2, Energy_Efficiency_TOPS/W, Throughput_TOPS, Throughput_FPS, Compute_Efficiency_TOPS/mm^2"
                            result_lines.append(title_line)
                        # line = f"{name_dataset}, {dim}, {speedUpDegree}, {technode}, {cellBit}, {acc}, {result_post_process['ChipArea_um^2']}, {result_post_process['Energy_Efficiency_TOPS/W']}, {result_post_process['Throughput_TOPS']}, {result_post_process['Throughput_FPS']}, {result_post_process['Compute_Efficiency_TOPS/mm^2']}"
                        line = (
                        f"{name_dataset}, {dim}, {speedUpDegree}, {technode}, {cellBit}, {acc}, "
                        f"{result_post_process['ChipArea_um^2']}, "
                        f"{result_post_process['Energy_Efficiency_TOPS/W']}, "
                        f"{result_post_process['Throughput_TOPS']}, "
                        f"{result_post_process['Throughput_FPS']}, "
                        f"{result_post_process['Compute_Efficiency_TOPS/mm^2']}"
                        )
                        print(line)
                        result_lines.append(line)
                        with open("result.csv", 'w') as f:
                            for line in result_lines:
                                f.write(line + '\n')
                    except Exception as e:
                        print(e)
                        line = (
                        f"{name_dataset}, {dim}, {speedUpDegree}, {technode}, {cellBit}, {acc}, "
                        )
                        print(line)
                        assert 0
                        continue
                    