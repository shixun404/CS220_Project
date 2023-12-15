from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch
import math
import pickle as pkl
name=0

class Rocket(nn.Module):
    def __init__(self, args, logger, num_classes=10, num_features=8192):
        super(Rocket, self).__init__()
        self.classifier = make_layers([('L', num_features, num_classes)], args, logger)

        self.conv_parameters = None
        self.f_mean = None
        self.f_std = None

    def forward(self, x):
        out = self.classifier(x)
        return out


class Bottleneck(nn.Module):
    def __init__(self, args, logger, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.conv = make_layers([('C', in_planes, 4*growth_rate, 1, 'same', 1),
                                  ('C', 4*growth_rate, growth_rate, 3, 'same', 1)], 
                                  args, logger)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, args, logger, in_planes, out_planes):
        super(Transition, self).__init__()
        self.conv = make_layers([('C', in_planes, out_planes, 1, 'same', 1)], args, logger)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.avgpool(out)
        return out


def make_layers(cfg, args, logger):
    global name
    layers = []
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            in_channels = v[1]
            out_channels = v[2]
            if v[4] == 'same':
                padding = v[3]//2
            else:
                padding = 0
            if args.mode == "WAGE":
                conv2d = QConv2d(in_channels, out_channels, kernel_size=v[3], stride=v[5], padding=padding,
                                 logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                                 wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
            elif args.mode == "FP":
                conv2d = FConv2d(in_channels, out_channels, kernel_size=v[3], stride=v[5], padding=padding,
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(name)+'_' )
            name += 1
            batchnorm = nn.BatchNorm2d(out_channels)
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, batchnorm, non_linearity_activation]
            in_channels = out_channels
        if v[0] == 'L':
            if args.mode == "WAGE":
                linear = QLinear(in_features=v[1], out_features=v[2], 
                                logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                                wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, 
                                name='FC'+str(i)+'_', model = args.model, parallelRead=args.parallelRead)
            elif args.mode == "FP":
                linear = FLinear(in_features=v[1], out_features=v[2],
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name='FC'+str(i)+'_')
            layers += [linear]
    return nn.Sequential(*layers)
    
    
def RocketNet(args, logger, num_classes, num_features, pretrained=None, parameters=None):
    model = Rocket(args, logger, num_classes=num_classes, num_features=num_features)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location='cuda'))
    if parameters is not None:
        with open(parameters, 'rb') as f:
            parameters, f_mean, f_std = pkl.load(f)
        model.conv_parameters = parameters
        model.f_mean = f_mean
        model.f_std = f_std
    return model
    
    