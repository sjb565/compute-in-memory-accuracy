import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy
from torchvision.models import resnet18
from torchvision import transforms, datasets
import torch.nn as nn

from utils import *
from noise import *

# if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_params = {
    'input_bits': None,
    'input_scale': None,
    'input_zp': 0,
    'weight_bits': None,
    'weight_scale': None,
    'weight_zp': 0,

    'input_noise': None,
    'weight_value_noise': None,
    'weight_physical_noise': None,
    'adc_noise': None,

    'num_rows': None,

    'padding': 0,
    'stride': 1,

    'input_slice_lsb_to_msb': None,
    'weight_slice_lsb_to_msb': None,

    'differential_weight': True,
}

### Terminology Clarification ###
# var: Tensor of Output Variance (diverged from ideal output)
# noise: Noise Object created from the variance tensor (or state proportional error)

verbose = True

class Conv2DNoiseEstimation:
    def __init__(self, stat, module_dict, layer_name, conv_params, prev_layer=None):
        self.stat = stat
        self.module_dict = module_dict
        self.layer_name  = layer_name
        self.conv_params = conv_params
        self.prev_layer  = prev_layer

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        
        conv = analog_params | self.conv_params
        input_pdf = self.stat[f'{self.layer_name}.in'] if self.prev_layer is None else self.stat[f'{self.prev_layer}.out']

        conv |= {
            'input_pdf': input_pdf,
            'input_noise': input_noise
        }
        conv_var, conv_stat = single_layer_conv2d_noise(**conv)

        if verbose== True:
            print(f"layer: {self.layer_name}")
            print(f"mean output_var: {torch.mean(conv_var)}\n")

        return conv_var, conv_stat


class FcNoiseEstimation:
    def __init__(self, stat, module_dict, layer_name):
        fc_weight, fc_bias = module_dict[layer_name].weight, module_dict[layer_name].bias
        fc_weight = fc_weight.unsqueeze(-1).unsqueeze(-1)

        self.conv_params = {
            'weight': fc_weight,
            'stride': 1,
            'padding': 0
        }
        self.stat = stat
        self.module_dict = module_dict
        self.layer_name  = layer_name

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        
        conv = analog_params.copy() | self.conv_params
        input_pdf = self.stat[f'{self.layer_name}.in']
        input_pdf = input_pdf.unsqueeze(-1).unsqueeze(-1) # make (C_in, 1, 1) for Convolution
        conv |= {
            'input_pdf': input_pdf,
            'input_noise': input_noise,
            'use_spatial_correlation': False,
        }
        conv_var, conv_stat = single_layer_conv2d_noise(**conv)
        return conv_var.flatten(), conv_stat


class BasicBlockNoiseEstimation:
    def __init__(self, stat, module_dict, layer_name):
        """
        - stat: dictionary of layer name -> PDF Object
        - module_dict: named modules dictionary
        - layer_name: name of the parent layer
        """
        self.stat = stat
        self.module_dict = module_dict
        self.layer_name = layer_name

        self.conv_list: list[Conv2DNoiseEstimation] = []
        for i in range(2):
            conv_iter, bn_iter = f'conv{i+1}', f'bn{i+1}'
            conv_name = f'{layer_name}.{conv_iter}'

            conv_params = get_fused_convbn_params(module_dict, layer_name, conv_iter, bn_iter)
            
            self.conv_list.append(
                Conv2DNoiseEstimation(stat, module_dict, conv_name, conv_params)
            )

        self.downsample = None

        self.ds_layer_name = f'{layer_name}.downsample'
        if f'{self.ds_layer_name}.0' in module_dict:
            downsample_params = get_fused_convbn_params(module_dict, self.ds_layer_name, '0', '1')
            self.downsample = Conv2DNoiseEstimation(stat, module_dict, self.ds_layer_name, downsample_params)

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        #TODO: elegant way to add quantization noise to each layer automatically

        output_noise = input_noise
        for i, conv in enumerate(self.conv_list):
            conv_var, conv_stat = conv(analog_params, output_noise)
            output_noise = ReluNoise(GaussianNoise(conv_var)) # - ReLU Noise (noise suppressed at x <= 0)

        # DOWNSAMPLE (for residual connection)
        #ds_var = self..get_expected_value(input_noise)
        if self.downsample:
            ds_var, ds_stat = self.downsample(analog_params, input_noise)
        else:
            ds_input_pdf = self.stat[f'{self.layer_name}.conv1.in']
            ds_var = ds_input_pdf.get_expected_value(input_noise) # Averaged ReLU error (not mathematically precise)

        output_var = conv_var + ds_var
        output_stat = {k: conv_var[k] + ds_stat[k] for k in conv_var.keys()}

        return output_var, output_stat

class LayerNoiseEstimation:
    def __init__(self, stat, module_dict, layer_name, num_blocks=2):
        """
        - stat: dictionary of layer name -> PDF Object
        - module_dict: named modules dictionary
        - layer_name: name of the parent layer
        """
        self.stat = stat
        self.module_dict = module_dict
        self.layer_name = layer_name

        block_names = [f'{layer_name}.{i}' for i in range(num_blocks)]
        self.blocks = [
            BasicBlockNoiseEstimation(stat, module_dict, block_name) for block_name in block_names
        ]

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        output_noise = input_noise

        for i, block in enumerate(self.blocks):
            output_var, output_stat = block(analog_params, output_noise)

            output_noise = ReluNoise(GaussianNoise(output_var))

        return output_var, output_stat

class ResNetNoiseEstimation:
    def __init__(self, stat, module_dict, num_layers = 4):
        """
        - stat: dictionary of layer name -> PDF Object
        - module_dict: named modules dictionary
        - layer_name: name of the parent layer
        """
        self.stat = stat
        self.module_dict = module_dict
        #conv1, relu, maxpool, layers, avgpool, fc

        conv1_params = get_fused_convbn_params(module_dict, '', 'conv1', 'bn1')
        self.conv1 = Conv2DNoiseEstimation(stat, module_dict, 'conv1', conv1_params)

        #TODO: asserted maxpool ~ avgpool assuming equal selection of local patch
        maxpool_params = get_avgpool_params(num_channels=64, kernel=3, stride=2, padding=1)
        self.maxpool = Conv2DNoiseEstimation(stat, module_dict, 'maxpool', maxpool_params) #, prev_layer = 'conv1'

        self.layer_names = [f'layer{i+1}' for i in range(num_layers)]
        self.layers = [
            LayerNoiseEstimation(stat, module_dict, layer_name) for layer_name in self.layer_names
        ]

        avgpool_params = get_avgpool_params(num_channels=512, kernel=7, stride=1, padding=0)
        self.avgpool = Conv2DNoiseEstimation(stat, module_dict, 'avgpool', avgpool_params) #prev_layer = 'layer4.1.conv2.out'

        self.fc = FcNoiseEstimation(stat, module_dict, 'fc')

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        # Conv1, Bn1
        conv1_var, conv1_stat = self.conv1(analog_params, input_noise)

        #NOTE: maxpool use default param since it's digitally calculated
        maxpool_var, _ = self.maxpool(default_params, GaussianNoise(conv1_var))
        
        output_noise = ReluNoise(GaussianNoise(maxpool_var))
        for i, layer in enumerate(self.layers):
            output_var, output_stat = layer(analog_params, output_noise)
            output_noise = ReluNoise(GaussianNoise(output_var))
        
        # Average Pool
        avgpool_var, _ = self.avgpool(default_params, GaussianNoise(output_var))
        avgpool_noise = GaussianNoise(avgpool_var)

        #TODO: fc layer with full-precision or to analog despite inefficiency?
        fc_var, fc_stat = self.fc(analog_params, avgpool_noise)

        return fc_var, fc_stat