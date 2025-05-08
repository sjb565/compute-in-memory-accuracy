import torch
from utils.analog_core_simulator import *
from utils.noise import *

"""
Noise Wrapper for ResNet-18.

ResNetNoiseEstimation object  
"""

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

propagate_noise = True
verbose = True

class Conv2DNoiseEstimation:
    def __init__(self, stat, module_dict, noise_propagate_dict, conv_name, bn_name, conv_params, prev_layer=None):
        self.stat        = stat
        self.module_dict = module_dict
        self.noise_propagate_dict = noise_propagate_dict
        self.conv_name   = conv_name
        self.bn_name     = bn_name
        self.conv_params = conv_params
        self.prev_layer  = prev_layer

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        
        conv = analog_params | self.conv_params
        input_pdf = self.stat[f'{self.conv_name}.in'] if self.prev_layer is None else self.stat[f'{self.prev_layer}.out']

        conv |= {
            'input_pdf': input_pdf,
            'input_noise': input_noise
        }
        add_scale_to_analog_params(conv)
        conv_var, conv_stat = single_layer_conv2d_noise(**conv)

        propagated_var = 0
        # Propagate Noise to the output tensor
        if propagate_noise and self.bn_name in self.noise_propagate_dict:
            noise_propagate_factor = self.noise_propagate_dict[self.bn_name] # (Cout, *network_output)
            per_channel_var = conv_var.mean(dim=-1).mean(dim=-1) # (Cout)
            
            propagated_var = torch.einsum('c...,c->...', noise_propagate_factor, per_channel_var)

        if verbose:
            print(f"layer\t{self.conv_name}")
            print(f"conv_var\t{torch.mean(conv_var)}")
            print(f"propagated_var\t{torch.mean(propagated_var)}")

        return {
            'layer': self.conv_name,
            'propagated_var': propagated_var,
            'layer_var': conv_var,
            'var_stat': conv_stat
        }


class FcNoiseEstimation:
    def __init__(self, stat, module_dict, noise_propagate_dict, layer_name):
        fc_weight, fc_bias = module_dict[layer_name].weight, module_dict[layer_name].bias
        fc_weight = fc_weight.unsqueeze(-1).unsqueeze(-1)

        self.conv_params = {
            'weight': fc_weight,
            'stride': 1,
            'padding': 0
        }
        self.stat = stat
        self.module_dict = module_dict
        self.noise_propagate_dict = noise_propagate_dict
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
        add_scale_to_analog_params(conv)
        conv_var, conv_stat = single_layer_conv2d_noise(**conv)
        if isinstance(conv_var, torch.Tensor):
            conv_var = conv_var.flatten()
        
        # Flatten for FC output
        for k, v in conv_stat.items():
            if isinstance(v, torch.Tensor):
                conv_stat[k] = v.flatten()

        propagated_var = 0
        # Propagate Noise to the output tensor
        if propagate_noise and self.layer_name in self.noise_propagate_dict:
            noise_propagate_factor = self.noise_propagate_dict[self.bn_name] # (Cout, *network_output)
            per_channel_var = conv_var.mean(dim=-1).mean(dim=-1) # (Cout)
            
            propagated_var = torch.einsum('c...,c->...', noise_propagate_factor, per_channel_var).flatten()

        return {
            'layer': self.layer_name,
            'propagated_var': propagated_var,
            'layer_var': conv_var,
            'var_stat': conv_stat
        }


class BasicBlockNoiseEstimation:
    def __init__(self, stat, module_dict, noise_propagate_dict, layer_name):
        """
        - stat: dictionary of layer name -> PDF Object
        - module_dict: named modules dictionary
        - layer_name: name of the parent layer
        """
        self.stat = stat
        self.module_dict = module_dict
        self.layer_name = layer_name
        self.noise_propagate_dict = noise_propagate_dict

        self.conv_list: list[Conv2DNoiseEstimation] = []
        for i in range(2):
            conv_iter, bn_iter = f'conv{i+1}', f'bn{i+1}'
            conv_name = f'{layer_name}.{conv_iter}'
            bn_name   = f'{layer_name}.{bn_iter}'

            conv_params = get_fused_convbn_params(module_dict, layer_name, conv_iter, bn_iter)
            
            self.conv_list.append(
                Conv2DNoiseEstimation(stat, module_dict, noise_propagate_dict, conv_name, bn_name, conv_params)
            )

        self.downsample = None

        self.ds_layer_name = f'{layer_name}.downsample'
        ds_conv_name = f'{self.ds_layer_name}.0'
        ds_bn_name = f'{self.ds_layer_name}.1'

        # check if downsampling (!= identity) exists
        if f'{ds_conv_name}.in' in self.stat:
            downsample_params = get_fused_convbn_params(module_dict, self.ds_layer_name, '0', '1')
            self.downsample = Conv2DNoiseEstimation(stat, module_dict, noise_propagate_dict, ds_conv_name, ds_bn_name, downsample_params)

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        #TODO: elegant way to add quantization noise to each layer automatically
        result_list = []

        for i, conv in enumerate(self.conv_list):
            conv_result = conv(analog_params)
            result_list.append(conv_result)

        # DOWNSAMPLE (for residual connection)
        #ds_var = self..get_expected_value(input_noise)
        if self.downsample is not None:
            ds_result = self.downsample(analog_params)
            result_list.append(ds_result)

        return result_list

class LayerNoiseEstimation:
    def __init__(self, stat, module_dict, noise_propagate_dict, layer_name, num_blocks=2):
        """
        - stat: dictionary of layer name -> PDF Object
        - module_dict: named modules dictionary
        - layer_name: name of the parent layer
        """
        self.stat = stat
        self.module_dict = module_dict
        self.layer_name = layer_name
        self.noise_propagate_dict = noise_propagate_dict

        block_names = [f'{layer_name}.{i}' for i in range(num_blocks)]
        self.blocks = [
            BasicBlockNoiseEstimation(stat, module_dict, noise_propagate_dict, block_name) for block_name in block_names
        ]

    def __call__(self, analog_params: dict = default_params, input_noise: Noise = None)->torch.Tensor:
        output_noise = input_noise
        result_list =[]

        for i, block in enumerate(self.blocks):
            block_result_list = block(analog_params)
            result_list += block_result_list

        return result_list

class ResNetNoiseEstimation:
    def __init__(self, 
                stat: dict[str, PDF], 
                module_dict: dict[str, torch.nn.Module], 
                noise_propagate_dict: dict[str, torch.Tensor], 
                num_layers: int = 4
                ):
        """
        - stat: layer_name -> PDF Object
        - module_dict: layer_name -> torch Module
        - noise_propagate_dict: layer_name -> noise propagation tensor (shape = (C_out, *Network_output.shape))
        - num_layers: ResNet-18 number of layers
        """
        self.stat = stat
        self.module_dict = module_dict
        self.noise_propagate_dict = noise_propagate_dict
        #conv1, relu, maxpool, layers, avgpool, fc

        conv1_params = get_fused_convbn_params(module_dict, '', 'conv1', 'bn1')
        self.conv1 = Conv2DNoiseEstimation(stat, module_dict, noise_propagate_dict, 'conv1', 'bn1', conv1_params)
        
        self.layer_names = [f'layer{i+1}' for i in range(num_layers)]
        self.layers = [
            LayerNoiseEstimation(stat, module_dict, noise_propagate_dict, layer_name) for layer_name in self.layer_names
        ]
        self.fc = FcNoiseEstimation(stat, module_dict, noise_propagate_dict, 'fc')

    def __call__(self, analog_params: dict = default_params, 
                 input_noise: Noise = None)->torch.Tensor:
        """
        Run a noise estimation for rull ResNet-18
        Parameters:
        - analog_params (dict): analog core parameters (sent to utils.analog_core_simulator.
        - input_noise (Noise): Additional Noise object to network input
        Returns:
        - torch.Tensor: List of results from each convolution layer.
            Individual result is a dict of:
                * layer: convolution layer name
                * propagated_var: noise tensor propagated to the network output
                * layer_var: noise tensor of corresponding convolution layer output
                * var_stat: detailed noise breakdown of layer_var (weight noise, input noise, etc.)
        """
        result_list = []

        # Conv1, Bn1
        conv1_result = self.conv1(analog_params)
        result_list.append(conv1_result)

        # NOTE: maxpool is ignored since digitally calculated

        # Layer 1-4
        for i, layer in enumerate(self.layers):
            layer_result_list = layer(analog_params)
            result_list += layer_result_list
        
        # Average Pool is ignored since digitally calculated

        # FC layer
        fc_result = self.fc(analog_params) #TODO: Full-precision? (default params)
        # Since fc_layer is the last, set the propagated noise to the network output as itself
        fc_result['propagated_var'] = fc_result['layer_var']
        result_list.append(fc_result)

        return result_list