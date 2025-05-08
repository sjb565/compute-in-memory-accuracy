import matplotlib.pyplot as plt
import numpy as np

# from functools import partial
# from typing import Any, List, Optional, Type, Union

import torch
import torchvision
import matplotlib.pyplot as plt

# For Model
import pickle

# For Labeling
from scipy.stats import linregress

# For exporting results to csv
import itertools
import csv
import sys, os
from datetime import datetime

file_dir = os.path.dirname(os.path.abspath(__file__))
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from utils.analog_core_simulator import *
from utils.noise import *
from utils.single_layer_simulator import run_first_layer_testing, write_to_csv
from resnet.noise_resnet import *
from resnet.dataset import CustomImageNetDataset, transform, DataLoader


# Turn off Training
torch.set_grad_enabled(False)


epsilon=1e-15
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

if __name__ == "__main__":
    

    # override ResNet18 model for probing, bring pretrained weights
    state_dict = torch.load(os.path.join(file_dir,'resnet/resnet18_state_dict.pth'))
    model = torchvision.models.resnet18()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    module_dict = dict(model.named_modules())

    # Initialize dataset and dataloader
    val_dir = os.path.join(file_dir, '../../val')
    dataset = CustomImageNetDataset(val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,) #num_workers=4

    # Import model from main.py
    # model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)  # Load a pre-trained ResNet-18 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load PDF
    with open(os.path.join(file_dir, 'resnet/stat.pkl'), 'rb') as f:
        stat = pickle.load(f)

    # load Noise Propagation Statistics
    with open(os.path.join(file_dir, 'resnet/noise_prop.pkl'), 'rb') as f:
        noise_propagate_dict = pickle.load(f)

    def dict_to_device(dict, device=device):
        for key, val in dict.items():
            if isinstance(val, PDF):
                val.p_zero = np_to_torch(val.p_zero, device)
                val.range  = np_to_torch(val.range, device)
                val.pdf    = np_to_torch(val.pdf, device)
            elif isinstance(val, torch.Tensor):
                dict[key] = np_to_torch(val, device)

    # convert any numpy array
    dict_to_device(stat)
    dict_to_device(noise_propagate_dict)

    # Get conv1 pdf
    conv1_in = stat['conv1.in']

    name = 'conv1'
    conv_params = get_fused_convbn_params(module_dict, parent_layer_name='', conv_name='conv1', bn_name='bn1', epsilon = epsilon)
    conv1_params = default_params | conv_params

    conv1_params = conv1_params | {
        'input_pdf': conv1_in,
        'input_scale': None,
        'input_zp': 0,
        'weight_scale': None,
        'weight_zp': 0,
        'input_bits': None, 'weight_bits': None,
    }

    params_sweep_list_raw = {
        'input_bits': [None], #[i for i in `range(1, 20)],
        'weight_bits': [6, 8, 10, 12, 14], #[32], #[i for i in range(1, 20)],

        #'input_noise': None,
        'weight_value_noise': [None],
        'weight_physical_noise':[None],# [None,ProportionalNoise(0.01), ProportionalNoise(0.04), ProportionalNoise(0.08), ProportionalNoise(0.16)], #+ [ProportionalNoise(i) for i in (0.02, 0.04, 0.08, 0.16,0.32, 0.64)], #[None, ProportionalNoise(0.01), ProportionalNoise(0.1)], (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.09, 0.1, 0.11, 0.12) # (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.12, 0.16, 0.32, 0.64)
        'differential_weight': [True],
        'weight_slice_lsb_to_msb': [None], #[1, 2, 4, 8],
        'adc_noise': [None], #[GaussianNoise(i**2) for i in (1, 5, 10, 15, 20)] #[None, ProportionalNoise(0.1)],
    }

    # 1. Make all combinations
    keys = list(params_sweep_list_raw.keys())
    values_list = [params_sweep_list_raw[key] for key in keys]
    combinations = list(itertools.product(*values_list))

    params_list = [dict(zip(keys, values)) for values in combinations]
    exclusive_params_list = []

    est_noise_tensors = []
    est_noise_stats   = []
    est_noise_list = []
    est_snr_list   = []

    resnet_noise = ResNetNoiseEstimation(
        stat, 
        module_dict,
        noise_propagate_dict
    )
    result_list = resnet_noise(conv1_params)
    total_var = 0

    first_layer_snr_list, last_layer_snr_list = [], []
    # Average output square precalculated with ImageNet-100 Dataset
    output_square_first = torch.load(os.path.join(file_dir,'resnet/first_layer_output_square.pt'))
    output_square_final = torch.load(os.path.join(file_dir,'resnet/final_layer_output_square.pt'))

    # 2. Run Simulation
    for params in params_list:
        params = conv1_params | params
        exclusive_params_list.append(params)

        add_scale_to_analog_params(params)
        result_list = resnet_noise(params)

        last_var = 0
        for result in result_list:
            last_var += result['propagated_var']
        first_var = result_list[0]['layer_var']

        first_snr = 10*np.log10((torch.mean(output_square_first)/torch.mean(first_var+1e-13)).item())
        last_snr = 10*np.log10((torch.mean(output_square_final)/torch.mean(last_var+1e-13)).item())

        first_layer_snr_list.append(first_snr)
        last_layer_snr_list.append(last_snr)

        print(f"first\t{first_snr}\tlast\t{last_snr}")
    
    # 5. Save to csv
    save_csv = True
    save_raw_data = False
    if save_csv or save_raw_data:
        data_dir = os.path.join(file_dir, 'results','resnet')
        test_name = 'weight_bits'
        os.makedirs(data_dir, exist_ok=True)
        dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = os.path.join(data_dir, '_'.join([dt_string, test_name]))
        os.makedirs(output_dir, exist_ok=True)

    if save_csv:
        csv_file_path = os.path.join(output_dir, f'summary.csv')
        write_to_csv(csv_file_path, keys, params_list, {'first_layer_snr_db': first_layer_snr_list, 'last_layer_snr_db': last_layer_snr_list})
