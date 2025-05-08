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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def write_to_csv(fn, keys, kwargs_list, result_dict):
    if not ('.csv' in fn):
        fn = fn+'.csv'
    result_keys = list(result_dict.keys())
    result_values_transposed = [result_dict[key] for key in result_keys]
    result_values = []
    for values in zip(*result_values_transposed):
        result_values.append(
            list(values)
    )

    with open(fn, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(keys + result_keys)
        # Write each kwargs + result
        for kwargs, result in zip(kwargs_list, result_values):
            row = [kwargs[key] for key in keys] + result
            writer.writerow(row)

    print(f"successfully saved to {fn}")


if __name__ == "__main__":
    

    # override ResNet18 model for probing, bring pretrained weights
    state_dict = torch.load(os.path.join(file_dir,'resnet/resnet18_state_dict.pth'))
    model = torchvision.models.resnet18()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    module_dict = dict(model.named_modules())

    # Initialize dataset and dataloader
    val_dir = os.path.join(file_dir, '../../val') # NOTE: Root directory of Dataset
    dataset = CustomImageNetDataset(val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,) #num_workers=4

    # Import model from main.py
    # model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)  # Load a pre-trained ResNet-18 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load PDF
    with open(os.path.join(file_dir, 'resnet/stat.pkl'), 'rb') as f:
        stat = pickle.load(f)

    def dict_to_device(dict, device=device):
        # send tensor attributes to target device
        for key, val in dict.items():
            if isinstance(val, PDF):
                val.range  = np_to_torch(val.range, device)
                val.pdf    = np_to_torch(val.pdf, device)
                if val.p_zero is not None:
                    val.p_zero = np_to_torch(val.p_zero, device)
            elif isinstance(val, torch.Tensor):
                dict[key] = np_to_torch(val, device)

    # convert any numpy array
    dict_to_device(stat)

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
        'use_spatial_correlation': False,
    }
    
    params_sweep_list_raw = {
        'input_bits': [None], #[i for i in `range(1, 20)],
        'weight_bits': [6, 8, 10, 12, 14], #[32], #[i for i in range(1, 20)],

        'weight_physical_noise': [None],    # ex) [ProportionalNoise(i) for i in (0.02, 0.04, 0.08, 0.16,0.32, 0.64)], #[None, ProportionalNoise(0.01), ProportionalNoise(0.1)], (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.09, 0.1, 0.11, 0.12) # (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.12, 0.16, 0.32, 0.64)
        'differential_weight': [True],
        'weight_slice_lsb_to_msb': [None],  # ex) [1, 2, 4, 8],
        'adc_noise': [None],                # ex) [GaussianNoise(i**2) for i in (1, 5, 10, 15, 20)] #[None, ProportionalNoise(0.1)],
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

    # 2. Run Simulation
    for params in params_list:
        params = conv1_params | params
        exclusive_params_list.append(params)

        add_scale_to_analog_params(params)
        noise, stats = single_layer_conv2d_noise(**params)

        #est_noise_stats.append(stats)
        est_noise_tensors.append(noise)
        est_noise_list.append(noise.mean().item())
    
    # 3. Actual Simulation of Multiple Samples ResNet Result
    noise_tensors, snr_tensors, output_square = run_first_layer_testing(
        model, dataloader, conv_params, exclusive_params_list, early_stop_at_batch=100, gen_independent_input=False
    )

    def trimmed_mean(tensor, trim_ratio=0.1):
        sorted_tensor, _ = torch.sort(tensor.flatten())
        n = sorted_tensor.shape[0]
        trim_n = int(n * trim_ratio)
        trimmed = sorted_tensor[trim_n : n - trim_n]
        return trimmed.mean().item()

    # 4. Calculate mean value of SNRs and Noise (=Output Variance from the Ideal)
    est_snr_tensors = list(map(lambda x: output_square/(x+epsilon), est_noise_tensors))
    est_snr_list = list(map(lambda x: trimmed_mean(x), est_snr_tensors)) #torch.mean(x).item()
    
    noise_list = list(map(lambda x: trimmed_mean(x),  noise_tensors))
    snr_list = list(map(lambda x: trimmed_mean(x),  snr_tensors))

    # 5. Save to csv
    save_csv = True
    save_raw_data = False
    if save_csv or save_raw_data:
        data_dir = os.path.join(file_dir, 'results','first_conv1_layer')
        test_name = 'test_name'
        os.makedirs(data_dir, exist_ok=True)
        dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = os.path.join(data_dir, '_'.join([dt_string, test_name]))
        os.makedirs(output_dir, exist_ok=True)

    if save_csv:
        csv_file_path = os.path.join(output_dir, f'summary.csv')
        write_to_csv(csv_file_path, keys, params_list, {'estimated noise': est_noise_list, 'actual noise': noise_list, 'estimated_snr': est_snr_list, 'snr': snr_list})

    # 6. Save detailed test results
    if save_raw_data:
        keys = ['est_noise_tensors', 'est_noise_list', 'est_snr_tensors', 'est_snr_list'
                'noise_tensors', 'noise_list', 'snr_list', 'snr_tensors'] #'est_noise_stats'
        values = [est_noise_tensors, est_noise_list, est_snr_tensors, est_snr_list,
                noise_tensors, noise_list, snr_list, snr_tensors]
        for k, v in zip(keys, values):
            torch.save(v, os.path.join(output_dir, f"{k}.pt"))

        print(f"Detailed Test Results successfully saved to {output_dir}!")