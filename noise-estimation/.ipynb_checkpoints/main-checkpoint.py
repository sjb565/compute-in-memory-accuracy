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
import os
from datetime import datetime

from data_fetch_utils import *
from utils import *
from noise import *
from noise_resnet import *

# Turn of Training
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

def run_first_layer_testing(model, dataloader, conv_params, params_list, early_stop_at_batch = None, gen_independent_input=False):
    num_tests, num_samples = len(params_list), 0
    noise_list    = []
    snr_list      = []
    output_square = None 

    # get conv_params
    weight, bias, stride, padding = [conv_params[key] for key in (
        'weight', 'bias', 'stride', 'padding'
    )]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            num_samples += inputs.shape[0]

            if gen_independent_input:
                # testing random samples
                inputs = get_location_independent_input(inputs.shape[0], params_list[0]['input_pdf'], device)

            # Early stop for fast testing
            if early_stop_at_batch and batch_idx > early_stop_at_batch:
                break

            # Get Ideal Outputs
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = F.conv2d(inputs, weight, bias, stride, padding)

            # set output_square
            if output_square is None:
                output_square = outputs**2
                # Cautious: outputs[1:] since noise shaped should not be `Batched'
                noise_list = [torch.zeros(outputs.shape[1:]) for i in range(num_tests)]
            else:
                output_square += outputs**2

            """
                # Quantization factors
                input_bits: int = None, input_scale: float = None, input_zp: int = 0,
                weight_bits: int = None, weight_scale: float = None, weight_zp: int = 0,

                # Noise Objects
                input_noise: Noise = None, # std unit: [g.t.]
                weight_value_noise: Noise = None, # std unit: [g.t.]
                weight_physical_noise: Noise = None, # std unit: [bits]
                adc_noise: Noise = None, # std unit: [bits]
            """
            for test_idx, params in enumerate(params_list):
                c_inputs, c_weight = inputs.clone(), weight.clone()

                # Add noise
                if params['input_noise']:
                    input_noise = params['input_noise']
                    input_std = torch.sqrt(input_noise(c_inputs))
                    noise = torch.randn_like(c_inputs, device=device) * input_std
                    # get variance noise and add to input
                    c_inputs += noise
                
                if params['weight_value_noise']:
                    weight_noise = params['weight_value_noise']
                    weight_std = torch.sqrt(weight_noise(c_weight))
                    noise = torch.randn_like(c_weight, device=device) * weight_std
                    # get variance noise and add to input
                    c_weight += noise

                # Quantization
                if params['input_bits']:
                    scale = params['input_scale']
                    c_inputs = torch.round(inputs/scale) * scale
                if params['weight_bits']:
                    scale = params['weight_scale']
                    c_weight = torch.round(weight/scale) * scale

                # If Weight Physical Noise
                if params['weight_physical_noise'] and params['weight_bits'] and params['differential_weight']:
                    scale, zp = params['weight_scale'], params['weight_zp']
                    q_weight = torch.round(c_weight/scale) + zp

                    weight_physical_noise = params['weight_physical_noise']
                    weight_std = torch.sqrt(weight_physical_noise(q_weight))
                    q_noise = torch.randn_like(q_weight, device=device) * weight_std
                    c_weight = scale * (q_weight+q_noise-zp)

                # Applied to Output
                if params['adc_noise']:
                    adc_noise_obj = params['adc_noise']
                    abs_out_list = []

                    bias_unsqueezed = bias.view(1, -1, *([1] * (outputs.ndim - 2)))  # bias is (C,)

                    if (c_inputs<0).any():
                        #TODO: Change for only non-zero inputs
                        in_pos = torch.where(c_inputs > 0, c_inputs, torch.zeros_like(c_inputs))
                        in_neg = torch.where(c_inputs < 0, c_inputs, torch.zeros_like(c_inputs))

                        out_pos = F.conv2d(in_pos, c_weight, bias=None, stride=stride, padding=padding)
                        out_neg = F.conv2d(in_neg, c_weight, bias=None, stride=stride, padding=padding)

                        abs_out_list = [out_pos, out_neg]

                    else:  
                        c_outputs_no_bias = F.conv2d(c_inputs, c_weight, bias=None, stride=stride, padding=padding)
                        abs_out_list.append(
                            c_outputs_no_bias
                        )
                    
                    c_outputs = sum(abs_out_list) + bias_unsqueezed

                    for out in abs_out_list:
                        total_scale = params['input_scale'] * params['weight_scale']
                        q_out = out/total_scale
                        adc_std = torch.sqrt(adc_noise_obj(q_out))
                        adc_noise = torch.randn_like(outputs, device=device) * adc_std
                        c_outputs += adc_noise * (total_scale)
                
                else:
                    c_outputs = F.conv2d(c_inputs, c_weight, bias, stride, padding)

                # Add error
                c_error = ((c_outputs - outputs)**2).sum(dim=0, keepdim=False)
                noise_list[test_idx] += c_error
                
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch {batch_idx + 1}') #: Top-1 Acc: {top1_total / total_samples:.4f}, Top-5 Acc: {top5_total / total_samples:.4f}')

    output_square = output_square / num_samples
    for test_idx, noise in enumerate(noise_list):
        noise = noise / num_samples
        noise_list[test_idx] = noise
        snr_list.append(
            output_square/(noise+epsilon)
        )
    return noise_list, snr_list, output_square
    
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
    state_dict = torch.load('resnet18/resnet18_state_dict.pth')
    model = torchvision.models.resnet18()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    module_dict = dict(model.named_modules())

    # Initialize dataset and dataloader
    val_dir = '../val'
    dataset = CustomImageNetDataset(val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False,) #num_workers=4

    # Import model from main.py
    # model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)  # Load a pre-trained ResNet-18 model
    useGPU = True
    device = torch.device('cuda' if torch.cuda.is_available() and useGPU else 'cpu')
    model.to(device)

    # load PDF
    with open('../stat.pkl', 'rb') as f:
        stat = pickle.load(f)

    # convert any numpy array
    for key, val in stat.items():
        if isinstance(val, PDF):
            if isinstance(val.p_zero, np.ndarray):
                val.p_zero = np_to_torch(val.p_zero, device=device)
                val.range  = np_to_torch(val.p_zero, device=device)
                val.pdf    = np_to_torch(val.pdf, device=device)

    # Get conv1 pdf #TODO: For normalized pdfs, replace this totally ****
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
        'input_bits': 8, 'weight_bits': 8,
    }


    add_scale_to_analog_params(conv1_params)
    resnet_noise = ResNetNoiseEstimation(
        stat, module_dict
    )
    var, stat = resnet_noise(conv1_params)
    print(var)

    params_sweep_list_raw = {
        'input_bits': [12], #[i for i in `range(1, 20)],
        'weight_bits': [2, 4, 6, 8, 10, 12], #[32], #[i for i in range(1, 20)],

        #'input_noise': None,
        #'weight_value_noise': None,
        'weight_physical_noise': [None], #+ [ProportionalNoise(i) for i in (0.02, 0.04, 0.08, 0.16,0.32, 0.64)], #[None, ProportionalNoise(0.01), ProportionalNoise(0.1)], (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.09, 0.1, 0.11, 0.12) # (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.12, 0.16, 0.32, 0.64)
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
        model, dataloader, conv_params, exclusive_params_list, 100
    )

    # 4. Calculate mean value of SNRs and Noise (=Output Variance from the Ideal)
    est_snr_tensors = list(map(lambda x: output_square/(x+epsilon), est_noise_tensors))
    est_snr_list = list(map(lambda x: torch.mean(x).item(), est_snr_tensors))
    
    noise_list = list(map(lambda x: torch.mean(x).item(),  noise_tensors))
    snr_list = list(map(lambda x: torch.mean(x).item(),  snr_tensors))

    # 5. Save to csv
    save_csv = True
    save_raw_data = False
    if save_csv or save_raw_data:
        data_dir = '../data'
        test_name = 'weight_bits'
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

    raise KeyError

    for name in noise_name_list:
        noise_est_result = noise_est[name] 
        noise_measure_result = (noise_measure[name] / num_samples) 
        noise_est_flat = noise_est_result.flatten()
        noise_measure_flat = noise_measure_result.flatten()

        avg_noise = (torch.sum(noise_measure_flat))**2/(noise_measure_flat.dim())
        noise_error = torch.sum((noise_est_flat-noise_measure_flat)**2)/noise_measure_flat.dim()
        print(avg_noise/noise_error)
        print(np.sqrt(noise_error/avg_noise))

        slope, intercept, r_value, p_value, std_err = linregress(noise_est_flat, noise_measure_flat)

        print(f'{torch.sum(noise_est_flat)/(noise_est_flat.dim())}')
        print(f'{torch.sum(noise_measure_flat)/(noise_measure_flat.dim())}')
        print(f"Exp Type: {name}")    
        # Print the results
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R^2: {r_value**2}")
        print("==========================")

        plt.title(name)
        # plt.plot(noise_measure_result.flatten(), noise_measure_result.flatten(), linestyle='--', linewidth=1.0, alpha=0.7)
        for c in range(64):
            plt.scatter(noise_measure_result[c,:,:].flatten(), noise_est_result[c,:,:].flatten(), alpha = 0.005) #noise_est_result[c,:,:].flatten()
        plt.show()
