import torch
import torch.nn.functional as F
import csv
from utils.analog_core_simulator import get_location_independent_input

# prevent divide by zero for SNR
epsilon = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_first_layer_testing(model, dataloader, conv_params, params_list, early_stop_at_batch = None, gen_independent_input=False, epsilon=epsilon):
    """
    Simulates noise and quantization effects on the first layer of a model and estimates per-configuration SNR.

    Args:
        model (torch.nn.Module): The model containing the first convolutional layer.
        dataloader (Iterable): Provides input batches and labels.
        conv_params (dict): Contains 'weight', 'bias', 'stride', and 'padding' for the conv layer.
        params_list (list of dict): Each dict defines a test configuration with optional noise and quantization.
        early_stop_at_batch (int, optional): If set, limits the number of batches processed.
        gen_independent_input (bool): Whether to generate synthetic inputs independent of spatial location.
        epsilon (float): Small constant to prevent division by zero in SNR computation.

    Returns:
        noise_list (list of Tensor): Per-test squared error accumulations.
        snr_list (list of Tensor): Estimated SNR per test configuration.
        output_square (Tensor): Average squared ideal output, used for SNR baseline.
    """

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
                output_square = (outputs**2).sum(dim=0)
                # Cautious: outputs[1:] since noise shaped should not be `Batched'
                noise_list = [torch.zeros(outputs.shape[1:]) for i in range(num_tests)]
            else:
                output_square += (outputs**2).sum(dim=0)

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

    # Get Average SNR and Noise
    for test_idx, noise in enumerate(noise_list):
        noise = noise / num_samples
        noise_list[test_idx] = noise
        snr_list.append(
            output_square/(noise+epsilon)
        )
    return noise_list, snr_list, output_square

def write_to_csv(fn, keys, kwargs_list, result_dict):
    """
    Saves experiment results to a CSV file.

    Args:
        fn (str): Output filename. '.csv' is appended if missing.
        keys (list of str): Keys to extract from each kwargs dict as CSV columns.
        kwargs_list (list of dict): List of parameter dictionaries used in experiments.
        result_dict (dict of str -> list): Maps result names to lists of values, aligned with kwargs_list.

    Writes:
        CSV file with each row as parameter + result combination.
    """

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

    
    