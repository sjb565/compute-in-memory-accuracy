import torch
import torch.nn.functional as F
import numpy as np

# if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_to_np(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    Parameters:
    - tensor (torch.Tensor): The input tensor.
    Returns:
    - numpy.ndarray: The converted NumPy array.
    """
    return tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.detach().numpy()

def np_to_torch(array, device=device, dtype=None):
    """
    Convert a NumPy array to a PyTorch tensor.
    Parameters:
    - array (numpy.ndarray): The input array.
    - device (torch.device, optional): The device to move the tensor to. Defaults to None.
    Returns:
    - torch.Tensor: The converted tensor.
    """
    if isinstance(array, np.ndarray):
        if dtype:
            array = torch.tensor(array, dtype=dtype)
        else:
            array = torch.tensor(array)
    
    if device:
        array = array.to(device)

    return array

def pad_to_size(x, pad_dims, mode='constant', value=0):
    if isinstance(pad_dims, int):
        pad_dims = (pad_dims,) * 4 # all four sides of the tensor

    return F.pad(x, pad_dims, mode=mode, value=value)

# NOTE: Importing here for noise module to get util functions
from utils.noise import Noise, PDF, GaussianNoise, ReluNoise
from utils.distribution_handler import *

def quantize_tensor(tensor, scale, zp=0):
    # Quantize input tensor using quantization scale and zeropoint
    if isinstance(scale, float):
        return torch.round(tensor/scale) + zp
    else:
        # per-channel quantization
        scale_inv = 1/scale

        while zp.dim() < tensor.dim():
            zp = zp.unsqueeze(-1)

        return torch.round(torch.einsum('i...,i->...', tensor, scale_inv)) + zp

def slice_tensor_lsb_to_msb(
        tensor,
        bit_slice_lsb_to_msb: int | list[int],
        num_bits: int
    ):
    # If constant slicing, repeat the constant slicing until it reaches num_bits
    if isinstance(bit_slice_lsb_to_msb, int):
        # Expand the slicing pattern
        slicing_pattern = []
        current_bits = 0

        while current_bits < num_bits:
            slicing_pattern.append(bit_slice_lsb_to_msb)
            current_bits += bit_slice_lsb_to_msb

        bit_slice_lsb_to_msb = slicing_pattern

    # Perform slicing
    sliced_weights = []
    bit_offset = 0
    tensor = tensor.to(dtype=torch.int)

    for bits in bit_slice_lsb_to_msb:
        mask = (1 << bits) - 1
        sliced = (tensor >> bit_offset) & mask
        sliced_weights.append(sliced)
        bit_offset += bits

    return sliced_weights, bit_slice_lsb_to_msb
    
def calc_weight_physical_noise(
    weight: torch.Tensor, weight_scale, weight_zp, weight_bits,
    weight_slice_lsb_to_msb,
    weight_physical_noise: Noise,
    input_square: torch.Tensor,
    differential_weight: bool= True,
    stride: int = 1,
    ):
    # Quantize weight and input
    weight_q = quantize_tensor(weight, weight_scale, weight_zp)

    if differential_weight:
        # for differential, we only need abs(weight) since it's only noise
        weight_q = abs(weight_q)
        w_pos = weight_q * (weight_q > 0)
        w_neg = weight_q * (weight_q < 0)
    else:
        # For Offset Subtraction
        weight_q += 1<<(weight_bits-1) 
    
    if weight_slice_lsb_to_msb:
        sliced_weights, weight_slice_lsb_to_msb  = slice_tensor_lsb_to_msb(weight_q, weight_slice_lsb_to_msb, weight_bits)
    else:
        sliced_weights = [weight_q]
        weight_slice_lsb_to_msb = [weight_bits]
    
    if differential_weight:
        (sliced_pos, _), (sliced_neg, _) = (slice_tensor_lsb_to_msb(w, weight_slice_lsb_to_msb, weight_bits) for w in (w_pos, w_neg))
        sliced_diff_weights = list(zip(sliced_pos, sliced_neg))
    
    noise = 0
    scale, slice_scales = 1, []
    for sliced, bits in zip(sliced_weights, weight_slice_lsb_to_msb):
        if weight_physical_noise:
            weight_physical_var = weight_physical_noise(sliced)
            # Accumulate sliced weight noise + consider 2^ith bit scaling
            physical_noise = F.conv2d(input_square, weight_physical_var, bias=None, stride=stride) * (scale**2)

            noise += physical_noise

        slice_scales.append(scale)
        scale = (scale << bits)
    
    # return positive and negative sliced weights for differential encoding
    if differential_weight:
        sliced_weights = sliced_diff_weights

    # Scale the noise unit back from integer to CNN floating point (note: only weight is quantized)
    return noise * (weight_scale**2), sliced_weights, slice_scales
    
def calc_input_value_noise(
        input_pdf: PDF,
        weight: torch.Tensor,
        input_bits: int | None, input_scale: float | None,
        input_noise: Noise | None,
        padding: int = 0,
        stride: int = 1,
        n_sample: int = 200
    ):
    input_noise_samples = 0
    use_relu = False

    # No Noise
    if not (input_bits or input_noise):
        return 0

    # Quantization Noise
    if input_bits:
        assert input_scale, f"input scale should exist with declared input bits = {input_bits}"
        quantization_noise = GaussianNoise((input_scale**2)/12) 
        input_noise_samples = get_gaussian_samples(
            input_pdf, quantization_noise, n_sample
        )

    if input_noise:
        # Check if Noise should be suppressed by ReLU
        if isinstance(input_noise, ReluNoise):
            use_relu = True

        # Add padding since Gaussian Noise Tensor size may differ from padded input
        input_noise = input_noise.copy()
        input_noise.add_padding(padding)
        input_val_noise_samples = get_gaussian_samples(
            input_pdf, input_noise, n_sample
        )
        # Add two samples
        if isinstance(input_noise_samples, torch.Tensor):
            input_noise_samples += input_val_noise_samples
        else:
            input_noise_samples = input_val_noise_samples

    # Apply ReLU if needed (Sparse Noise Simulation)
    if use_relu:
        input_noise_samples = relu_suppress_neg_noise(
            input_pdf, input_noise_samples
        )
    
    # Calculate output variance for each output layer
    output_noise_samples = F.conv2d(input_noise_samples, weight, bias=None, stride=stride)
    
    return output_noise_samples.var(dim=0, correction=1)
    


def find_pdf_bin(
        value, pdf_ranges
    ):
    return np.searchsorted(pdf_ranges, value, side='right') - 1

def get_uniform_bin_index(
        value, bin_width, n_bin
    ):
    #return min(value // bin_width, n_bin - 1)
    return int(value / bin_width)

def slice_inputs(
        input_pdf, input_scale, input_zp, input_bits,
        input_slice_lsb_to_msb,
        num_samples = 10
    ):
    quantized_input_pdf = input_pdf.copy()
    quantized_input_pdf.range = quantize_tensor(quantized_input_pdf.range, input_scale, input_zp)

    if input_slice_lsb_to_msb is None:
        input_slice_lsb_to_msb = [input_bits]
        #return [input_pdf], [1]
    elif isinstance(input_slice_lsb_to_msb, int):
        # Expand the slicing pattern
        full_bits = input_bits
        slice_size = input_slice_lsb_to_msb
        input_slice_lsb_to_msb = [slice_size] * (full_bits // slice_size)
        if full_bits % slice_size:
            input_slice_lsb_to_msb.append(full_bits % slice_size)

    num_slices = len(input_slice_lsb_to_msb)
    sliced_input_range = [torch.linspace(0, 2**b, min(quantized_input_pdf.n_bin, 2**b) + 1) for b in input_slice_lsb_to_msb]
    sliced_input_pdf = [torch.zeros((min(quantized_input_pdf.n_bin, 2**b), *quantized_input_pdf.pdf.shape[1:])) for b in input_slice_lsb_to_msb]
    sliced_p_zero = torch.zeros((num_slices, *quantized_input_pdf.pdf.shape[1:]))

    masks = [(1 << bits) - 1 for bits in input_slice_lsb_to_msb]
    shifts = np.cumsum([0] + input_slice_lsb_to_msb[:-1])
    bin_widths = [(2 ** bits) / len(ranges[:-1]) for bits, ranges in zip(input_slice_lsb_to_msb, sliced_input_range)]
    n_bins = [len(ranges) - 1 for ranges in sliced_input_range]


    def update_pdf(samples, p):
        """
        samples: Tensor of shape (num_samples,) — sampled quantized values
        p: Tensor of shape (num_idx,) — probabilities for each idx
        """
        p_flat = p.flatten()
        num_samples = samples.size(0)
        num_idx = p_flat.size(0)
        p_expand = p_flat.unsqueeze(0).expand(num_samples, num_idx).float()
        samples_expand = samples.unsqueeze(-1).expand(num_samples, num_idx)
        for i, (shift, mask, width, nb) in enumerate(zip(shifts, masks, bin_widths, n_bins)):
            sliced_samples = (samples_expand >> shift) & mask  # shape: (num_samples, num_idx)

            # Zero slice probability
            zero_counts = (sliced_samples == 0).sum(dim=0)  # shape: (num_idx,)
            sliced_p_zero[i].view(-1).add_(zero_counts.float() * p_flat)

            # Non-zero mask
            nonzero_mask = sliced_samples != 0

            p_mask = p_expand * nonzero_mask.view(num_samples, num_idx)  # broadcast mask over num_idx dims

            sliced_input_pdf[i].view(nb, -1).scatter_add_(0, (sliced_samples/width).long(), p_expand)


    #def update_pdf(samples, idx, p):
    #    # takes a value generated by monte carlo, slices it and adds probability to corresponding buckets 
    #    for i, (shift, mask, width, nb) in enumerate(zip(shifts, masks, bin_widths, n_bins)):
    #        sliced_samples = (samples >> shift) & mask
    #        sliced_p_zero[i][idx] += torch.count_nonzero(sliced_samples == 0) * p

    #        #bin_indices = np.minimum((sliced_samples[sliced_samples != 0] / width).astype(int), nb - 1)
    #        bin_indices = (sliced_samples[sliced_samples != 0] / width).long()  # Convert to int
    #        np.add.at(sliced_input_pdf[i, :, *idx], bin_indices, p)
    #        #torch.scatter_add_(sliced_input_pdf[i], 0, bin_indices.unsqueeze(0), p)


    for b in range(quantized_input_pdf.n_bin):
        #lo, hi = sorted(quantized_input_pdf.range[b:b+2])
        bounds = quantized_input_pdf.range[b:b+2].long()
        lo, hi = torch.min(bounds).item(), torch.max(bounds).item()
        pdf_slice = quantized_input_pdf.pdf[b]
        samples = torch.randint(lo, hi, (num_samples,))
        p = pdf_slice / num_samples
        update_pdf(samples, p)
        #for idx in np.ndindex(pdf_slice.shape):
        #    p = pdf_slice[idx] / num_samples
        #    update_pdf(samples, idx, p)
    # skip if input p_zero doesn't exist
    if quantized_input_pdf.p_zero is not None:
        update_pdf(torch.tensor(input_zp), quantized_input_pdf.p_zero)
        #for idx in np.ndindex(quantized_input_pdf.p_zero.shape):
        #    update_pdf(torch.tensor(input_zp), idx, quantized_input_pdf.p_zero[idx])

    scale, slice_scales = 1, []
    for bits in input_slice_lsb_to_msb:
        slice_scales.append(scale)
        scale <<= bits
    
    return [PDF(sliced_input_range[i], sliced_input_pdf[i], sliced_p_zero[i]) for i in range(input_bits)], slice_scales

def add_scale_to_analog_params(params):
    """
    params should include: input_pdf, weight, input_bits, weight_bits
    """
    input_bits, weight_bits = params['input_bits'], params['weight_bits']
    input_pdf, weight = params['input_pdf'], params['weight']

    # get input, weight scale depending on number of input bits
    if input_bits:
        input_min, input_max = torch.min(input_pdf.range).item(), torch.max(input_pdf.range).item()
        params['input_scale'] = (input_max - input_min) / (1<<input_bits)

    if weight_bits:
        weight_min, weight_max = torch.min(weight).item(), torch.max(weight).item()
        params['weight_scale'] = (weight_max - weight_min) / (1<<weight_bits)

# Noise estimation functions

def single_layer_conv2d_noise(
        # (N, Cin, H, W) x (Cin, Cout, H, W)
        input_pdf: PDF,
        weight: torch.Tensor, bias: torch.Tensor = None,

        # Quantization factors
        input_bits: int = None, input_scale: float = None, input_zp: int = 0,
        weight_bits: int = None, weight_scale: float = None, weight_zp: int = 0,

        # Noise Objects
        input_noise: Noise = None, # std unit: [g.t.]
        use_spatial_correlation: bool = True, # spatial correlation/locality of inputs
        weight_value_noise: Noise = None, # std unit: [g.t.]
        weight_physical_noise: Noise = None, # std unit: [bits]
        adc_noise: Noise = None, # std unit: [bits]

        # Physical Configuration
        num_rows: int = None,

        padding: int | list[int] | tuple[int] = 0,
        stride: int = 1,

        input_slice_lsb_to_msb: int | list[int]= None,
        weight_slice_lsb_to_msb: int | list[int]= None,

        # Encoding Specifications
        differential_weight = True,
    ):
    # Get Dimensions
    n_bin, Cin, H, W = input_pdf.pdf.shape
    Cout, Cin, R, S = weight.shape #TODO: Check if Cin, Cout order is correct!

    # Calculate Output Dimensions
    P = (H + 2*padding - R) // stride + 1
    Q = (W + 2*padding - S) // stride + 1
    output_var = torch.zeros((Cout, P, Q), device=device)

    # Pad input pdfs before tiling
    input_pdf = input_pdf.add_padding(padding=padding, value=0)

    # Initial value = 0
    input_val_noise_est, weight_val_noise_est, weight_physical_noise_est, input_physical_noise_est, adc_noise_est= 0, 0, 0, 0, 0

    weight_square = (weight**2)
    input_square_ev = input_pdf.get_expected_value(lambda x: x**2).float()

    # 1) Noise from slicing-independent noise (input value errors and weight quantization errors)
    #  - Noise from input variance

    # If input_bits, add quantization noise automatically
    input_val_noise_est = calc_input_value_noise(
        input_pdf,
        weight,
        input_bits, input_scale,
        input_noise,
        padding,
        stride,
        #n_sample
    )

    #  - Noise from weight variance (None default since we use exact weights)
    if weight_value_noise:
        weight_var = weight_value_noise(weight)
        weight_val_noise_est = F.conv2d(input_square_ev, weight_var, bias=None, stride=stride)

    # - Noise from weight quantization, which is predictable (classified as weight value error)
    if weight_bits:
        assert weight_scale, f"weight scale should exist with declared weight bits = {weight_bits}"

        # 1) Noise assuming no input spatial correlation
        input_ev = input_pdf.get_expected_value(lambda x: x).float()
        weight_q = torch.round(weight/weight_scale)*weight_scale
        weight_err = weight_q - weight
        
        # (W^2)*(<x^2>-<x>^2)
        diagonal_var = F.conv2d(input_square_ev-(input_ev**2), weight_err**2, bias=None, stride=stride)
        # (W*<x>)^2
        off_diagonal_var = (F.conv2d(input_ev, weight_err, bias=None, stride=stride))**2

        weight_val_uncorrelated = diagonal_var + off_diagonal_var

        # 2) Noise assuming full spatial correlation
        if use_spatial_correlation:
            input_spatial_correlation = get_spatial_correlation(
                k=(R, S), in_activation_size=(H, W)
            )
            input_spatial_correlation=0
            #    formula = sum(W_error * sqrt(<x^2>))^2
            weight_val_correlated = F.conv2d(torch.sqrt(input_square_ev), weight_err, bias=None, stride=stride)**2
            
            weight_val_noise_est += weight_val_uncorrelated * (1-input_spatial_correlation) + weight_val_correlated * input_spatial_correlation
        else:
            weight_val_noise_est += weight_val_uncorrelated

    if weight_bits:
        # noise from physical ReRAM Variation
        weight_physical_noise_est, sliced_weights, weight_slice_scales = calc_weight_physical_noise(
            weight, weight_scale, weight_zp, weight_bits,
            weight_slice_lsb_to_msb,
            weight_physical_noise,
            input_square_ev,
            differential_weight=differential_weight,
            stride=stride
        )
        
    if input_bits:
        q_input_pdf = input_pdf.copy()
        q_input_pdf.range = quantize_tensor(q_input_pdf.range, input_scale, input_zp)
        sliced_input, input_slice_scales = [], []

        # If negative input--two separate cycles of positive inputs
        if (q_input_pdf.range < 0).any():
            for abs_pdf in q_input_pdf.get_pos_neg_pdf():
                sliced_input.append(abs_pdf)
                input_slice_scales.append(1)
                #print(abs_pdf.pdf, abs_pdf.range)
        else:
            sliced_input, input_slice_scales = [q_input_pdf], [1]
        #sliced_input, input_slice_scales = slice_inputs(input_pdf, input_scale, input_zp, input_bits, input_slice_lsb_to_msb)
        
    adc_noise_est = torch.zeros(output_var.shape, device=device)

    # 2) Noise from ADC thermal
    if adc_noise and input_bits and weight_bits:
        for input_slice, input_slice_scale in zip(sliced_input, input_slice_scales):
            for weight_slice, weight_slice_scale in zip(sliced_weights, weight_slice_scales):

                # Differential: ADC output subtracted
                if differential_weight:
                    weight_slice = weight_slice[0] - weight_slice[1]
                
                # change dtype to float
                weight_slice = weight_slice.to(dtype=torch.float)

                assert weight_slice.shape == weight.shape, f"sliced filter shape {weight_slice.shape} =/= filter shape {weight.shape}"
            
                # Combine scaling factors
                slice_scale = input_slice_scale * weight_slice_scale
                tile_size = Cin * R * S

                # Strided Patch: Row-Major Order but the code is  order-insensitive
                for q, w in enumerate(range(0, W, stride)):
                    for p, h in enumerate(range(0, H, stride)):

                        input_tile = input_slice[:, h:h+R, w:w+S].flatten()
                        weight_tile = weight_slice.flatten(start_dim = 1)
                        
                        assert input_tile.pdf.shape[1] == tile_size, f"tile size {input_tile.pdf.shape[1]} =/= {Cin} x {R} x {S} (Cin x R x S)"
                        assert weight_tile.shape[1] == tile_size, f"tile size {weight_tile.shape} =/= {Cout}, {Cin} x {R} x {S} (Cout, Cin x R x S)"
                        
                        # VMM by num_rows (maximum number of rows to accumulate by ADC)
                        offset, vmm_total_noise = 0, torch.zeros(Cout, device=device)

                        while offset < tile_size:
                            size = max(num_rows if num_rows else 0, tile_size-offset)
                            input_vector = input_tile[offset:offset+size]
                            input_vector_avg = input_vector.get_expected_value(lambda x: x, trapezoidal=False).float()
                            weight_matrix = weight_tile[:, offset:offset+size]

                            vmm_result = torch.einsum('i,ji->j', input_vector_avg, weight_matrix)
                            vmm_total_noise += adc_noise(vmm_result)

                            offset += size

                        adc_noise_est[:,p,q] += vmm_total_noise * ((slice_scale * input_scale * weight_scale)**2)

    noise_stat = {
        'input_val_noise_est': input_val_noise_est,
        'weight_val_noise_est': weight_val_noise_est,
        'weight_physical_noise_est': weight_physical_noise_est,
        'input_physical_noise_est': input_physical_noise_est,
        'adc_noise_est': adc_noise_est
    }
    noise_total = sum(noise_stat.values()) + torch.zeros(output_var.shape, device=device)

    return noise_total, noise_stat


def get_fused_convbn_params(module_dict, parent_layer_name='', conv_name='conv', bn_name='bn', epsilon = 1e-15):
    """
    Returns: dictionary of keys ('weights', 'bias', 'stride', 'padding') mapped to corresponding fused conv-bn parameters
    """

    # convolution weight
    if parent_layer_name != '':
        conv_name = '.'.join([parent_layer_name, conv_name])
        bn_name   = '.'.join([parent_layer_name, bn_name])
    
    # get conv params
    conv_layer = module_dict[conv_name]
    conv_weight = conv_layer.weight
    # get batch norm params
    bn_layer = module_dict[bn_name]
    bn_weight = bn_layer.weight
    bn_bias   = bn_layer.bias
    bn_mean   = bn_layer.running_mean
    bn_var    = bn_layer.running_var
    
    # fuse params
    bias = -bn_weight * bn_mean / torch.sqrt(bn_var + epsilon) + bn_bias
    weight = conv_weight * (bn_weight / (bn_var + epsilon)).view(-1, 1, 1, 1)

    stride= conv_layer.stride[0] #TODO: Cautious if stride isn't W, H symmetric in future cases
    padding= conv_layer.padding[0]

    output = {
        'weight': weight,
        'bias': bias, # we don't need bias for error calculation
        'stride': stride,
        'padding': padding
    }
    
    return output

def get_avgpool_params(num_channels, kernel, stride=1, padding=0):

    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    
    # get filter size--avgpool weights are 1/(filter size)
    filter_size = 1
    for k in kernel:
        filter_size *= k
    weight = torch.zeros((num_channels, num_channels, *kernel), device=device)

    for c in range(num_channels):
        weight[c, c, :, :] = 1/filter_size
        
    return {
        'weight': weight,
        'stride': stride,
        'padding': padding,
        'bias': None
    }

def get_location_independent_input(num_batch: int, pdf: PDF, device=device)->torch.Tensor:
    # Get theoretically localtion-independent input tensor from PDF probability.
    # return_shape = (num_batch, *input_shape)
    bin_range, pdf_tensor = pdf.range, pdf.pdf
    num_bins, *shape = pdf_tensor.shape
    random_sel = torch.rand(num_batch, *shape, device=device)

    # cumulative distribution function -- if random_sel falls within the bin, the bin range is chosen
    cdf = torch.zeros(random_sel.shape, device=device)
    output = torch.zeros(random_sel.shape, device=device)

    def get_masked_bin_output(cdf, current_pdf, range_min, range_max, output, epsilon=1e-15):
        # for current bin, get cdf (at range_min) and cdf_next (at range_max)
        cdf_next = cdf + current_pdf
        # within the range
        mask = (random_sel > cdf) * (random_sel <= cdf_next)
        # linear interpolation of cdf to find the value
        linear_interpolated = (random_sel - cdf)/(current_pdf+epsilon)*(range_max-range_min) + range_min
        masked_val = torch.where(mask, linear_interpolated, torch.zeros_like(output))

        output += masked_val
        return cdf_next, output

    if pdf.p_zero is not None:
        p_zero = pdf.p_zero.unsqueeze(0)
        cdf, output = get_masked_bin_output(cdf, p_zero, 0, 0, output)
    
    for idx in range(num_bins):
        range_min, range_max = sorted(bin_range[idx:idx+2])
        current_pdf = pdf_tensor[idx]
        cdf, output = get_masked_bin_output(cdf, current_pdf, range_min, range_max, output)
    
    return output
    