from utils.noise import Noise, PDF
import torch
import numpy as np

#TODO: Change for different network
A_fit, tau_fit, beta_fit = [1.16378446, 10.05172115, 0.60425503]

# Define the stretched exponential
def stretched_exp(x, A=A_fit, tau=tau_fit, beta=beta_fit):
    return A * np.exp(-(x / tau) ** beta)


def get_spatial_correlation(
        k: list[int] | tuple[int], 
        in_activation_size: list[int] | tuple[int], 
        correlation_fn=stretched_exp, 
        image_size: list[int] | tuple[int] = (224, 224)):
    """
    k: tuple or list of odd integer, kernel size
    in_activation_size: width/height size of input activation
        (r= float, ratio of input_size / current_size)
    correlation_fn: function that takes distance and returns correlation
    Returns: average correlation over selected integer grid points
    """
    assert k[0] % 2 == 1 and k[1] % 2 == 1, "k must be odd"
    # Map current activation layer to original image
    r_w = image_size[0] / in_activation_size[0]
    r_h = image_size[1] / in_activation_size[1]

    # Compute bounds
    min_w, max_w = -(k[0]//2) - 0.5,  (k[0]//2) + 0.5
    min_h, max_h = -(k[1]//2) - 0.5,  (k[1]//2) + 0.5

    # Create grid in range [-half, half] * r, excluding central box
    points = []
    for w in range(int(np.floor(min_w * r_w)) + 1, int(np.ceil(max_w * r_w))):
        for h in range(int(np.floor(min_h * r_h)) + 1, int(np.ceil(max_h * r_h))):

            # Check if (i,j) is outside the central (-0.5, 0.5)*r box
            if not (-0.5 * r_w < w < 0.5 * r_w and -0.5 * r_h < h < 0.5 * r_h):
                dist = np.sqrt(w**2 + h**2)
                points.append(correlation_fn(dist))

    if not points:
        return 0.0  # Or raise error
    return np.mean(points)

def get_lognorm_samples(
        input_pdf: PDF, 
        noise_obj: Noise, 
        n_sample: int= 30, 
        s: float = 2.340967613, 
        noise_to_scale: float = 4.710978985
    ):
    # empirical distribution of noise is log-norm
    # sample multiple-shots to get approximate input noise distribution
    # s (const.) ~ 2.340967613
    # scale ~ noise (var) / 4.710978985
    noise = input_pdf.get_expected_value(noise_obj, ignore_relu=True)
    
    device = noise.device
    dtype = noise.dtype
    
    # 1. Compute log-normal parameters
    scale = noise / noise_to_scale
    mean = torch.log(scale+1e-15)
    std = s

    # 2. Expand mean/std to (n_sample, *noise.shape)
    mean_exp = mean.unsqueeze(0).expand(n_sample, *noise.shape)

    # 3. Sample from log-normal (sample from normal then exp)
    normal_sample = torch.randn_like(mean_exp, dtype=dtype, device=device) * std + mean_exp
    lognorm_sample = torch.exp(normal_sample)

    # assign sign
    random_bits = torch.randint(0, 2, lognorm_sample.shape, device=device, dtype=torch.int8)

    delta = torch.sqrt(lognorm_sample) * (2 * random_bits - 1)  # shape: (n_sample, *noise.shape)
    print(f"{torch.mean(lognorm_sample)} vs {torch.mean(noise)}")
    return delta

def get_gaussian_samples(
        input_pdf: PDF, 
        noise_obj: Noise, 
        n_sample: int= 30
    ):
    noise   = input_pdf.get_expected_value(noise_obj, ignore_relu=True)
    std     = torch.sqrt(noise)
    std_exp = std.unsqueeze(0).expand(n_sample, *noise.shape) # expand

    normal_sample = torch.randn_like(std_exp, dtype=noise.dtype, device=noise.device) * std_exp
    return normal_sample



def relu_suppress_neg_noise(input_pdf:PDF, noise_samples:torch.Tensor):
    p_non_pos = input_pdf.get_non_pos_prob()
    n_sample = noise_samples.shape[0]
    
    # Expand p_non_neg to (n_sample, *shape)
    p = p_non_pos.unsqueeze(0).expand(n_sample, *p_non_pos.shape)

    # Draw Bernoulli mask: 1 = keep, 0 = suppress
    keep_mask = p < torch.rand_like(p)
    print(f"{torch.sum(keep_mask.flatten())/(keep_mask.flatten().shape[0])} preserved")

    # Suppress
    return noise_samples * keep_mask
