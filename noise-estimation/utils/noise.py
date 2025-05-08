import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

# if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_bin = 50

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

class PDF:
    def __init__(self, pdf_range, pdf, p_zero=None, device=device, dtype=torch.float):
        """
        Initialize the PDF object.
        Parameters:
        - range (list): The range of the PDF (length: n_bin + 1).
        - pdf (list): The PDF values. (shape: (n_bin, *input_activation.shape))
        - device (torch.device): The device to use for the tensors.
        """
        self.range = np_to_torch(pdf_range, dtype=dtype, device=device)
        self.pdf = np_to_torch(pdf, dtype=dtype, device=device)
        self.n_bin = pdf.shape[0]
        self.p_zero = None

        if p_zero is not None:
            self.p_zero = np_to_torch(p_zero, dtype=dtype, device=device) # probability of zero value (if counted or None)
            
        self.device = device

        assert self.range.shape[0] == self.n_bin+1, "Length of range should be len(pdf)+1"

    def get_expected_value(self, func=lambda x: x, trapezoidal=True, apply_relu_suppress=False, ignore_relu=False):
        """
        Calculate the expected value of a function with the PDF.
        Parameters:
        - func (function): The function to apply to the PDF.
        - trapezoidal: Use trapezoidal approx
        - apply_relu_suppress: If true, suppress the output noise by non-positive probability (since noise at value <= 0 == 0)
        - ignore_relu: If wanted, ignore applying ReluNoise at all
        Returns:
        - torch.Tensor: The expected value. (shape: (*input_activation.shape))
        """
        # Iterative Call to get expected value more efficiently
        if isinstance(func, AddNoise):
            return (
                self.get_expected_value(func.noise_1, trapezoidal, apply_relu_suppress, ignore_relu) +  
                self.get_expected_value(func.noise_2, trapezoidal, apply_relu_suppress, ignore_relu)
            )
        elif isinstance(func, ReluNoise):
            if ignore_relu:
                return self.get_expected_value(func.noise, trapezoidal, apply_relu_suppress, ignore_relu)
            
            func_value = self.get_expected_value(func.noise, trapezoidal, apply_relu_suppress=True)
            #non_pos_prob = self.get_non_pos_prob()
            
            return func_value
        
        elif isinstance(func, GaussianNoise):
            return func(self.pdf[0])
        
        if trapezoidal:
            # Calculate the expected value using the trapezoidal rule
            func_value = func(self.range)/2
            func_value = func_value[1:] + func_value[:-1]
        else:
            # Calculate the expected value using the midpoint rule
            func_value = func(self.range[:-1])

        if relu:
            func_value *= (self.range[:-1] >= 0)

        while func_value.dim() < self.pdf.dim():
            func_value = func_value.unsqueeze(-1)
        func_value = func_value * self.pdf
        func_value = func_value.sum(dim=0, keepdim=False)
        
        if (self.p_zero is not None) and (not relu):
            func_value += self.p_zero * func(torch.zeros(func_value.shape, device=self.device))

        return func_value
    
    def __getitem__(self, idx_range):
        if not isinstance(idx_range, (tuple, list)):
            idx_range = (idx_range,)

        sliced_p_zero = self.p_zero[idx_range] if self.p_zero is not None else None

        return PDF(self.range,
                    self.pdf[(slice(None), *idx_range)], 
                    p_zero=sliced_p_zero, 
                    device=self.device
                )

    def add_padding(self, padding=0, mode='constant', value=0):
        # add padding to pdf and p_zero

        if isinstance(padding, int):
            padding = (padding,) * 4 # all four sides of the tensor
        
        pdf_padded= F.pad(self.pdf, padding, mode=mode, value=0)
        p_zero_padded = self.p_zero

        # add 1 to which the padding is 0
        if value == 0:
            if self.p_zero is None:
                zeros = torch.zeros(self.pdf.shape[1:])
                p_zero_padded = F.pad(zeros, padding, mode=mode, value=1)
            else:
                p_zero_padded = F.pad(self.p_zero, padding, mode=mode, value=1)
        else:
            found_value_bin = False
            for i in range(self.n_bin):
                # Create a mask for values in the i-th bin range: [range[i], range[i+1])
                min_val, max_val = sorted(self.range[i:i+2])
                if (value >= min_val) and (value < max_val):
                    zeros = torch.zeros(self.pdf.shape[1:])

                    one_padded_tensor_zeroes = F.pad(
                        zeros, padding, 
                        mode=mode, value=1
                    ) #NOTE: padding 1 (probability of padding = 1)

                    pdf_padded[i] += one_padded_tensor_zeroes
                    found_value_bin = True
                    break
            if not found_value_bin:
                raise LookupError(f"padding value {value} not in a pdf range of bins--failed to pad")
            
            if self.p_zero is not None:
                p_zero_padded = F.pad(self.p_zero, padding, mode=mode, value=0.0)
                        
        return PDF(
            self.range.clone(),
            pdf_padded,
            p_zero_padded,
            self.device
        )

    def flatten(self):
        # flatten the pdf tensor
        p_zero_flatten = self.p_zero.flatten() if self.p_zero is not None else None
        p_zero_flatten = self.p_zero.flatten() if self.p_zero is not None else None

        return PDF(
            self.range,
            self.pdf.flatten(start_dim=1),
            p_zero_flatten,
            self.device
        )

    def unsqueeze(self, dim=0):
        # unsqueeze the pdf tensor
        p_zero = self.p_zero.unsqueeze(dim) if self.p_zero is not None else None
        adjusted_dim = dim + 1 if dim >= 0 else dim
        pdf = self.pdf.unsqueeze(adjusted_dim)

        return PDF(self.range, pdf, p_zero, self.device)
    
    def copy(self):
        p_zero_clone = self.p_zero.clone() if self.p_zero is not None else None
        return PDF(self.range.clone(), self.pdf.clone(), p_zero_clone, self.device)
    
    def normalize(self):
        # normalize PDF probability s.t. sum(pdf) of each element == 1
        sum_pdf = self.pdf.sum(dim=0, keepdim=False)
        if self.p_zero is not None:
            sum_pdf += self.p_zero

        self.pdf = self.pdf / sum_pdf
        if self.p_zero is not None:
            self.p_zero = self.p_zero / sum_pdf

    def get_non_pos_prob(self):
        # get probability of each input location being <= 0
        prob = torch.zeros(self.pdf.shape[1:])
        
        if self.p_zero is not None:
            prob += self.p_zero
            
        for i in range(self.n_bin):
            low, high = self.range[i:i+2]
            
            if high <= 0:
                prob += self.pdf[i]
    
            elif low < 0 and high > 0:
                prob += self.pdf[i] * (0-low)/(high-low+1e-9)
            
            else:
                break
        return prob
                
    
    def get_pos_neg_pdf(self):
        # slice into positive and negative range PDFs and conver to Absolute Value
        # NOTE: the resulting PDFs are `normalized`` (sum of each pdf = 1)
        #       to treat it as completely new PDFs
        # Assumptions:
        #       Range is assumed to be sorted in increasing order
        #       Range is assumed to include both negative AND positive value
        zero_idx = torch.where((self.range[:-1] <= 0) & (self.range[1:] > 0))[0][0].item()
        neg_pdf, pos_pdf = self.copy(), self.copy()

        # get splitting boundaries
        low  = self.range[zero_idx].item()
        high = self.range[zero_idx+1].item()

        # edge case:
        if low == 0:
            neg_pdf.pdf, pos_pdf.pdf = neg_pdf.pdf[:zero_idx], pos_pdf.pdf[zero_idx:]
            neg_pdf.range = neg_pdf.range[:zero_idx+1]
            pos_pdf.range = pos_pdf.range[zero_idx:]

        # typical case: range[zero_idx] < 0 < range[zero_idx+1]
        if low < 0:
            neg_pdf.pdf, pos_pdf.pdf = neg_pdf.pdf[:zero_idx+1], pos_pdf.pdf[zero_idx:]
            neg_pdf.range = neg_pdf.range[:zero_idx+2]
            pos_pdf.range = pos_pdf.range[zero_idx:]
            
            neg_pdf.range[-1] = 0
            pos_pdf.range[0]  = 0

            # split probability assuming uniform distribution within a single bin
            neg_pdf.pdf[-1] *= - float(low) / (high - low + 1e-8)
            pos_pdf.pdf[0]  *= float(high) / (high - low + 1e-8)
        # neg_pdf = clipping to max 0, pos_pdf = clipping to min 0 --> accumulate opposite sign's PDFs

        neg_prob, pos_prob = neg_pdf.pdf.sum(dim=0), pos_pdf.pdf.sum(dim=0)

        # clipped to 0 accurately
        if self.p_zero is not None:
            neg_pdf.p_zero += pos_prob
            pos_pdf.p_zero += neg_prob
        else:
            neg_pdf.p_zero = pos_prob
            pos_pdf.p_zero = neg_prob

        # flip negative pdf to get absolute pdf value
        neg_pdf.pdf = torch.flip(pos_pdf.pdf, dims=[0])
        neg_pdf.range = -torch.flip(pos_pdf.range, dims=[0])
        
        return (pos_pdf, neg_pdf)


class Noise:
    def __init__(self, func=None):
        self.func = func
        self.padding = 0

    def __call__(self, x):
        """
        Call the noise function.
        Parameters:
        - x (torch.Tensor): The input tensor.
        Returns:
        - torch.Tensor: The noisy tensor.
        """
        return self.func(x)

    def __add__(self, other):
        """
        Define addition for noise objects.
        """
        raise NotImplementedError

    def __mul__(self, scalar):
        """
        Define scalar multiplication for noise objects.
        """
        raise NotImplementedError

    def __rmul__(self, scalar):
        """
        Define reverse scalar multiplication for noise objects.
        """
        return self.__mul__(scalar)
    
    def __getitem__(self, idx_range):
        """
        Define indexing for noise objects.
        """
        return self
    
    def add_padding(self, padding:int = 0):
        # add padding variable (for Gaussian Noise)
        self.padding = padding

        # search attributes and add_padding
        for name, attr in vars(self).items():
            if isinstance(attr, Noise):
                attr.add_padding(padding)

    def copy(self):
        # Deepcopy Noise with all tensors and Noises anti-aliased
        # 1. Create a new instance of the same class
        copied = self.__class__.__new__(self.__class__)
        
        # 2. Copy attributes
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(copied, attr, value.clone())
            elif isinstance(value, Noise):
                setattr(copied, attr, value.copy())
            else:
                setattr(copied, attr, value)
        
        return copied

        
class AddNoise(Noise):
    def __init__(self, noise_1=None, noise_2=None):
        """
        Sum of two noises
        """
        super().__init__()
        self.noise_1 = noise_1
        self.noise_2 = noise_2
        
    def __call__(self, x):
        return self.noise_1(x) + self.noise_2(x)

    def __mul__(self, scalar):
        return AddNoise(self.noise_1 * scalar, self.noise_2 * scalar)
    
    def __getitem__(self, idx_range):
        """
        Define indexing for noise objects.
        """
        return AddNoise(self.noise_1[idx_range], self.noise_2[idx_range])
    
def relu(var, x):
    """
    Input-dependent variance after applying ReLU.
    """
    std = torch.sqrt(var)
    var_half = var/2
    e = 1e-15 # avoid division by zero
    while x.dim() < std.dim():
        x = x.unsqueeze(-1)

    scaled_var = torch.tanh((x-0.2)/(std+e))* var_half + var_half

    #TODO: supress ReLU noise at zero
    scaled_var = torch.where(x <= 0, torch.zeros_like(scaled_var), scaled_var)

    return scaled_var

class ReluNoise(Noise):
    def __init__(self, noise=None):
        """
        Initialize ReLU noise, added on top of sub-noise
        """
        super().__init__()
        self.noise = noise
    
    def __call__(self, x):
        var = self.noise(x)
        return self.relu(var, x)
    
    def __mul__(self, scalar):
        return ReluNoise(self.noise * scalar)
    
    def __getitem__(self, idx_range):
        """
        Define indexing for noise objects.
        """
        return ReluNoise(self.noise[idx_range])


class GaussianNoise(Noise):
    def __init__(self, var=0):
        """
        Initialize Gaussian noise.
        Parameters:
        - var (float or tensor): Variance of the Gaussian noise.
        """
        super().__init__()
        self.var = var

    def __call__(self, x):
        if isinstance(self.var, torch.Tensor):
            var = self.var
            if self.padding:
                var = pad_to_size(var, self.padding, value = 0)

            # location-dependent variance
            if var.shape == x.shape[-self.var.dim():]: 
                # Variance shape must match input shape
                out_shape = x.shape

            # If input is range and variance determines the tensor shape
            elif x.dim() == 1:
                out_shape = (*x.shape, *var.shape)

            variance = var * torch.ones(out_shape, device=x.device) # broadcast variance
            return variance
        
        else:
            # uniform variance
            return self.var * torch.ones(x.shape, device=x.device)

    def __getitem__(self, idx_range):
        """
        Define indexing for noise objects.
        """
        if not isinstance(idx_range, (tuple, list)):
            idx_range = (idx_range,)

        if isinstance(self.var, torch.Tensor):
            return GaussianNoise(self.var[idx_range])
        else:
            return GaussianNoise(self.var)
        
    def __repr__(self):
        if isinstance(self.var, torch.Tensor):
            return f"var={torch.mean(self.var)}, shape={self.var.shape}"
        return str(self.var)
    
    def __mul__(self, scalar):
        return GaussianNoise(self.var * (scalar**2))

class ProportionalNoise(Noise):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x):
        return (self.alpha * x)**2
    
    def __repr__(self):
        return f'alpha={self.alpha}'
    
    def __mul__(self, scalar):
        return ProportionalNoise(self.alpha * (scalar**2))
    
if __name__ == "__main__":
    x_range = torch.Tensor([np.random.uniform(-1, 1, (100,)) for i in range(4)])
    weight =  torch.Tensor( np.random.uniform(-1, 1, tuple((100,20))) )

    noise = GaussianNoise(torch.Tensor(np.random.uniform(0.5, 1, tuple((100,20)))))

    plt.scatter(torch_to_np(weight), torch_to_np(noise(weight)))
    plt.show()
    #print(x_range[0], weight[0], x_range[0] @ weight)
    pdf = (1/3,1/3,1/3)
    pdf = PDF(x_range, pdf)

    g_1= GaussianNoise(0.1)
