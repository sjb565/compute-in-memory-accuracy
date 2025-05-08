"""
Parameterizable inference simulation script for CIFAR-10 ResNets.
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import warnings, sys, time
#from build_resnet_cifar10 import ResNet_cifar10
warnings.filterwarnings('ignore')
sys.path.append("../../") # to import dnn_inference_params
sys.path.append("../../../../") # to import simulator
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, reinitialize
from find_adc_range import find_adc_range
from dnn_inference_params import dnn_inference_params

import json
import os
import torchvision
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import itertools

useGPU = True #True # use GPU?
N = 500 # number of images
batch_size = 4 
Nruns = 1
print_progress = True



class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load mapping from synset to integer class
        with open("../resnet18/imagenet_class_index.json") as f:
            idx_to_label = json.load(f)

        # Create reverse mapping: synset -> class index
        self.synset_to_idx = {v[0]: int(k) for k, v in idx_to_label.items()}

        # Iterate over each class folder
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = self.synset_to_idx.get(class_name, None)
            if label is None:
                continue

            # Iterate over each image file in the class folder
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(class_path, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),  # Resize the shorter side to 256 pixels
    transforms.CenterCrop(224),                                        # Crop the center 224x224 pixels
    transforms.ToTensor(),                                             # Convert the image to a PyTorch tensor
    transforms.Normalize(                                              # Normalize using ImageNet's mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])
val_dir = '/n/home05/kekim/Downloads/val'
dataset = CustomImageNetDataset(val_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# parameter combinationrs
params_sweep_list_raw = {
    'input_bits': [8,10,12], #[i for i in range(1, 20)],
    'weight_bits': [8,10,12], #[32], #[i for i in range(1, 20)],

    #'input_noise': None,
    #'weight_value_noise': None,
    'weight_physical_noise': [0,0.01, .04, .08, .16], #+ [ProportionalNoise(i) for i in (0.02, 0.04, 0.08, 0.16,0.32, 0.64)], #[None, ProportionalNoise(0.01), ProportionalNoise(0.1)], (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.09, 0.1, 0.11, 0.12) # (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.12, 0.16, 0.32, 0.64)
    'differential_weight': ["BALANCED", "OFFSET"],
    'weight_slice_lsb_to_msb': [1,2,4], #[1, 2, 4, 8],
    'adc_noise': [None], #[GaussianNoise(i**2) for i in (1, 5, 10, 15, 20)] #[None, ProportionalNoise(0.1)],
}

print("Model: ResNet-18")
print("CIFAR-10: using "+("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(N))
print("Number of runs: {:d}".format(Nruns))
print("Batch size: {:d}".format(batch_size))
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

##### Load Pytorch model
#resnet_model = ResNet_cifar10(n)
resnet_model = models.resnet18(pretrained=True)
resnet_model = resnet_model.to(device)
#resnet_model.load_state_dict(
#    torch.load('./models/resnet18.pth',
#    map_location=torch.device(device)))
resnet_model.eval()
n_layers = len(convertible_modules(resnet_model))

# 1. Make all combinations
keys = list(params_sweep_list_raw.keys())
values_list = [params_sweep_list_raw[key] for key in keys]
combinations = list(itertools.product(*values_list))

# Create a list of CrossSimParameters objects
cross_sim_params_list = [None] * n_layers

params_list = [dict(zip(keys, values)) for values in combinations]

### Load input limits
input_ranges = np.load("./calibrated_config/input_limits_ResNet18.npy")

# Define a hook function
# Storage for activations
first_layer_outputs = []
last_layer_outputs = []

def hook_fn(module, input, output):
    batch_outputs = output.detach().cpu()
    for i in range(batch_outputs.size(0)):
        if len(first_layer_outputs) < N:
            first_layer_outputs.append(batch_outputs[i])

def last_layer_hook_fn(module, input, output):
    batch_outputs = output.detach().cpu()
    for i in range(batch_outputs.size(0)):
        if len(last_layer_outputs) < N:
            last_layer_outputs.append(batch_outputs[i])


def run_inf(inputbits, weightbits, weight_noise, corestyle, nslices, adc_noise):
    # Storage for activations
    first_layer_outputs.clear() 
    last_layer_outputs.clear() 
    # Params arguments common to all layers
    base_params_args = {
        'ideal' : False,
        ## Mapping style
        'core_style' : corestyle,
        'Nslices' : nslices if corestyle!="OFFSET" else 1,
        ## Weight value representation and precision
        'weight_bits' : int(weightbits),
        'weight_percentile' : 100,
        'digital_bias' : True,
        'digital_offset': corestyle=="OFFSET",
        ## Memory device
        'Rmin' : 1e4,
        'Rmax' : 1e6,
        'infinite_on_off_ratio' : True,
        'error_model' : "none",
        'alpha_error' : 0.0,
        'proportional_error' : False,
        'noise_model' : "none",
        'alpha_noise' : 0.0,
        'proportional_noise' : False,
        'drift_model' : "none",
        't_drift' : 0,
        ## Array properties
        'NrowsMax' : 1152,
        'NcolsMax' : None,
        'Rp_row' : 0, # ohms
        'Rp_col' : 0, # ohms
        'interleaved_posneg' : False,
        'subtract_current_in_xbar' : True,
        'current_from_input' : True,
        ## Input quantization
        'input_bits' : 8,
        'input_bitslicing' : False,
        'input_slice_size' : 8,
        ## ADC
        'adc_bits' : 0,
        'adc_range_option' : "CALIBRATED",
        'adc_type' : "generic",
        ## Simulation parameters
        'useGPU' : useGPU,
        ## Profiling
        'profile_xbar_inputs' : False,
        'profile_adc_inputs' : False,
        'ntest' : N,
        }
    
    ### Load ADC limits
    adc_limit_name=f"adc_limits_ResNet18_corestyle_{corestyle}_nslices_{nslices}_weightbits_{weightbits}.npy"
    adc_ranges = find_adc_range(base_params_args, n_layers, True, adc_limit_name)

    ### Set the parameters
    for k in range(n_layers):
        params_args_k = base_params_args.copy()
        params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
        params_args_k['input_range'] = input_ranges[k]
        params_args_k['adc_range'] = adc_ranges[k]
        cross_sim_params_list[k] = dnn_inference_params(**params_args_k)

    #### Convert PyTorch layers to analog layers
    analog_resnet = from_torch(resnet_model, cross_sim_params_list, fuse_batchnorm=True, bias_rows=0)


    # Attach the hook to the first layer
    first_layer = list(analog_resnet.children())[0]
    last_layer = list(analog_resnet.children())[-1]

    for idx, layer in enumerate(analog_resnet.children()):
        print(f"Layer {idx}: {layer}")
    first_layer.register_forward_hook(hook_fn)
    last_layer.register_forward_hook(last_layer_hook_fn)


    #### Load and transform CIFAR-10 dataset
    #normalize = transforms.Normalize(
    #    mean = [0.485, 0.456, 0.406],
    #    std  = [0.229, 0.224, 0.225])
    #dataset = datasets.CIFAR10(root='./',train=False, download=True, 
    #    transform= transforms.Compose([transforms.ToTensor(), normalize]))
    #dataset = torch.utils.data.Subset(dataset, np.arange(N))
    #cifar10_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    #### Initialize dataset and dataloader



    #### Run inference and evaluate accuracy
    accuracies = np.zeros(Nruns)
    for m in range(Nruns):

        T1 = time.time()
        y_pred, y, k = np.zeros(N), np.zeros(N), 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= N // batch_size:
                break
            inputs = inputs.to(device)
            output = analog_resnet(inputs)
            output = output.to(device)
            y_pred_k = output.data.cpu().detach().numpy()
            if batch_size == 1:
                y_pred[k] = y_pred_k.argmax()
                y[k] = labels.cpu().detach().numpy()
                k += 1
            else:
                batch_size_k = y_pred_k.shape[0]
                y_pred[k:(k+batch_size_k)] = y_pred_k.argmax(axis=1)
                y[k:(k+batch_size_k)] = labels.cpu().detach().numpy()
                k += batch_size_k
            if print_progress:
                print("Image {:d}/{:d}, accuracy so far = {:.2f}%".format(
                    k, N, 100*np.sum(y[:k] == y_pred[:k])/k), end="\r")
        
        T2 = time.time()
        top1 = np.sum(y == y_pred)/len(y)
        accuracies[m] = top1
        print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
        print('Accuracy: {:.2f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))
        if m < (Nruns - 1):
            reinitialize(analog_resnet)

    if Nruns > 1:
        print("==========")
        print("Mean accuracy:  {:.2f}%".format(100*np.mean(accuracies)))
        print("Stdev accuracy: {:.2f}%".format(100*np.std(accuracies)))


    print("Captured", len(first_layer_outputs), "batches of first layer outputs.")
    print("Shape of first batch output:", first_layer_outputs[0].shape)

    print("Captured", len(last_layer_outputs), "batches of last layer outputs.")
    print("Shape of last batch output:", last_layer_outputs[0].shape)

    # Convert lists of tensors to single numpy arrays
    first_layer_outputs_np = torch.stack(first_layer_outputs).numpy()
    last_layer_outputs_np = torch.stack(last_layer_outputs).numpy()

    dirname = f"activations_ibits_{inputbits}_wbits_{weightbits}_corestyle_{corestyle}_weightnoise_{weight_noise}_weightslice_{nslices}"
    # Create output directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)

    # Save to .npy files
    np.save(f"{dirname}/last_layer_outputs.npy", last_layer_outputs_np)
    np.save(f"{dirname}/first_layer_outputs.npy", first_layer_outputs_np)
    np.save(f"{dirname}/accuracy.npy", top1)

    print(f"Saved last layer output  activations to '{dirname}/last_layer_outputs.npy' with shape {last_layer_outputs_np.shape}")
    print(f"Saved first layer output activations to '{dirname}/first_layer_outputs.npy' with shape {first_layer_outputs_np.shape}")





# iterate through combinations
for i, params in enumerate(params_list):
    print(f"Combination {i+1}:")
    if params['differential_weight'] == "OFFSET" and params['weight_slice_lsb_to_msb'] > 1:
            continue
    run_inf(params['input_bits'], params['weight_bits'], params['weight_physical_noise'], params['differential_weight'], params['weight_slice_lsb_to_msb'], params['adc_noise'])




