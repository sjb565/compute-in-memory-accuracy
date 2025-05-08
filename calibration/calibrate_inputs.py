"""
Script to obtain calibrated crossbar input ranges for CIFAR-10 ResNets.
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
from simulator.algorithms.dnn.torch.profile import get_profiled_xbar_inputs
from dnn_inference_params import dnn_inference_params
from calibration import calibrate_input_limits

import json
import os
import torchvision
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader

useGPU = False # use GPU?
N = 200 # number of images from the TRAINING set
batch_size = 8
Nruns = 1
print_progress = True

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

##### Set the simulation parameters

# Create a list of CrossSimParameters objects
params_list = [None] * n_layers

# Params arguments common to all layers
base_params_args = {
    'ideal' : False,
    ## Mapping style
    'core_style' : "BALANCED",
    'Nslices' : 1,
    ## Weight value representation and precision
    'weight_bits' : 8,
    'weight_percentile' : 100,
    'digital_bias' : True,
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
    'input_bits' : 0,
    'input_bitslicing' : False,
    'input_slice_size' : 1,
    ## ADC
    'adc_bits' : 0,
    'adc_range_option' : "CALIBRATED",
    'adc_type' : "generic",
    'adc_per_ibit' : False,
    ## Simulation parameters
    'useGPU' : useGPU,
    ## Profiling
    'profile_xbar_inputs' : True,
    'profile_adc_inputs' : False,
    'ntest' : N,
    }

### Set the parameters
for k in range(n_layers):
    params_args_k = base_params_args.copy()
    params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
    params_list[k] = dnn_inference_params(**params_args_k)

#### Convert PyTorch layers to analog layers
analog_resnet = from_torch(resnet_model, params_list, fuse_batchnorm=True, bias_rows=0)

#### Load and transform CIFAR-10 dataset
#normalize = transforms.Normalize(
#    mean = [0.485, 0.456, 0.406],
#    std  = [0.229, 0.224, 0.225])
#dataset = datasets.CIFAR10(root='./',train=True, download=True, 
#    transform= transforms.Compose([transforms.ToTensor(), normalize]))
#dataset = torch.utils.data.Subset(dataset, np.arange(N))
#cifar10_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

#### Initialize dataset and dataloader
class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load mapping from synset to integer class
        with open("../quantized_resnet18/imagenet_class_index.json") as f:
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
val_dir = '../../val'
dataset = CustomImageNetDataset(val_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#### Run inference and evaluate accuracy
T1 = time.time()
y_pred, y, k = np.zeros(N), np.zeros(N), 0
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= N // batch_size:
            break
        #torch.cuda.empty_cache()
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
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
print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
print('Accuracy: {:.2f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))

#### Retrieve profiled inputs and calibrate limits
print("Collecting profiled data")
profiled_inputs = get_profiled_xbar_inputs(analog_resnet)
print("Optimizing input limits")
calibrated_ranges = calibrate_input_limits(profiled_inputs, Nbits=8)

# Manually calibrate first layer's limits based on value range of CIFAR-10 images
calibrated_ranges[0,:] = np.array([-2.64, 2.64])

np.save("./calibrated_config/input_limits_ResNet18.npy",
    calibrated_ranges)