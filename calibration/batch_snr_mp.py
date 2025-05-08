import torch
from torchvision import transforms, models
import numpy as np
import warnings, sys, time
import os, json, gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import itertools
import multiprocessing as mp

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules
from find_adc_range import find_adc_range
from dnn_inference_params import dnn_inference_params

warnings.filterwarnings('ignore')
sys.path.append("../../")
sys.path.append("../../../../")

useGPU = True
N = 5000
batch_size = 128
Nruns = 1
print_progress = True

# Dataset
class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open("../resnet18/imagenet_class_index.json") as f:
            idx_to_label = json.load(f)
        self.synset_to_idx = {v[0]: int(k) for k, v in idx_to_label.items()}

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            label = self.synset_to_idx.get(class_name, None)
            if label is None:
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transform and dataloader
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_dir = '/n/home05/kekim/Downloads/val'
dataset = CustomImageNetDataset(val_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Parameter sweep setup
params_sweep_list_raw = {
    'input_bits': [8, 10, 12],
    'weight_bits': [8, 10, 12],
    'weight_physical_noise': [0, 0.01, .04, .08, .16],
    'differential_weight': ["BALANCED", "OFFSET"],
    'weight_slice_lsb_to_msb': [1, 2, 4],
    'adc_noise': [None],
}
keys = list(params_sweep_list_raw.keys())
values_list = [params_sweep_list_raw[key] for key in keys]
params_list = [dict(zip(keys, values)) for values in itertools.product(*values_list)]

# Load model and clean references
print("\nPreparing PyTorch ResNet-18 model...")
device = torch.device("cuda:0" if torch.cuda.is_available() and useGPU else "cpu")
resnet_model = models.resnet18(pretrained=True).to(device)
resnet_model.eval()
n_layers = len(convertible_modules(resnet_model))
input_ranges = np.load("./calibrated_config/input_limits_ResNet18.npy")

clean_first_layer_outputs = []
clean_last_layer_outputs = []

def clean_first_hook(module, input, output):
    for i in range(output.size(0)):
        if len(clean_first_layer_outputs) < N:
            clean_first_layer_outputs.append(output[i].detach().cpu())

def clean_last_hook(module, input, output):
    for i in range(output.size(0)):
        if len(clean_last_layer_outputs) < N:
            clean_last_layer_outputs.append(output[i].detach().cpu())

resnet_model.bn1.register_forward_hook(clean_first_hook)
resnet_model.fc.register_forward_hook(clean_last_hook)

print("Running clean inference...")
with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(dataloader):
        if batch_idx >= N // batch_size:
            break
        inputs = inputs.to(device)
        _ = resnet_model(inputs)
print("Clean inference done.")

ref_first = torch.stack(clean_first_layer_outputs).numpy()
ref_last = torch.stack(clean_last_layer_outputs).numpy()

def run_inf(inputbits, weightbits, weight_noise, corestyle, nslices, adc_noise, ref_first_outputs, ref_last_outputs):
    import torch
    import numpy as np
    import gc
    from simulator.algorithms.dnn.torch.convert import from_torch
    from find_adc_range import find_adc_range
    from dnn_inference_params import dnn_inference_params

    crosssim_first_layer_outputs = []
    crosssim_last_layer_outputs = []

    def crosssim_first_hook(module, input, output):
        for i in range(output.size(0)):
            crosssim_first_layer_outputs.append(output[i].detach().cpu())

    def crosssim_last_hook(module, input, output):
        for i in range(output.size(0)):
            crosssim_last_layer_outputs.append(output[i].detach().cpu())

    base_params_args = {
        'ideal': False,
        'core_style': corestyle,
        'Nslices': nslices if corestyle != "OFFSET" else 1,
        'weight_bits': int(weightbits),
        'weight_percentile': 100,
        'digital_bias': True,
        'digital_offset': corestyle == "OFFSET",
        'Rmin': 1e4,
        'Rmax': 1e6,
        'infinite_on_off_ratio': True,
        'error_model': "none",
        'alpha_error': 0.0,
        'proportional_error': False,
        'noise_model': "none",
        'alpha_noise': 0.0,
        'proportional_noise': False,
        'drift_model': "none",
        't_drift': 0,
        'NrowsMax': 1152,
        'NcolsMax': None,
        'Rp_row': 0,
        'Rp_col': 0,
        'interleaved_posneg': False,
        'subtract_current_in_xbar': True,
        'current_from_input': True,
        'input_bits': int(inputbits),
        'input_bitslicing': False,
        'input_slice_size': 8,
        'adc_bits': 0,
        'adc_range_option': "CALIBRATED",
        'adc_type': "generic",
        'useGPU': useGPU,
        'profile_xbar_inputs': False,
        'profile_adc_inputs': False,
        'ntest': N,
    }

    adc_limit_name = f"adc_limits_ResNet18_corestyle_{corestyle}_nslices_{nslices}_weightbits_{weightbits}.npy"
    adc_ranges = find_adc_range(base_params_args, n_layers, True, adc_limit_name)

    cross_sim_params_list = []
    for k in range(n_layers):
        params_args_k = base_params_args.copy()
        params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
        params_args_k['input_range'] = input_ranges[k]
        params_args_k['adc_range'] = adc_ranges[k]
        cross_sim_params_list.append(dnn_inference_params(**params_args_k))

    analog_resnet = from_torch(resnet_model, cross_sim_params_list, fuse_batchnorm=True, bias_rows=0)
    list(analog_resnet.children())[0].register_forward_hook(crosssim_first_hook)
    list(analog_resnet.children())[-1].register_forward_hook(crosssim_last_hook)

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= N // batch_size:
                break
            inputs = inputs.to(device)
            _ = analog_resnet(inputs)

    epsilon = 1e-12
    snr_fn = lambda s, n: 10 * np.log10(np.sum(s ** 2) / (np.sum((s - n) ** 2) + epsilon))
    snrs_first = [snr_fn(s, n) for s, n in zip(ref_first_outputs, torch.stack(crosssim_first_layer_outputs).numpy())]
    snrs_last  = [snr_fn(s, n) for s, n in zip(ref_last_outputs,  torch.stack(crosssim_last_layer_outputs).numpy())]

    avg_snr_first = np.mean(snrs_first)
    avg_snr_last  = np.mean(snrs_last)

    dirname = f"snr_results_ibits_{inputbits}_wbits_{weightbits}_corestyle_{corestyle}_weightnoise_{weight_noise}_weightslice_{nslices}"
    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/avg_snr.txt", "w") as f:
        f.write(f"first_layer_snr_db: {avg_snr_first:.4f}\n")
        f.write(f"last_layer_snr_db:  {avg_snr_last:.4f}\n")

    print(f"\n✅ Saved SNRs to {dirname}/avg_snr.txt | First: {avg_snr_first:.2f} dB | Last: {avg_snr_last:.2f} dB")
    del analog_resnet
    torch.cuda.empty_cache()
    gc.collect()


def run_wrapper(i, params, ref_first, ref_last):
    try:
        run_inf(params['input_bits'], params['weight_bits'],
                params['weight_physical_noise'],
                params['differential_weight'],
                params['weight_slice_lsb_to_msb'],
                params['adc_noise'],
                ref_first, ref_last)
    except Exception as e:
        print(f"❌ Combination {i+1} failed with error: {e}")

if __name__ == '__main__':
    ctx = mp.get_context("spawn")
    for i, params in enumerate(params_list):
        if params['differential_weight'] == "OFFSET" and params['weight_slice_lsb_to_msb'] > 1:
            continue
        print(f"\nRunning combination {i+1}/{len(params_list)}...")
        p = ctx.Process(target=run_wrapper, args=(i, params, ref_first, ref_last))
        p.start()
        p.join()
        torch.cuda.empty_cache()