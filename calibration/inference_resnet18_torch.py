import torch
import torchvision
from torchvision import transforms, models
import numpy as np
import os, time, json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Configuration
useGPU = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")
N = 5000
batch_size = 256
val_dir = '/n/home05/kekim/Downloads/val'

print("Model: ResNet-18 (Standard PyTorch)")
print("Device:", device)
print("Number of images:", N)

# Load standard pretrained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
resnet_model = models.resnet18(weights=weights).to(device)
resnet_model.eval()

# Prepare dataset
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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageNetDataset(val_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Activation storage
batch_first_layer_outputs = []
final_layer_outputs = []

def hook_fn_bn1_output(module, input, output):
    for i in range(output.size(0)):
        if len(batch_first_layer_outputs) < N:
            batch_first_layer_outputs.append(output[i].detach().cpu())

def hook_fn_fc_output(module, input, output):
    for i in range(output.size(0)):
        if len(final_layer_outputs) < N:
            final_layer_outputs.append(output[i].detach().cpu())

# Register hooks
hook_bn1 = resnet_model.bn1.register_forward_hook(hook_fn_bn1_output)
hook_fc = resnet_model.fc.register_forward_hook(hook_fn_fc_output)

# Inference loop
y_pred, y_true, k = np.zeros(N), np.zeros(N), 0
start_time = time.time()

for batch_idx, (inputs, labels) in enumerate(dataloader):
    if k >= N:
        break

    inputs, labels = inputs.to(device), labels.to(device)
    outputs = resnet_model(inputs)
    preds = outputs.argmax(dim=1)

    b = min(inputs.size(0), N - k)
    y_pred[k:k+b] = preds[:b].cpu().numpy()
    y_true[k:k+b] = labels[:b].cpu().numpy()
    k += b

    print("Processed {:d}/{:d} images | Accuracy: {:.2f}%".format(
        k, N, 100*np.sum(y_pred[:k] == y_true[:k]) / k), end='\r')

end_time = time.time()
acc = 100 * np.sum(y_pred == y_true) / N
print(f"\nInference complete in {end_time - start_time:.2f}s")
print(f"Top-1 Accuracy: {acc:.2f}% ({int(np.sum(y_pred == y_true))}/{N})")

# Save activations
first_layer_outputs_np = torch.stack(batch_first_layer_outputs).numpy()
final_layer_outputs_np = torch.stack(final_layer_outputs).numpy()

os.makedirs("activations", exist_ok=True)
np.save("activations/first_layer_outputs_bn1.npy", first_layer_outputs_np)
np.save("activations/final_layer_outputs.npy", final_layer_outputs_np)

print("Saved activations:")
print(" - BN1 outputs shape:", first_layer_outputs_np.shape)
print(" - Final layer outputs shape:", final_layer_outputs_np.shape)
