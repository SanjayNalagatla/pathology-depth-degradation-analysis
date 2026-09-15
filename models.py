"""
models.py - Neural Network Architectures for Pathology Image Classification
Compares:
1. ShallowPathologyNet: 2 hidden conv layers + bottleneck (3 total feature layers)
2. DeepPathologyNet: 10 hidden conv layers + bottleneck (plain feedforward without residual shortcuts)

Includes gradient tracking hooks, representation extraction, and complexity profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowPathologyNet(nn.Module):
    """
    Shallow Architecture (2 Conv hidden layers + 1 Dense bottleneck)
    Represents an easily trainable network with robust gradient propagation.
    """
    def __init__(self, num_classes=2, latent_dim=64):
        super(ShallowPathologyNet, self).__init__()
        self.name = "ShallowNet (2 Hidden Layers)"
        self.latent_dim = latent_dim
        
        # Layer 1: Conv Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        # Layer 2: Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Adaptive pooling to fixed 4x4 spatial grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Penultimate dense representation layer
        self.fc_latent = nn.Linear(32 * 4 * 4, latent_dim)
        
        # Final classification layer
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Track gradient history
        self.grad_history = {}

    def forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        latent = F.relu(self.fc_latent(x))
        return latent

    def forward(self, x):
        latent = self.forward_features(x)
        logits = self.classifier(latent)
        return logits

    def get_layer_gradient_norms(self):
        """Compute the Frobenius L2 norm of gradients for each key weight layer."""
        grads = {}
        for name, param in self.named_parameters():
            if 'weight' in name and param.grad is not None:
                grads[name] = param.grad.data.norm(2).item()
        return grads


class DeepPathologyNet(nn.Module):
    """
    Deep Plain Architecture (10 Conv hidden layers + 1 Dense bottleneck)
    Constructed without residual/skip connections to illustrate the classic
    architectural challenges: vanishing gradients, degradation problem, and optimization pathology.
    """
    def __init__(self, num_classes=2, latent_dim=64):
        super(DeepPathologyNet, self).__init__()
        self.name = "DeepNet (10 Hidden Layers)"
        self.latent_dim = latent_dim
        
        # Stage 1: Conv layers 1-3 (16 channels)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        # Stage 2: Conv layers 4-6 (32 channels)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Stage 3: Conv layers 7-10 (32 channels)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Penultimate representation layer
        self.fc_latent = nn.Linear(32 * 4 * 4, latent_dim)
        
        # Final classification layer
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        self.grad_history = {}

    def forward_features(self, x):
        # Stage 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        
        # Stage 2
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        
        # Stage 3
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        latent = F.relu(self.fc_latent(x))
        return latent

    def forward(self, x):
        latent = self.forward_features(x)
        logits = self.classifier(latent)
        return logits

    def get_layer_gradient_norms(self):
        """Compute the Frobenius L2 norm of gradients for each key weight layer."""
        grads = {}
        for name, param in self.named_parameters():
            if 'weight' in name and param.grad is not None:
                grads[name] = param.grad.data.norm(2).item()
        return grads


def calculate_model_complexity(model, input_size=(1, 3, 64, 64)):
    """
    Profile model complexity:
    - Trainable Parameter Count
    - Non-trainable Parameter Count
    - Model Size in Memory (KB)
    - Theoretical Multiply-Accumulate Operations (MACs / FLOPs estimate)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    size_kb = (param_bytes + buffer_bytes) / 1024.0
    
    # Estimate FLOPs (2 * MACs) using a forward pass hook
    flops = 0
    hooks = []
    
    def conv_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        output_channels, output_h, output_w = output.shape[1:]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        # MACs per output element: in_channels * kernel_h * kernel_w
        macs = batch_size * output_channels * output_h * output_w * (in_channels * kernel_h * kernel_w)
        flops += 2 * macs
        
    def linear_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].size(0)
        in_features = module.in_features
        out_features = module.out_features
        macs = batch_size * in_features * out_features
        flops += 2 * macs
        
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
            
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        _ = model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    return {
        "model_name": getattr(model, 'name', 'Model'),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_kb": round(size_kb, 2),
        "macs": int(flops // 2),
        "flops": int(flops),
        "flops_m": round(flops / 1e6, 2)
    }

if __name__ == "__main__":
    shallow = ShallowPathologyNet()
    deep = DeepPathologyNet()
    
    comp_shallow = calculate_model_complexity(shallow)
    comp_deep = calculate_model_complexity(deep)
    
    print("--- ShallowNet Complexity ---")
    for k, v in comp_shallow.items():
        print(f"  {k}: {v}")
        
    print("\n--- DeepNet Complexity ---")
    for k, v in comp_deep.items():
        print(f"  {k}: {v}")
