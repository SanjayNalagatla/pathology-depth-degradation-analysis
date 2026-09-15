"""
dataset.py - Histopathology Image Dataset Generator and PyTorch DataLoader
Simulates high-fidelity 64x64 H&E stained pathology patches:
- Class 0: Benign (organized glandular structures, low nuclear-to-cytoplasmic ratio, regular nuclei)
- Class 1: Malignant (severe nuclear pleomorphism, hyperchromasia, high cell density/crowding, architectural disorder)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Fixed random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_pathology_patch(label, size=64):
    """
    Synthesize an H&E stained tissue patch (size x size x 3).
    H&E Staining properties:
    - Hematoxylin: Deep purple/blue stain for cell nuclei and nucleic acids.
    - Eosin: Pink/magenta stain for cytoplasm, collagen, and extracellular matrix.
    """
    # Base background: Eosinophilic stroma (light pink / magenta)
    # RGB baseline: R ~ [0.85, 0.95], G ~ [0.70, 0.82], B ~ [0.80, 0.90]
    base_r = np.random.uniform(0.85, 0.92)
    base_g = np.random.uniform(0.72, 0.80)
    base_b = np.random.uniform(0.80, 0.88)
    
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = base_r + np.random.normal(0, 0.02, (size, size))
    img[:, :, 1] = base_g + np.random.normal(0, 0.02, (size, size))
    img[:, :, 2] = base_b + np.random.normal(0, 0.02, (size, size))

    # Add collagen / connective tissue fibrils (linear eosinophilic striations)
    for _ in range(np.random.randint(3, 7)):
        angle = np.random.uniform(0, np.pi)
        x0, y0 = np.random.uniform(0, size, 2)
        length = np.random.uniform(size * 0.4, size * 0.9)
        for t in np.linspace(-length/2, length/2, int(length * 2)):
            xi = int(x0 + t * np.cos(angle))
            yi = int(y0 + t * np.sin(angle))
            if 0 <= xi < size and 0 <= yi < size:
                # Eosin-rich fiber (darker pink)
                img[max(0, yi-1):min(size, yi+2), max(0, xi-1):min(size, xi+2), 0] *= 0.95
                img[max(0, yi-1):min(size, yi+2), max(0, xi-1):min(size, xi+2), 1] *= 0.88
                img[max(0, yi-1):min(size, yi+2), max(0, xi-1):min(size, xi+2), 2] *= 0.92

    yy, xx = np.mgrid[0:size, 0:size]

    if label == 0:
        # BENIGN:
        # 1. Glandular / lumen architecture (circular or elliptical clear lumen)
        # 2. Regularly spaced, uniform-sized basal nuclei (low density, 12-25 nuclei)
        # 3. Moderate to low nuclear-to-cytoplasmic ratio
        num_glands = np.random.randint(1, 3)
        for _ in range(num_glands):
            gx = np.random.uniform(size * 0.25, size * 0.75)
            gy = np.random.uniform(size * 0.25, size * 0.75)
            rx = np.random.uniform(size * 0.15, size * 0.25)
            ry = np.random.uniform(size * 0.15, size * 0.25)
            lumen_mask = (((xx - gx) / rx)**2 + ((yy - gy) / ry)**2) < 1.0
            # Lumen is pale/white
            img[lumen_mask] = np.clip(img[lumen_mask] + 0.18, 0, 0.98)

        num_nuclei = np.random.randint(15, 28)
        for _ in range(num_nuclei):
            cx = np.random.uniform(4, size - 5)
            cy = np.random.uniform(4, size - 5)
            # Regular small elliptical radius (radius ~ 2.0 to 3.2)
            rx = np.random.uniform(2.0, 3.2)
            ry = np.random.uniform(2.0, 3.2)
            dist_sq = ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2
            nucleus_mask = dist_sq < 1.0
            
            # Hematoxylin stain: deep purple/blue (Low R, Low G, Higher B relative)
            if np.any(nucleus_mask):
                intensity = np.exp(-0.5 * dist_sq[nucleus_mask])
                img[nucleus_mask, 0] = img[nucleus_mask, 0] * (1 - 0.65 * intensity) + 0.32 * (0.65 * intensity)
                img[nucleus_mask, 1] = img[nucleus_mask, 1] * (1 - 0.78 * intensity) + 0.18 * (0.78 * intensity)
                img[nucleus_mask, 2] = img[nucleus_mask, 2] * (1 - 0.45 * intensity) + 0.52 * (0.45 * intensity)
    else:
        # MALIGNANT:
        # 1. High nuclear density and crowding (55-95 nuclei)
        # 2. Marked nuclear pleomorphism (irregular shapes, enlargement, hyperchromasia)
        # 3. High nuclear-to-cytoplasmic (N:C) ratio, loss of organized glandular structures
        # 4. Focal necrotic / dense chromatin clusters
        num_nuclei = np.random.randint(55, 95)
        for _ in range(num_nuclei):
            cx = np.random.uniform(2, size - 3)
            cy = np.random.uniform(2, size - 3)
            # Irregular enlarged nuclei (radius ~ 2.8 to 5.2) with random orientation
            rx = np.random.uniform(2.8, 5.2)
            ry = np.random.uniform(2.2, 4.2)
            angle = np.random.uniform(0, np.pi)
            
            x_rot = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            y_rot = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
            dist_sq = (x_rot / rx)**2 + (y_rot / ry)**2
            nucleus_mask = dist_sq < 1.0
            
            if np.any(nucleus_mask):
                intensity = np.exp(-0.4 * dist_sq[nucleus_mask])
                # Hyperchromatic (much darker purple/blue)
                img[nucleus_mask, 0] = img[nucleus_mask, 0] * (1 - 0.80 * intensity) + 0.22 * (0.80 * intensity)
                img[nucleus_mask, 1] = img[nucleus_mask, 1] * (1 - 0.88 * intensity) + 0.10 * (0.88 * intensity)
                img[nucleus_mask, 2] = img[nucleus_mask, 2] * (1 - 0.60 * intensity) + 0.48 * (0.60 * intensity)

    # Add realistic microscopy optical blur / sensor noise
    img = np.clip(img + np.random.normal(0, 0.015, img.shape), 0.0, 1.0)
    return img

class PathologyDataset(Dataset):
    def __init__(self, num_samples=1600, size=64, balance=0.5):
        super().__init__()
        self.size = size
        self.num_samples = num_samples
        
        self.images = []
        self.labels = []
        
        num_pos = int(num_samples * balance)
        num_neg = num_samples - num_pos
        
        # Class 0: Benign
        for _ in range(num_neg):
            img = generate_pathology_patch(0, size=size)
            self.images.append(img)
            self.labels.append(0)
            
        # Class 1: Malignant
        for _ in range(num_pos):
            img = generate_pathology_patch(1, size=size)
            self.images.append(img)
            self.labels.append(1)
            
        # Shuffle indices deterministically
        indices = np.random.permutation(num_samples)
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert HWC [0, 1] to CHW torch.FloatTensor
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        # Standardize
        mean = torch.tensor([0.70, 0.60, 0.68]).view(3, 1, 1)
        std = torch.tensor([0.20, 0.20, 0.20]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor, torch.tensor(label, dtype=torch.long)

def get_data_loaders(num_samples=1600, batch_size=32, train_val_test=(0.70, 0.15, 0.15)):
    """Generate train, val, and test DataLoaders."""
    total_samples = num_samples
    dataset = PathologyDataset(num_samples=total_samples, size=64)
    
    n_train = int(total_samples * train_val_test[0])
    n_val = int(total_samples * train_val_test[1])
    n_test = total_samples - n_train - n_val
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def visualize_dataset_samples(output_path="results/sample_pathology_patches.png"):
    """Save a sample grid comparing Benign vs Malignant pathology patches."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    fig.suptitle("Histopathology Image Classification Dataset (H&E Staining Simulation)\nBenign Glandular Tissue vs. Malignant Nuclear Pleomorphism", fontsize=13, fontweight='bold', y=0.98)
    
    for i in range(6):
        benign_img = generate_pathology_patch(0, size=64)
        axes[0, i].imshow(benign_img)
        axes[0, i].set_title(f"Benign #{i+1}", fontsize=10, fontweight='bold', color='#1b5e20')
        axes[0, i].axis('off')
        
        malignant_img = generate_pathology_patch(1, size=64)
        axes[1, i].imshow(malignant_img)
        axes[1, i].set_title(f"Malignant #{i+1}", fontsize=10, fontweight='bold', color='#b71c1c')
        axes[1, i].axis('off')
        
    axes[0, 0].text(-15, 32, "CLASS 0:\nBENIGN", fontsize=11, fontweight='bold', color='#1b5e20',
                    ha='right', va='center', rotation=90)
    axes[1, 0].text(-15, 32, "CLASS 1:\nMALIGNANT", fontsize=11, fontweight='bold', color='#b71c1c',
                    ha='right', va='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample pathology patches saved successfully to {output_path}")

if __name__ == "__main__":
    visualize_dataset_samples()
