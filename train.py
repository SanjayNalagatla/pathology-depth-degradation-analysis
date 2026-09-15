"""
train.py - Training, Evaluation, and Gradient Tracking Engine
Compares ShallowPathologyNet vs DeepPathologyNet on the histopathology task.
Records:
- Per-epoch loss & accuracy (train and validation)
- Layer-wise gradient norm dynamics across depth
- Detailed test metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Latent feature embeddings for manifold visualization
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve

from dataset import get_data_loaders
from models import ShallowPathologyNet, DeepPathologyNet, calculate_model_complexity

def train_and_evaluate_model(model, train_loader, val_loader, test_loader, 
                             epochs=25, lr=0.01, device='cpu'):
    """
    Train a model while tracking layer-wise gradient norms and evaluation metrics.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Standard SGD with momentum highlights depth optimization dynamics clearly
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
        "layer_gradient_norms": {}  # layer_name -> list of epoch mean gradient norms
    }
    
    # Initialize gradient tracking structures for key weight tensors
    weight_layers = [name for name, param in model.named_parameters() if 'weight' in name]
    for layer in weight_layers:
        history["layer_gradient_norms"][layer] = []
        
    print(f"\n==========================================")
    print(f"Starting Training: {model.name}")
    print(f"Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"==========================================")
    
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Temp accumulator for gradient norms in this epoch
        epoch_layer_grads = {layer: [] for layer in weight_layers}
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Record gradient norm for each weight layer before optimizer step
            for name, param in model.named_parameters():
                if name in epoch_layer_grads and param.grad is not None:
                    epoch_layer_grads[name].append(param.grad.data.norm(2).item())
                    
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        epoch_duration = time.time() - epoch_start_time
        
        # Aggregate gradient norms for epoch
        for layer in weight_layers:
            mean_grad = np.mean(epoch_layer_grads[layer]) if epoch_layer_grads[layer] else 0.0
            history["layer_gradient_norms"][layer].append(float(mean_grad))
            
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["epoch_times"].append(float(epoch_duration))
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                  f"Time: {epoch_duration:.2f}s")
            
    total_training_time = time.time() - total_start_time
    print(f"Training completed in {total_training_time:.2f}s")
    
    # Test Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    all_latents = []
    
    # Inference Latency Benchmark
    latency_times = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            latency_times.append((t1 - t0) / images.size(0))
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            latents = model.forward_features(images)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_latents.extend(latents.cpu().numpy())
            
    test_loss = test_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_latents = np.array(all_latents)
    
    # Metrics calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    avg_latency_ms = float(np.mean(latency_times) * 1000.0)
    throughput = float(1.0 / np.mean(latency_times))
    
    complexity = calculate_model_complexity(model)
    
    test_metrics = {
        "model_name": model.name,
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "total_training_time_s": round(total_training_time, 2),
        "avg_latency_ms_per_sample": round(avg_latency_ms, 4),
        "throughput_samples_per_sec": round(throughput, 1),
        "complexity": complexity
    }
    
    return history, test_metrics, all_latents, all_labels

def run_experiment(num_samples=1600, epochs=25, batch_size=32, lr=0.01):
    os.makedirs("results", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running experiment on device: {device}")
    
    train_loader, val_loader, test_loader = get_data_loaders(num_samples=num_samples, batch_size=batch_size)
    
    # Instantiate models
    shallow_net = ShallowPathologyNet()
    deep_net = DeepPathologyNet()
    
    # Train ShallowNet
    history_shallow, test_metrics_shallow, latents_shallow, labels_test = train_and_evaluate_model(
        shallow_net, train_loader, val_loader, test_loader, epochs=epochs, lr=lr, device=device
    )
    
    # Train DeepNet
    history_deep, test_metrics_deep, latents_deep, _ = train_and_evaluate_model(
        deep_net, train_loader, val_loader, test_loader, epochs=epochs, lr=lr, device=device
    )
    
    # Save model weights
    torch.save(shallow_net.state_dict(), "results/shallow_net.pth")
    torch.save(deep_net.state_dict(), "results/deep_net.pth")
    print("Saved model checkpoints to results/")
    
    # Save latent representations
    np.savez_compressed(
        "results/test_embeddings.npz",
        latents_shallow=latents_shallow,
        latents_deep=latents_deep,
        labels=labels_test
    )
    print("Saved latent test representations to results/test_embeddings.npz")
    
    # Save histories and metrics
    all_histories = {
        "shallow": history_shallow,
        "deep": history_deep
    }
    with open("results/training_history.json", "w") as f:
        json.dump(all_histories, f, indent=2)
        
    all_metrics = {
        "shallow": test_metrics_shallow,
        "deep": test_metrics_deep
    }
    with open("results/test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print("Saved metrics and training histories to results/")
    print("\nExperiment Run Complete!")

if __name__ == "__main__":
    run_experiment(num_samples=1600, epochs=25, batch_size=32, lr=0.01)
