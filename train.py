"""
train.py - Training, Evaluation, and Gradient Tracking Engine
Compares ShallowPathologyNet vs DeepPathologyNet on the histopathology task.
Records:
- Per-epoch loss & accuracy (train and validation)
- Layer-wise gradient norm dynamics across depth
- Detailed test metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Latent feature embeddings for manifold visualization
Supports progress and log streaming callbacks for real-time dashboard execution.
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
                             epochs=25, lr=0.01, device='cpu',
                             progress_callback=None, log_callback=None):
    """
    Train a model while tracking layer-wise gradient norms and evaluation metrics.
    Supports progress and log callbacks for real-time dashboard updates.
    """
    def emit_log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

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
        
    emit_log(f"\n==========================================")
    emit_log(f"Starting Training: {model.name}")
    emit_log(f"Epochs: {epochs} | LR: {lr} | Device: {device}")
    emit_log(f"==========================================")
    
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
        
        epoch_msg = (f"[{model.name}] Epoch [{epoch:02d}/{epochs:02d}] "
                     f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                     f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                     f"Time: {epoch_duration:.2f}s")
        emit_log(epoch_msg)
        
        if progress_callback:
            progress_callback({
                "type": "epoch",
                "model": model.name,
                "model_key": "shallow" if "Shallow" in model.name else "deep",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "epoch_duration": float(epoch_duration)
            })
            
    total_training_time = time.time() - total_start_time
    emit_log(f"Training of {model.name} completed in {total_training_time:.2f}s")
    
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
    
    emit_log(f"\nEvaluation Results: {model.name}")
    emit_log(f"Test Accuracy: {accuracy*100:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    emit_log(f"Avg Inference Latency: {avg_latency_ms:.4f} ms | Throughput: {throughput:.1f} samples/sec")
    emit_log(f"Parameters: {complexity['trainable_params']:,} | FLOPs: {complexity['flops_m']:.2f} M | Size: {complexity['size_kb']:.2f} KB")
    
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

def run_experiment(num_samples=1600, epochs=25, batch_size=32, lr=0.01, device=None,
                   progress_callback=None, log_callback=None):
    def emit_log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    os.makedirs("results", exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emit_log(f"Running experiment on device: {device} | Samples: {num_samples} | Epochs: {epochs} | LR: {lr}")
    
    train_loader, val_loader, test_loader = get_data_loaders(num_samples=num_samples, batch_size=batch_size)
    
    # Instantiate models
    shallow_net = ShallowPathologyNet()
    deep_net = DeepPathologyNet()
    
    # Train ShallowNet
    emit_log("\n>>> Phase 1/2: Training ShallowNet (2 Hidden Layers) <<<")
    history_shallow, test_metrics_shallow, latents_shallow, labels_test = train_and_evaluate_model(
        shallow_net, train_loader, val_loader, test_loader, epochs=epochs, lr=lr, device=device,
        progress_callback=progress_callback, log_callback=log_callback
    )
    
    # Train DeepNet
    emit_log("\n>>> Phase 2/2: Training DeepNet (10 Hidden Layers) <<<")
    history_deep, test_metrics_deep, latents_deep, _ = train_and_evaluate_model(
        deep_net, train_loader, val_loader, test_loader, epochs=epochs, lr=lr, device=device,
        progress_callback=progress_callback, log_callback=log_callback
    )
    
    # Save model weights
    torch.save(shallow_net.state_dict(), "results/shallow_net.pth")
    torch.save(deep_net.state_dict(), "results/deep_net.pth")
    emit_log("Saved model checkpoints to results/")
    
    # Save latent representations
    np.savez_compressed(
        "results/test_embeddings.npz",
        latents_shallow=latents_shallow,
        latents_deep=latents_deep,
        labels=labels_test
    )
    emit_log("Saved latent test representations to results/test_embeddings.npz")
    
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
        
    emit_log("Saved metrics and training histories to results/")
    
    # Automatically regenerate all analytical plots & summary
    emit_log("\n>>> Regenerating All Diagnostic Plots & Analysis Summary <<<")
    try:
        import analyze
        analyze.generate_summary()
        emit_log("All diagnostic figures and metrics summary successfully refreshed in 'results/'!")
    except Exception as e:
        emit_log(f"Plot regeneration notice: {e}")
        
    emit_log("\n[SUCCESS] Experiment Run Complete!")
    return all_histories, all_metrics

if __name__ == "__main__":
    run_experiment(num_samples=1600, epochs=25, batch_size=32, lr=0.01)
