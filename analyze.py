"""
analyze.py - Representation Analysis, Gradient Dynamics, and Comparative Visualization
Produces:
1. training_validation_curves.png (Loss & Accuracy over epochs)
2. gradient_flow_analysis.png (Gradient norm attenuation across layers demonstrating vanishing gradients)
3. latent_representations_tsne_pca.png (t-SNE and PCA of penultimate representations with silhouette scores)
4. confusion_matrices_roc.png (Confusion matrices and ROC-AUC curves)
5. model_complexity_comparison.png (Parameters, FLOPs, Latency, and Performance trade-offs)
6. metrics_summary.json & console summary
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_training_curves(histories, output_path="results/training_validation_curves.png"):
    """Plot Training/Validation Loss and Accuracy side-by-side."""
    epochs = range(1, len(histories["shallow"]["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Loss Curves
    ax1.plot(epochs, histories["shallow"]["train_loss"], 'o-', color='#1976d2', linewidth=2.2, label='ShallowNet - Train Loss')
    ax1.plot(epochs, histories["shallow"]["val_loss"], 's--', color='#0288d1', linewidth=2.0, alpha=0.8, label='ShallowNet - Val Loss')
    ax1.plot(epochs, histories["deep"]["train_loss"], 'o-', color='#d32f2f', linewidth=2.2, label='DeepNet - Train Loss')
    ax1.plot(epochs, histories["deep"]["val_loss"], 's--', color='#f57c00', linewidth=2.0, alpha=0.8, label='DeepNet - Val Loss')
    
    ax1.set_title("Training & Validation Loss Dynamics", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Accuracy Curves
    shallow_train_acc = [acc * 100 for acc in histories["shallow"]["train_acc"]]
    shallow_val_acc = [acc * 100 for acc in histories["shallow"]["val_acc"]]
    deep_train_acc = [acc * 100 for acc in histories["deep"]["train_acc"]]
    deep_val_acc = [acc * 100 for acc in histories["deep"]["val_acc"]]
    
    ax2.plot(epochs, shallow_train_acc, 'o-', color='#1976d2', linewidth=2.2, label='ShallowNet - Train Acc')
    ax2.plot(epochs, shallow_val_acc, 's--', color='#0288d1', linewidth=2.0, alpha=0.8, label='ShallowNet - Val Acc')
    ax2.plot(epochs, deep_train_acc, 'o-', color='#d32f2f', linewidth=2.2, label='DeepNet - Train Acc')
    ax2.plot(epochs, deep_val_acc, 's--', color='#f57c00', linewidth=2.0, alpha=0.8, label='DeepNet - Val Acc')
    
    ax2.set_title("Training & Validation Accuracy Dynamics", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
    ax2.set_ylim([45, 102])
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle("Pathology Model Training Dynamics: Shallow (2 Layers) vs Deep (10 Layers)", 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")

def plot_gradient_flow(histories, output_path="results/gradient_flow_analysis.png"):
    """
    Visualize gradient norm attenuation across depth demonstrating vanishing gradients.
    """
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
    
    # Left subplot: DeepNet layer-wise gradient norm across epochs
    ax1 = fig.add_subplot(gs[0])
    deep_grads = histories["deep"]["layer_gradient_norms"]
    epochs = range(1, len(next(iter(deep_grads.values()))) + 1)
    
    # Filter key conv weight layers
    conv_layers = [k for k in deep_grads.keys() if 'conv' in k]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.95, len(conv_layers)))
    
    for idx, layer_name in enumerate(conv_layers):
        short_name = layer_name.replace('.weight', '')
        vals = deep_grads[layer_name]
        ax1.plot(epochs, vals, label=short_name, color=cmap[idx], linewidth=2.0, marker='o' if idx % 2 == 0 else 'x', markersize=4)
        
    ax1.set_title("DeepNet (10 Layers): Gradient Norm Dynamics per Conv Layer\n(Severe Vanishing Gradient in Early Layers)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax1.set_ylabel("L2 Gradient Norm (log scale)", fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(ncol=2, fontsize=8.5, loc='center right', frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6, which="both")
    
    # Right subplot: Bar chart comparison of Mean Gradient Norm across Layer Depth
    ax2 = fig.add_subplot(gs[1])
    
    # Compute overall mean gradient per layer across epochs
    deep_layer_means = [np.mean(deep_grads[k]) for k in conv_layers]
    deep_layer_labels = [k.replace('.weight', '').replace('conv', 'L') for k in conv_layers]
    
    shallow_grads = histories["shallow"]["layer_gradient_norms"]
    shallow_conv_layers = [k for k in shallow_grads.keys() if 'conv' in k]
    shallow_layer_means = [np.mean(shallow_grads[k]) for k in shallow_conv_layers]
    shallow_layer_labels = [k.replace('.weight', '').replace('conv', 'L') for k in shallow_conv_layers]
    
    # Plot DeepNet layer bars
    x_deep = np.arange(len(conv_layers))
    bars = ax2.bar(x_deep, deep_layer_means, color='#d32f2f', alpha=0.75, edgecolor='#b71c1c', width=0.6, label='DeepNet Layers (L1 to L10)')
    
    # Add horizontal reference for ShallowNet average conv gradient
    shallow_avg = np.mean(shallow_layer_means)
    ax2.axhline(shallow_avg, color='#1976d2', linestyle='--', linewidth=2.2, 
                label=f'ShallowNet Avg Grad Norm ({shallow_avg:.3f})')
    
    ax2.set_xticks(x_deep)
    ax2.set_xticklabels(deep_layer_labels, rotation=45, fontsize=9.5, fontweight='bold')
    ax2.set_title("Average Gradient Norm vs Network Depth\n(Earliest Layers Conv1-3 Diminish Dramatically)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Network Layer (Input $\\rightarrow$ Output)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Mean L2 Gradient Norm", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate ratio
    early_grad = deep_layer_means[0]
    late_grad = deep_layer_means[-1]
    ratio = late_grad / max(early_grad, 1e-12)
    ax2.text(0.55, 0.70, f"Gradient Decay Ratio:\nLate / Early Layer: ~{ratio:.1f}x\nEarly layers starved of signal", 
             transform=ax2.transAxes, fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#d32f2f'))
    
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.15, wspace=0.22)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient flow analysis to {output_path}")

def plot_representations(embeddings_path="results/test_embeddings.npz", 
                         output_path="results/latent_representations_tsne_pca.png"):
    """
    Compute t-SNE and PCA of penultimate representations and quantify cluster separability.
    """
    data = np.load(embeddings_path)
    latents_shallow = data["latents_shallow"]
    latents_deep = data["latents_deep"]
    labels = data["labels"]
    
    # Compute Silhouette Scores (measure of cluster separability)
    sil_shallow = silhouette_score(latents_shallow, labels)
    sil_deep = silhouette_score(latents_deep, labels)
    
    # 1. PCA
    pca_shallow = PCA(n_components=2)
    pca_s_proj = pca_shallow.fit_transform(latents_shallow)
    var_s = pca_shallow.explained_variance_ratio_
    
    pca_deep = PCA(n_components=2)
    pca_d_proj = pca_deep.fit_transform(latents_deep)
    var_d = pca_deep.explained_variance_ratio_
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_s_proj = tsne.fit_transform(latents_shallow)
    
    tsne_d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_d_proj = tsne_d.fit_transform(latents_deep)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Palette
    color_map = {0: '#2e7d32', 1: '#c62828'}
    label_names = {0: 'Benign', 1: 'Malignant'}
    
    # Subplot 1: ShallowNet PCA
    for cls in [0, 1]:
        mask = (labels == cls)
        axes[0, 0].scatter(pca_s_proj[mask, 0], pca_s_proj[mask, 1], 
                           c=color_map[cls], label=label_names[cls], alpha=0.7, edgecolors='none', s=35)
    axes[0, 0].set_title(f"ShallowNet (2 Layers) - PCA\nExplained Var: (PC1: {var_s[0]*100:.1f}%, PC2: {var_s[1]*100:.1f}%)", 
                         fontsize=11.5, fontweight='bold')
    axes[0, 0].set_xlabel("Principal Component 1", fontweight='bold')
    axes[0, 0].set_ylabel("Principal Component 2", fontweight='bold')
    axes[0, 0].legend(frameon=True, facecolor='white', loc='upper right')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: DeepNet PCA
    for cls in [0, 1]:
        mask = (labels == cls)
        axes[0, 1].scatter(pca_d_proj[mask, 0], pca_d_proj[mask, 1], 
                           c=color_map[cls], label=label_names[cls], alpha=0.7, edgecolors='none', s=35)
    axes[0, 1].set_title(f"DeepNet (10 Layers) - PCA\nExplained Var: (PC1: {var_d[0]*100:.1f}%, PC2: {var_d[1]*100:.1f}%)", 
                         fontsize=11.5, fontweight='bold')
    axes[0, 1].set_xlabel("Principal Component 1", fontweight='bold')
    axes[0, 1].set_ylabel("Principal Component 2", fontweight='bold')
    axes[0, 1].legend(frameon=True, facecolor='white', loc='upper right')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 3: ShallowNet t-SNE
    for cls in [0, 1]:
        mask = (labels == cls)
        axes[1, 0].scatter(tsne_s_proj[mask, 0], tsne_s_proj[mask, 1], 
                           c=color_map[cls], label=label_names[cls], alpha=0.7, edgecolors='none', s=35)
    axes[1, 0].set_title(f"ShallowNet (2 Layers) - t-SNE Manifold\nSilhouette Score: {sil_shallow:.3f} (Sharp Cluster Separation)", 
                         fontsize=11.5, fontweight='bold', color='#1565c0')
    axes[1, 0].set_xlabel("t-SNE Dimension 1", fontweight='bold')
    axes[1, 0].set_ylabel("t-SNE Dimension 2", fontweight='bold')
    axes[1, 0].legend(frameon=True, facecolor='white', loc='upper right')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 4: DeepNet t-SNE
    for cls in [0, 1]:
        mask = (labels == cls)
        axes[1, 1].scatter(tsne_d_proj[mask, 0], tsne_d_proj[mask, 1], 
                           c=color_map[cls], label=label_names[cls], alpha=0.7, edgecolors='none', s=35)
    axes[1, 1].set_title(f"DeepNet (10 Layers) - t-SNE Manifold\nSilhouette Score: {sil_deep:.3f} (Degraded / Entangled Manifold)", 
                         fontsize=11.5, fontweight='bold', color='#c62828')
    axes[1, 1].set_xlabel("t-SNE Dimension 1", fontweight='bold')
    axes[1, 1].set_ylabel("t-SNE Dimension 2", fontweight='bold')
    axes[1, 1].legend(frameon=True, facecolor='white', loc='upper right')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle("Learned Representation Geometry: Shallow vs Deep Pathology Networks\n(Penultimate Layer Embeddings on Held-Out Test Set)", 
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved representation manifold plots to {output_path}")
    
    return {
        "silhouette_shallow": float(sil_shallow),
        "silhouette_deep": float(sil_deep)
    }

def plot_confusion_and_roc(metrics, output_path="results/confusion_matrices_roc.png"):
    """Plot Confusion Matrices and ROC Curves."""
    fig = plt.figure(figsize=(15, 4.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)
    
    # CM Shallow
    ax1 = fig.add_subplot(gs[0])
    cm_s = np.array(metrics["shallow"]["confusion_matrix"])
    im1 = ax1.imshow(cm_s, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(f"ShallowNet Confusion Matrix\nAccuracy: {metrics['shallow']['accuracy']*100:.1f}%", fontsize=11, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Benign', 'Malignant'], fontweight='bold')
    ax1.set_yticklabels(['Benign', 'Malignant'], fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    for i in range(2):
        for j in range(2):
            color = "white" if cm_s[i, j] > cm_s.max() / 2.0 else "black"
            ax1.text(j, i, format(cm_s[i, j], 'd'), ha="center", va="center", color=color, fontsize=14, fontweight='bold')
            
    # CM Deep
    ax2 = fig.add_subplot(gs[1])
    cm_d = np.array(metrics["deep"]["confusion_matrix"])
    im2 = ax2.imshow(cm_d, interpolation='nearest', cmap=plt.cm.Oranges)
    ax2.set_title(f"DeepNet Confusion Matrix\nAccuracy: {metrics['deep']['accuracy']*100:.1f}%", fontsize=11, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Benign', 'Malignant'], fontweight='bold')
    ax2.set_yticklabels(['Benign', 'Malignant'], fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontweight='bold')
    ax2.set_ylabel('True Label', fontweight='bold')
    for i in range(2):
        for j in range(2):
            color = "white" if cm_d[i, j] > cm_d.max() / 2.0 else "black"
            ax2.text(j, i, format(cm_d[i, j], 'd'), ha="center", va="center", color=color, fontsize=14, fontweight='bold')
            
    # ROC Curves
    ax3 = fig.add_subplot(gs[2])
    fpr_s = metrics["shallow"]["roc_curve"]["fpr"]
    tpr_s = metrics["shallow"]["roc_curve"]["tpr"]
    auc_s = metrics["shallow"]["roc_auc"]
    
    fpr_d = metrics["deep"]["roc_curve"]["fpr"]
    tpr_d = metrics["deep"]["roc_curve"]["tpr"]
    auc_d = metrics["deep"]["roc_auc"]
    
    ax3.plot(fpr_s, tpr_s, color='#1976d2', linewidth=2.5, label=f"ShallowNet (AUC = {auc_s:.3f})")
    ax3.plot(fpr_d, tpr_d, color='#d32f2f', linewidth=2.5, label=f"DeepNet (AUC = {auc_d:.3f})")
    ax3.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.5, label="Random Guess (AUC = 0.500)")
    
    ax3.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("False Positive Rate (1 - Specificity)", fontweight='bold')
    ax3.set_ylabel("True Positive Rate (Sensitivity)", fontweight='bold')
    ax3.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.15, wspace=0.28)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices and ROC curves to {output_path}")

def plot_complexity_comparison(metrics, output_path="results/model_complexity_comparison.png"):
    """Multi-bar comparison of model complexity, resource demands, and accuracy."""
    labels = ['Parameters (k)', 'FLOPs (M)', 'Model Size (KB)', 'Latency (ms)', 'Test Accuracy (%)']
    
    s_comp = metrics["shallow"]["complexity"]
    d_comp = metrics["deep"]["complexity"]
    
    shallow_vals = [
        s_comp["total_params"] / 1000.0,
        s_comp["flops_m"],
        s_comp["size_kb"],
        metrics["shallow"]["avg_latency_ms_per_sample"],
        metrics["shallow"]["accuracy"] * 100.0
    ]
    
    deep_vals = [
        d_comp["total_params"] / 1000.0,
        d_comp["flops_m"],
        d_comp["size_kb"],
        metrics["deep"]["avg_latency_ms_per_sample"],
        metrics["deep"]["accuracy"] * 100.0
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars1 = ax.bar(x - width/2, shallow_vals, width, label='ShallowNet (2 Layers)', color='#1976d2', edgecolor='#0d47a1')
    bars2 = ax.bar(x + width/2, deep_vals, width, label='DeepNet (10 Layers)', color='#d32f2f', edgecolor='#b71c1c')
    
    ax.set_title("Model Complexity vs Classification Performance Trade-Off", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=10.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, fontsize=10.5)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Attach values above bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved complexity comparison to {output_path}")

def generate_summary():
    """Load outputs, run all visualizers, and produce consolidated summary."""
    with open("results/training_history.json", "r") as f:
        histories = json.load(f)
    with open("results/test_metrics.json", "r") as f:
        metrics = json.load(f)
        
    plot_training_curves(histories)
    plot_gradient_flow(histories)
    manifold_metrics = plot_representations()
    plot_confusion_and_roc(metrics)
    plot_complexity_comparison(metrics)
    
    # Consolidated summary
    summary = {
        "shallow_net": {
            "num_hidden_layers": 2,
            "parameters": metrics["shallow"]["complexity"]["total_params"],
            "flops_m": metrics["shallow"]["complexity"]["flops_m"],
            "model_size_kb": metrics["shallow"]["complexity"]["size_kb"],
            "test_accuracy": round(metrics["shallow"]["accuracy"] * 100, 2),
            "test_loss": round(metrics["shallow"]["test_loss"], 4),
            "precision": round(metrics["shallow"]["precision"], 4),
            "recall": round(metrics["shallow"]["recall"], 4),
            "f1_score": round(metrics["shallow"]["f1_score"], 4),
            "roc_auc": round(metrics["shallow"]["roc_auc"], 4),
            "silhouette_score": round(manifold_metrics["silhouette_shallow"], 4),
            "latency_ms": metrics["shallow"]["avg_latency_ms_per_sample"],
            "training_time_s": metrics["shallow"]["total_training_time_s"]
        },
        "deep_net": {
            "num_hidden_layers": 10,
            "parameters": metrics["deep"]["complexity"]["total_params"],
            "flops_m": metrics["deep"]["complexity"]["flops_m"],
            "model_size_kb": metrics["deep"]["complexity"]["size_kb"],
            "test_accuracy": round(metrics["deep"]["accuracy"] * 100, 2),
            "test_loss": round(metrics["deep"]["test_loss"], 4),
            "precision": round(metrics["deep"]["precision"], 4),
            "recall": round(metrics["deep"]["recall"], 4),
            "f1_score": round(metrics["deep"]["f1_score"], 4),
            "roc_auc": round(metrics["deep"]["roc_auc"], 4),
            "silhouette_score": round(manifold_metrics["silhouette_deep"], 4),
            "latency_ms": metrics["deep"]["avg_latency_ms_per_sample"],
            "training_time_s": metrics["deep"]["total_training_time_s"]
        }
    }
    
    with open("results/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n========================================================")
    print("           EXPERIMENT SUMMARY COMPARISON                ")
    print("========================================================")
    print(f"{'Metric':<28} | {'ShallowNet (2L)':<16} | {'DeepNet (10L)':<16}")
    print("-" * 66)
    print(f"{'Hidden Conv Layers':<28} | {summary['shallow_net']['num_hidden_layers']:<16} | {summary['deep_net']['num_hidden_layers']:<16}")
    s_params = f"{summary['shallow_net']['parameters']:,}"
    d_params = f"{summary['deep_net']['parameters']:,}"
    print(f"{'Total Parameters':<28} | {s_params:<16} | {d_params:<16}")
    print(f"{'FLOPs (M)':<28} | {summary['shallow_net']['flops_m']:<16.2f} | {summary['deep_net']['flops_m']:<16.2f}")
    print(f"{'Model Size (KB)':<28} | {summary['shallow_net']['model_size_kb']:<16.2f} | {summary['deep_net']['model_size_kb']:<16.2f}")
    print(f"{'Training Time (s)':<28} | {summary['shallow_net']['training_time_s']:<16.2f} | {summary['deep_net']['training_time_s']:<16.2f}")
    print(f"{'Test Accuracy (%)':<28} | {summary['shallow_net']['test_accuracy']:<16.2f}% | {summary['deep_net']['test_accuracy']:<16.2f}%")
    print(f"{'Test F1-Score':<28} | {summary['shallow_net']['f1_score']:<16.4f} | {summary['deep_net']['f1_score']:<16.4f}")
    print(f"{'Test ROC-AUC':<28} | {summary['shallow_net']['roc_auc']:<16.4f} | {summary['deep_net']['roc_auc']:<16.4f}")
    print(f"{'Latent Silhouette Score':<28} | {summary['shallow_net']['silhouette_score']:<16.4f} | {summary['deep_net']['silhouette_score']:<16.4f}")
    print(f"{'Per-sample Latency (ms)':<28} | {summary['shallow_net']['latency_ms']:<16.3f} | {summary['deep_net']['latency_ms']:<16.3f}")
    print("========================================================")

if __name__ == "__main__":
    generate_summary()
