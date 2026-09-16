"""
app.py - Interactive Presentation Web Dashboard Server
Architectural Challenges in Deep Learning: Pathology Model Depth Analysis

Features:
- Live Model Training Runner with real-time SSE streaming (logs, epoch curves, metrics)
- Real-time Interactive Histopathology Patch Diagnosis (ShallowNet vs DeepNet)
- Model Complexity, Latency, and Theoretical Degradation Visualizers
- Fullscreen Presentation Mode for live presentation delivery
"""

import os
import io
import time
import json
import base64
import queue
import threading
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

from models import ShallowPathologyNet, DeepPathologyNet, calculate_model_complexity
from dataset import generate_pathology_patch
import train
import analyze

app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global execution & streaming state
run_lock = threading.Lock()
is_running = False
event_queue = queue.Queue()
active_run_info = {
    "status": "idle",
    "mode": "none",
    "start_time": None,
    "elapsed_s": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_model": "",
    "logs": []
}

# In-memory models for real-time live diagnosis
device = 'cuda' if torch.cuda.is_available() else 'cpu'
shallow_model = ShallowPathologyNet().to(device)
deep_model = DeepPathologyNet().to(device)
models_ready = False

def load_cached_models():
    global models_ready
    shallow_path = os.path.join(RESULTS_DIR, "shallow_net.pth")
    deep_path = os.path.join(RESULTS_DIR, "deep_net.pth")
    try:
        if os.path.exists(shallow_path):
            shallow_model.load_state_dict(torch.load(shallow_path, map_location=device, weights_only=True))
            shallow_model.eval()
        if os.path.exists(deep_path):
            deep_model.load_state_dict(torch.load(deep_path, map_location=device, weights_only=True))
            deep_model.eval()
        models_ready = os.path.exists(shallow_path) and os.path.exists(deep_path)
        print(f"[Dashboard] Cached model weights loaded (Models Ready: {models_ready})")
    except Exception as e:
        print(f"[Dashboard] Model load notification: {e}")

load_cached_models()

# ==============================================================================
# Helper Functions
# ==============================================================================
def patch_to_base64(img_np):
    """Convert float32 (H, W, 3) image in [0, 1] to base64 PNG."""
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def preprocess_patch_for_torch(img_np):
    """Preprocess (64, 64, 3) float32 numpy patch to normalized torch tensor."""
    tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
    mean = torch.tensor([0.70, 0.60, 0.68]).view(3, 1, 1)
    std = torch.tensor([0.20, 0.20, 0.20]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)

# ==============================================================================
# API Endpoints
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/api/status', methods=['GET'])
def get_status():
    shallow_path = os.path.join(RESULTS_DIR, "shallow_net.pth")
    deep_path = os.path.join(RESULTS_DIR, "deep_net.pth")
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    
    summary = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except Exception:
            pass
            
    return jsonify({
        "is_running": is_running,
        "run_info": active_run_info,
        "models_ready": os.path.exists(shallow_path) and os.path.exists(deep_path),
        "device": device,
        "summary": summary
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    test_metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    
    data = {"test_metrics": None, "history": None, "summary": None}
    if os.path.exists(test_metrics_path):
        try:
            with open(test_metrics_path, 'r') as f:
                data["test_metrics"] = json.load(f)
        except Exception:
            pass
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                data["history"] = json.load(f)
        except Exception:
            pass
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                data["summary"] = json.load(f)
        except Exception:
            pass
            
    return jsonify(data)

@app.route('/api/diagnose', methods=['POST'])
def run_live_diagnose():
    """
    Generate or process an H&E tissue patch, run dual inference (ShallowNet vs DeepNet),
    measure per-model inference latencies, and return predictions & biological morphology.
    """
    req_data = request.get_json() or {}
    label_requested = req_data.get('label')  # 0: Benign, 1: Malignant, or None for random
    
    if label_requested is not None and label_requested in [0, 1]:
        true_label = int(label_requested)
    else:
        true_label = int(np.random.choice([0, 1]))
        
    # Synthesize patch
    patch_img = generate_pathology_patch(true_label, size=64)
    patch_b64 = patch_to_base64(patch_img)
    tensor = preprocess_patch_for_torch(patch_img)
    
    # 1. ShallowNet Inference
    shallow_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        shallow_out = shallow_model(tensor)
        t1 = time.perf_counter()
        shallow_latency_ms = (t1 - t0) * 1000.0
        shallow_probs = F.softmax(shallow_out, dim=1)[0].cpu().numpy()
        shallow_pred = int(torch.argmax(shallow_out, dim=1).item())
        
    # 2. DeepNet Inference
    deep_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        deep_out = deep_model(tensor)
        t1 = time.perf_counter()
        deep_latency_ms = (t1 - t0) * 1000.0
        deep_probs = F.softmax(deep_out, dim=1)[0].cpu().numpy()
        deep_pred = int(torch.argmax(deep_out, dim=1).item())
        
    morphology = {
        0: {
            "title": "Benign Prostatic / Epithelial Tissue",
            "lumens": "Organized glandular circular clearings intact",
            "nuclei": "Small, uniform basal chromatin; low pleomorphism",
            "nc_ratio": "Normal Low Nuclear-to-Cytoplasmic ratio (~1:4)",
            "risk_assessment": "Negative for high-grade malignancy"
        },
        1: {
            "title": "Malignant Invasive Carcinoma",
            "lumens": "Complete loss of glandular architecture and lumen clearance",
            "nuclei": "Severe nuclear pleomorphism, hyperchromasia, chromatin clumping",
            "nc_ratio": "Significantly elevated Nuclear-to-Cytoplasmic ratio (~1:1)",
            "risk_assessment": "High-grade invasive neoplastic proliferation"
        }
    }
    
    return jsonify({
        "true_label": true_label,
        "true_class_name": "Benign" if true_label == 0 else "Malignant",
        "image_b64": patch_b64,
        "morphology": morphology[true_label],
        "shallow": {
            "predicted_label": shallow_pred,
            "predicted_class_name": "Benign" if shallow_pred == 0 else "Malignant",
            "confidence": float(shallow_probs[shallow_pred]),
            "prob_benign": float(shallow_probs[0]),
            "prob_malignant": float(shallow_probs[1]),
            "latency_ms": round(shallow_latency_ms, 3),
            "is_correct": shallow_pred == true_label
        },
        "deep": {
            "predicted_label": deep_pred,
            "predicted_class_name": "Benign" if deep_pred == 0 else "Malignant",
            "confidence": float(deep_probs[deep_pred]),
            "prob_benign": float(deep_probs[0]),
            "prob_malignant": float(deep_probs[1]),
            "latency_ms": round(deep_latency_ms, 3),
            "is_correct": deep_pred == true_label
        },
        "latency_speedup": round(deep_latency_ms / (shallow_latency_ms + 1e-6), 2)
    })

# ==============================================================================
# Live Execution Engine & SSE Streaming
# ==============================================================================
def background_training_worker(num_samples, epochs, batch_size, lr, mode_name):
    global is_running, active_run_info
    
    def log_handler(msg):
        msg_str = str(msg)
        active_run_info["logs"].append(msg_str)
        if len(active_run_info["logs"]) > 200:
            active_run_info["logs"].pop(0)
        event_queue.put({"type": "log", "message": msg_str, "timestamp": time.strftime("%H:%M:%S")})
        
    def progress_handler(data):
        active_run_info["current_epoch"] = data["epoch"]
        active_run_info["total_epochs"] = data["total_epochs"]
        active_run_info["current_model"] = data["model"]
        event_queue.put({"type": "epoch_progress", "data": data})

    start_time = time.time()
    active_run_info["status"] = "running"
    active_run_info["mode"] = mode_name
    active_run_info["start_time"] = start_time
    active_run_info["current_epoch"] = 0
    active_run_info["total_epochs"] = epochs
    active_run_info["logs"] = []
    
    event_queue.put({
        "type": "start",
        "mode": mode_name,
        "epochs": epochs,
        "samples": num_samples,
        "lr": lr,
        "batch_size": batch_size
    })
    
    try:
        log_handler(f"Starting experiment '{mode_name}'...")
        log_handler(f"Parameters: Samples={num_samples}, Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
        
        histories, metrics = train.run_experiment(
            num_samples=num_samples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            progress_callback=progress_handler,
            log_callback=log_handler
        )
        
        load_cached_models()
        total_time = round(time.time() - start_time, 2)
        active_run_info["status"] = "completed"
        active_run_info["elapsed_s"] = total_time
        
        event_queue.put({
            "type": "complete",
            "elapsed_s": total_time,
            "metrics": metrics
        })
        log_handler(f"Experiment finished successfully in {total_time}s!")
        
    except Exception as e:
        err_msg = f"Experiment encountered an error: {str(e)}"
        active_run_info["status"] = "error"
        log_handler(err_msg)
        event_queue.put({"type": "error", "error": err_msg})
        
    finally:
        with run_lock:
            is_running = False

@app.route('/api/run', methods=['POST'])
def run_program():
    global is_running
    with run_lock:
        if is_running:
            return jsonify({"status": "error", "message": "An experiment is already in progress."}), 409
        is_running = True
        
    req = request.get_json() or {}
    mode = req.get('mode', 'quick')
    
    if mode == 'quick':
        # Fast 10-15s presentation demo
        epochs = int(req.get('epochs', 5))
        samples = int(req.get('samples', 400))
        lr = float(req.get('lr', 0.02))
        batch_size = int(req.get('batch_size', 32))
        mode_name = "Presentation Quick Demo (5 Epochs)"
    elif mode == 'standard':
        epochs = int(req.get('epochs', 15))
        samples = int(req.get('samples', 1000))
        lr = float(req.get('lr', 0.015))
        batch_size = int(req.get('batch_size', 32))
        mode_name = "Standard Benchmark Run (15 Epochs)"
    elif mode == 'full':
        epochs = int(req.get('epochs', 25))
        samples = int(req.get('samples', 1600))
        lr = float(req.get('lr', 0.01))
        batch_size = int(req.get('batch_size', 32))
        mode_name = "Full Depth Degradation Reproduction (25 Epochs)"
    else:
        epochs = max(1, min(50, int(req.get('epochs', 10))))
        samples = max(100, min(3000, int(req.get('samples', 600))))
        lr = max(0.0001, min(0.5, float(req.get('lr', 0.015))))
        batch_size = int(req.get('batch_size', 32))
        mode_name = f"Custom Experiment ({epochs} Epochs, {samples} Samples)"
        
    worker_thread = threading.Thread(
        target=background_training_worker,
        args=(samples, epochs, batch_size, lr, mode_name),
        daemon=True
    )
    worker_thread.start()
    
    return jsonify({
        "status": "started",
        "mode": mode_name,
        "epochs": epochs,
        "samples": samples,
        "lr": lr,
        "batch_size": batch_size
    })

@app.route('/api/run/stream')
def stream_execution():
    """Server-Sent Events endpoint streaming real-time experiment events."""
    def event_stream():
        # Send initial status
        yield f"data: {json.dumps({'type': 'init', 'status': active_run_info['status'], 'logs': active_run_info['logs'][-20:]})}\n\n"
        while True:
            try:
                msg = event_queue.get(timeout=1.0)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ["complete", "error"] and not is_running:
                    break
            except queue.Empty:
                # Send heartbeat keep-alive
                yield f": heartbeat\n\n"
                if not is_running and active_run_info["status"] in ["completed", "error", "idle"]:
                    # No active run and queue drained
                    pass
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" Pathology Model Depth Analysis - Presentation Dashboard")
    print(f" Server running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
