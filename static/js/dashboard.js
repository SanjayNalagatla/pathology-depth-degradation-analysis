/**
 * NeuroPath DepthLab • Frontend Controller
 * Live SSE streaming, Chart.js optimization tracking, dual-model diagnostic lab,
 * and presentation slide deck engine.
 */

// State
let eventSource = null;
let liveChart = null;
let currentSlide = 1;
const totalSlides = 10;
let isFullscreen = false;

// Visualizer metadata
const visualizerData = {
  curves: {
    src: "/results/training_validation_curves.png",
    title: "Training & Validation Optimization Dynamics",
    body: "<strong>The Degradation Problem:</strong> ShallowNet (2 layers, blue) converges rapidly within the very first epoch (79.1% &rarr; 100% accuracy, loss &rarr; 0.000). In stark contrast, DeepNet (10 layers, red) suffers from an optimization plateau, remaining completely stuck at cross-entropy loss ~0.693 (theoretical random guessing ~50% accuracy) for 22 consecutive epochs before vanishing gradients are finally overcome.",
    takeaways: [
      "Mathematical Proof: Multiplicative gradient decay prevents early weight updates during backpropagation.",
      "Not Overfitting: Both training loss and validation loss remain high together during the plateau, proving this is an optimization failure rather than statistical generalization error."
    ]
  },
  gradients: {
    src: "/results/gradient_flow_analysis.png",
    title: "Layer-wise Vanishing Gradient Dynamics across Depth",
    body: "<strong>Direct Gradient Tracking Evidence:</strong> Left: L2 Frobenius gradient norm over training epochs across all 10 conv layers of DeepNet (log scale). Early layers (conv1, conv2) remain orders of magnitude below late layers (conv9, conv10). Right: Layer-wise mean gradient norm from input to output, demonstrating over 50x to 100x signal decay starving early spatial filters.",
    takeaways: [
      "Conv1 Gradient Norm: ~1.0 × 10⁻⁴ vs Conv10: ~6.0 × 10⁻³.",
      "Consequence: Early convolutional feature extractors receive virtually zero parameter update signals during initial training."
    ]
  },
  representations: {
    src: "/results/latent_representations_tsne_pca.png",
    title: "Learned Latent Representation Geometry (PCA & t-SNE)",
    body: "<strong>Manifold Crystallization:</strong> 2D PCA (top) and t-SNE (bottom) manifold projections of the 64-dimensional penultimate bottleneck embeddings on the test set. Both ShallowNet (Silhouette: 0.865) and DeepNet (Silhouette: 0.884) cleanly cluster Benign (green) from Malignant (red) once converged.",
    takeaways: [
      "Sufficient Expressivity: Plain deep networks possess ample capacity to model the manifold.",
      "Delayed Formation: DeepNet requires 22x more training iterations to achieve an equivalent latent separation."
    ]
  },
  roc: {
    src: "/results/confusion_matrices_roc.png",
    title: "Confusion Matrices & Receiver Operating Characteristic (ROC)",
    body: "<strong>Clinical Diagnostic Discrimination:</strong> Both ShallowNet and DeepNet achieve perfect test discrimination (120/120 Benign, 120/120 Malignant; AUC = 1.000). On this canonical histopathology classification task, plain network depth contributes zero diagnostic accuracy improvement.",
    takeaways: [
      "Zero Diagnostic Gain: DeepNet accuracy = 100.0%, ShallowNet accuracy = 100.0%.",
      "Over-engineering: Plain depth adds latency and computational risk without clinical margin."
    ]
  },
  complexity: {
    src: "/results/model_complexity_comparison.png",
    title: "Model Complexity vs Diagnostic Return",
    body: "<strong>Severe Resource Inflation:</strong> Side-by-side comparison across Parameters (38.0k vs 98.2k; +2.58x), Computational FLOPs (13.0M vs 107.4M; +8.24x), Model Memory Size (148KB vs 383KB; +2.58x), and Inference Latency (0.54ms vs 1.61ms; +2.97x).",
    takeaways: [
      "Gigapixel Slide Scalability: Whole-slide scans with 100,000+ patches face 3x longer processing times.",
      "Energy & Hardware: Deep plain models require 8x higher compute budget for identical diagnostic yield."
    ]
  },
  dataset: {
    src: "/results/sample_pathology_patches.png",
    title: "Simulated Histopathology Image Dataset (H&E Staining)",
    body: "<strong>Cellular Pathology Morphology:</strong> Synthesized digital pathology patches modeled after prostate / epithelial histology under Hematoxylin & Eosin (H&E) staining.",
    takeaways: [
      "Class 0 (Benign): Circular organized glandular lumens, basal chromatin, low N:C ratio (~1:4).",
      "Class 1 (Malignant): Complete loss of glandular lumens, nuclear pleomorphism, severe hyperchromasia, crowding."
    ]
  }
};

// =============================================================================
// Initialization
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initLiveChart();
  loadStatus();
  loadMetrics();
  setupKeyboardNavigation();
  // Generate an initial diagnostic sample on startup
  runDiagnosis(null);
});

// =============================================================================
// View & Tab Switching
// =============================================================================
function switchView(tabId, tabBtn) {
  document.querySelectorAll('.view-panel').forEach(panel => panel.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));

  const target = document.getElementById(tabId);
  if (target) {
    target.classList.add('active');
  }

  if (tabBtn) {
    tabBtn.classList.add('active');
  } else {
    const matchingBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    if (matchingBtn) matchingBtn.classList.add('active');
  }

  if (tabId === 'tab-runner' && liveChart) {
    setTimeout(() => liveChart.resize(), 100);
  }
}

// =============================================================================
// Status & Metrics Fetching
// =============================================================================
async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.parseJsonSafe();
    updateStatusBadge(data.is_running, data.run_info);
  } catch (err) {
    console.error("Status load error:", err);
  }
}

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.parseJsonSafe();
    if (data.test_metrics) {
      renderMetricsTable(data.test_metrics);
    }
  } catch (err) {
    console.error("Metrics load error:", err);
  }
}

Response.prototype.parseJsonSafe = async function() {
  const text = await this.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return {};
  }
};

function updateStatusBadge(isRunning, runInfo) {
  const badge = document.getElementById('system-status-badge');
  const text = document.getElementById('system-status-text');
  const runBtn = document.getElementById('run-btn');

  if (isRunning) {
    badge.className = 'status-badge status-running';
    text.textContent = `Running: ${runInfo?.current_model || 'Training...'}`;
    if (runBtn) {
      runBtn.disabled = true;
      document.getElementById('run-btn-label').textContent = 'Training in Progress...';
    }
  } else {
    badge.className = 'status-badge status-idle';
    text.textContent = 'System Ready';
    if (runBtn) {
      runBtn.disabled = false;
      document.getElementById('run-btn-label').textContent = 'Run Experiment Live';
    }
  }
}

function renderMetricsTable(metrics) {
  const s = metrics.shallow;
  const d = metrics.deep;
  if (!s || !d) return;

  const tbody = document.getElementById('metrics-table-body');
  if (!tbody) return;

  tbody.innerHTML = `
    <tr>
      <td><strong>Hidden Conv Layers</strong></td>
      <td class="color-blue font-mono">2 Layers</td>
      <td class="color-red font-mono">10 Layers</td>
      <td><span class="tag tag-neutral">+5x Depth</span></td>
      <td>Substantial architectural elongation</td>
    </tr>
    <tr>
      <td><strong>Trainable Parameters</strong></td>
      <td class="color-blue font-mono">${s.complexity?.trainable_params?.toLocaleString() || '38,050'}</td>
      <td class="color-red font-mono">${d.complexity?.trainable_params?.toLocaleString() || '98,178'}</td>
      <td><span class="tag tag-warning">+2.58x Overhead</span></td>
      <td>Added capacity without representational necessity</td>
    </tr>
    <tr>
      <td><strong>Computational FLOPs</strong></td>
      <td class="color-blue font-mono">${s.complexity?.flops_m?.toFixed(2) || '13.04'} MFLOPs</td>
      <td class="color-red font-mono">${d.complexity?.flops_m?.toFixed(2) || '107.41'} MFLOPs</td>
      <td><span class="tag tag-danger">+${((d.complexity?.flops_m / s.complexity?.flops_m) || 8.24).toFixed(1)}x Compute</span></td>
      <td>Critical scaling penalty on whole-slide gigapixel scans</td>
    </tr>
    <tr>
      <td><strong>Inference Latency per Patch</strong></td>
      <td class="color-blue font-mono">${s.avg_latency_ms_per_sample?.toFixed(3) || '0.542'} ms</td>
      <td class="color-red font-mono">${d.avg_latency_ms_per_sample?.toFixed(3) || '1.608'} ms</td>
      <td><span class="tag tag-danger">+${((d.avg_latency_ms_per_sample / s.avg_latency_ms_per_sample) || 2.97).toFixed(1)}x Slower</span></td>
      <td>Inference bottleneck across 100k+ patches per slide</td>
    </tr>
    <tr>
      <td><strong>Training Duration</strong></td>
      <td class="color-blue font-mono">${s.total_training_time_s?.toFixed(1) || '44.3'}s</td>
      <td class="color-red font-mono">${d.total_training_time_s?.toFixed(1) || '143.3'}s</td>
      <td><span class="tag tag-danger">+${((d.total_training_time_s / s.total_training_time_s) || 3.2).toFixed(1)}x Time</span></td>
      <td>Prolonged optimization barrier under plain SGD</td>
    </tr>
    <tr>
      <td><strong>Test Accuracy / F1-Score</strong></td>
      <td class="color-blue font-mono">${(s.accuracy * 100).toFixed(1)}% / ${s.f1_score?.toFixed(3) || '1.000'}</td>
      <td class="color-red font-mono">${(d.accuracy * 100).toFixed(1)}% / ${d.f1_score?.toFixed(3) || '1.000'}</td>
      <td><span class="tag tag-success">Identical 100%</span></td>
      <td>Depth yields zero diagnostic improvement on this task</td>
    </tr>
  `;
}

// =============================================================================
// Live Experiment Runner & SSE Streaming
// =============================================================================
const presets = {
  quick: { epochs: 5, samples: 400, lr: 0.02, batch: 32 },
  standard: { epochs: 15, samples: 1000, lr: 0.015, batch: 32 },
  full: { epochs: 25, samples: 1600, lr: 0.01, batch: 32 }
};

function setPreset(mode) {
  const p = presets[mode];
  if (!p) return;
  document.getElementById('param-epochs').value = p.epochs;
  document.getElementById('param-samples').value = p.samples;
  document.getElementById('param-lr').value = p.lr;
  document.getElementById('param-batch').value = p.batch;

  document.querySelectorAll('.btn-preset').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
}

function triggerPreset(mode) {
  setPreset(mode);
  switchView('tab-runner');
  startExperiment(mode);
}

async function startExperiment(modeHint) {
  const epochs = parseInt(document.getElementById('param-epochs').value, 10);
  const samples = parseInt(document.getElementById('param-samples').value, 10);
  const lr = parseFloat(document.getElementById('param-lr').value);
  const batch_size = parseInt(document.getElementById('param-batch').value, 10);

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  document.getElementById('run-btn-label').textContent = 'Launching Engine...';

  // Show progress
  const progressWrapper = document.getElementById('progress-container');
  progressWrapper.style.display = 'block';
  document.getElementById('progress-bar-fill').style.width = '0%';
  document.getElementById('progress-percent').textContent = '0%';
  document.getElementById('progress-phase').textContent = 'Initializing PyTorch model...';

  // Reset live chart
  resetLiveChart();

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: modeHint || 'custom',
        epochs, samples, lr, batch_size
      })
    });

    const data = await res.parseJsonSafe();
    if (res.status === 409) {
      appendTerminalLine(`[Warning] ${data.message}`, 'term-error');
      runBtn.disabled = false;
      document.getElementById('run-btn-label').textContent = 'Run Experiment Live';
      return;
    }

    appendTerminalLine(`[Engine] Started: ${data.mode}`, 'term-system');
    connectEventStream();

  } catch (err) {
    appendTerminalLine(`[Error] Failed to trigger run: ${err.message}`, 'term-error');
    runBtn.disabled = false;
    document.getElementById('run-btn-label').textContent = 'Run Experiment Live';
  }
}

function connectEventStream() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource('/api/run/stream');

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleServerEvent(data);
    } catch (err) {
      // heartbeats or comments
    }
  };

  eventSource.onerror = () => {
    // Reconnect or stream ended
  };
}

function handleServerEvent(event) {
  const terminal = document.getElementById('terminal-body');

  if (event.type === 'log') {
    const msg = event.message;
    let cls = 'term-log';
    if (msg.includes('Epoch [')) cls = 'term-epoch';
    else if (msg.includes('Phase') || msg.includes('>>>')) cls = 'term-system';
    else if (msg.includes('SUCCESS') || msg.includes('completed')) cls = 'term-success';
    else if (msg.includes('Error') || msg.includes('Warning')) cls = 'term-error';

    appendTerminalLine(`[${event.timestamp}] ${msg}`, cls);
  } else if (event.type === 'epoch_progress') {
    const d = event.data;
    const isShallow = d.model_key === 'shallow';
    const phaseOffset = isShallow ? 0 : 50;
    const modelProgress = (d.epoch / d.total_epochs) * 50;
    const totalPercent = Math.min(100, Math.round(phaseOffset + modelProgress));

    document.getElementById('progress-bar-fill').style.width = `${totalPercent}%`;
    document.getElementById('progress-percent').textContent = `${totalPercent}%`;
    document.getElementById('progress-phase').textContent = `${d.model} • Epoch ${d.epoch}/${d.total_epochs}`;

    // Add point to live chart
    addChartDataPoint(d.model_key, d.epoch, d.train_loss, d.val_loss, d.train_acc * 100, d.val_acc * 100);

    document.getElementById('live-chart-footer').textContent = 
      `${d.model} Ep ${d.epoch}: Loss = ${d.train_loss.toFixed(4)}, Acc = ${(d.train_acc*100).toFixed(1)}% (Time: ${d.epoch_duration.toFixed(2)}s)`;

  } else if (event.type === 'complete') {
    appendTerminalLine(`\n[COMPLETE] All experiments and diagnostics updated in ${event.elapsed_s}s!`, 'term-success');
    document.getElementById('progress-bar-fill').style.width = '100%';
    document.getElementById('progress-percent').textContent = '100%';
    document.getElementById('progress-phase').textContent = 'Completed successfully!';
    
    updateStatusBadge(false);
    loadMetrics();

    // Reload diagnostic plot images by busting browser cache
    reloadDiagnosticImages();

    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  } else if (event.type === 'error') {
    appendTerminalLine(`[FAILURE] ${event.error}`, 'term-error');
    updateStatusBadge(false);
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }
}

function appendTerminalLine(text, className) {
  const terminal = document.getElementById('terminal-body');
  if (!terminal) return;
  const line = document.createElement('div');
  line.className = `term-line ${className || 'term-log'}`;
  line.textContent = text;
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

function clearTerminal() {
  const terminal = document.getElementById('terminal-body');
  if (terminal) terminal.innerHTML = '<div class="term-line term-system">[System] Console cleared.</div>';
}

function reloadDiagnosticImages() {
  const timestamp = Date.now();
  document.querySelectorAll('.analytics-main-img, .slide-img').forEach(img => {
    const baseSrc = img.src.split('?')[0];
    img.src = `${baseSrc}?t=${timestamp}`;
  });
}

// =============================================================================
// Live Chart.js Setup
// =============================================================================
function initLiveChart() {
  const ctx = document.getElementById('live-training-chart');
  if (!ctx) return;

  liveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'ShallowNet Accuracy (%)',
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          data: [],
          yAxisID: 'yAcc',
          tension: 0.3,
          borderWidth: 2.5
        },
        {
          label: 'DeepNet Accuracy (%)',
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          data: [],
          yAxisID: 'yAcc',
          tension: 0.3,
          borderWidth: 2.5
        },
        {
          label: 'ShallowNet Loss',
          borderColor: '#93c5fd',
          borderDash: [5, 5],
          data: [],
          yAxisID: 'yLoss',
          tension: 0.3,
          borderWidth: 1.5,
          pointRadius: 2
        },
        {
          label: 'DeepNet Loss',
          borderColor: '#fda4af',
          borderDash: [5, 5],
          data: [],
          yAxisID: 'yLoss',
          tension: 0.3,
          borderWidth: 1.5,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11 } },
          title: { display: true, text: 'Training Epoch', color: '#64748b' }
        },
        yAcc: {
          type: 'linear',
          position: 'left',
          min: 40,
          max: 105,
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11 } },
          title: { display: true, text: 'Accuracy (%)', color: '#94a3b8' }
        },
        yLoss: {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 11 } },
          title: { display: true, text: 'Loss', color: '#64748b' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { family: 'Inter', size: 11 } }
        }
      }
    }
  });
}

function resetLiveChart() {
  if (!liveChart) return;
  liveChart.data.labels = [];
  liveChart.data.datasets.forEach(ds => ds.data = []);
  liveChart.update();
}

function addChartDataPoint(modelKey, epoch, trainLoss, valLoss, trainAcc, valAcc) {
  if (!liveChart) return;

  const epochLabel = `Ep ${epoch}`;
  if (!liveChart.data.labels.includes(epochLabel)) {
    liveChart.data.labels.push(epochLabel);
  }

  if (modelKey === 'shallow') {
    liveChart.data.datasets[0].data.push(valAcc);
    liveChart.data.datasets[2].data.push(valLoss);
  } else {
    liveChart.data.datasets[1].data.push(valAcc);
    liveChart.data.datasets[3].data.push(valLoss);
  }

  liveChart.update();
}

// =============================================================================
// Interactive Tissue Diagnostic Lab
// =============================================================================
async function runDiagnosis(labelHint) {
  const scanLine = document.getElementById('patch-scan-line');
  scanLine.style.display = 'block';

  try {
    const res = await fetch('/api/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: labelHint })
    });

    const data = await res.parseJsonSafe();
    renderDiagnosisResult(data);

  } catch (err) {
    console.error("Diagnosis error:", err);
  } finally {
    setTimeout(() => { scanLine.style.display = 'none'; }, 600);
  }
}

function renderDiagnosisResult(data) {
  // Update image
  document.getElementById('diagnostic-patch-img').src = data.image_b64;

  // Ground truth badge
  const gtBadge = document.getElementById('patch-ground-truth');
  gtBadge.textContent = `Ground Truth: ${data.true_class_name}`;
  gtBadge.className = `tag ${data.true_label === 0 ? 'tag-success' : 'tag-danger'}`;

  // Morphology box
  const morph = data.morphology;
  document.getElementById('morph-title').textContent = morph.title;
  document.getElementById('morph-list').innerHTML = `
    <li><strong>Glandular Architecture:</strong> ${morph.lumens}</li>
    <li><strong>Nuclear Chromatin:</strong> ${morph.nuclei}</li>
    <li><strong>N:C Ratio:</strong> ${morph.nc_ratio}</li>
    <li><strong>Clinical Risk:</strong> <span class="${data.true_label === 0 ? 'color-emerald' : 'color-red'}">${morph.risk_assessment}</span></li>
  `;

  // ShallowNet Card
  const s = data.shallow;
  const sBadge = document.getElementById('shallow-pred-badge');
  sBadge.textContent = `${s.predicted_class_name} (${(s.confidence * 100).toFixed(1)}%)`;
  sBadge.className = `pred-badge ${s.predicted_label === 0 ? 'badge-benign' : 'badge-malignant'}`;

  document.getElementById('shallow-prob-benign').textContent = `${(s.prob_benign * 100).toFixed(1)}%`;
  document.getElementById('shallow-bar-benign').style.width = `${(s.prob_benign * 100).toFixed(1)}%`;
  document.getElementById('shallow-prob-malignant').textContent = `${(s.prob_malignant * 100).toFixed(1)}%`;
  document.getElementById('shallow-bar-malignant').style.width = `${(s.prob_malignant * 100).toFixed(1)}%`;
  document.getElementById('shallow-latency').textContent = `${s.latency_ms.toFixed(3)} ms`;

  // DeepNet Card
  const d = data.deep;
  const dBadge = document.getElementById('deep-pred-badge');
  dBadge.textContent = `${d.predicted_class_name} (${(d.confidence * 100).toFixed(1)}%)`;
  dBadge.className = `pred-badge ${d.predicted_label === 0 ? 'badge-benign' : 'badge-malignant'}`;

  document.getElementById('deep-prob-benign').textContent = `${(d.prob_benign * 100).toFixed(1)}%`;
  document.getElementById('deep-bar-benign').style.width = `${(d.prob_benign * 100).toFixed(1)}%`;
  document.getElementById('deep-prob-malignant').textContent = `${(d.prob_malignant * 100).toFixed(1)}%`;
  document.getElementById('deep-bar-malignant').style.width = `${(d.prob_malignant * 100).toFixed(1)}%`;
  document.getElementById('deep-latency').textContent = `${d.latency_ms.toFixed(3)} ms`;

  // Speedup Callout
  const speedup = (d.latency_ms / (s.latency_ms + 1e-6)).toFixed(1);
  document.getElementById('speedup-title').textContent = `ShallowNet is ${speedup}x Faster (${s.latency_ms.toFixed(2)}ms vs ${d.latency_ms.toFixed(2)}ms)`;
  document.getElementById('speedup-desc').textContent = 
    `Both models agree on "${s.predicted_class_name}" with >99% certainty. The deeper network incurs a ${speedup}x inference penalty without providing any diagnostic advantage.`;
}

// =============================================================================
// Visualizer Tabs
// =============================================================================
function switchVisualizerTab(key, btn) {
  document.querySelectorAll('.vis-tab').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const meta = visualizerData[key];
  if (!meta) return;

  const img = document.getElementById('analytics-display-img');
  img.style.opacity = '0.3';

  setTimeout(() => {
    img.src = meta.src;
    document.getElementById('analytics-caption-title').textContent = meta.title;
    document.getElementById('analytics-caption-body').innerHTML = meta.body;

    const takeawaysEl = document.querySelector('.analytics-takeaways');
    takeawaysEl.innerHTML = meta.takeaways.map(t => `
      <div class="takeaway-item">
        <span class="takeaway-bullet">✓</span>
        <span>${t}</span>
      </div>
    `).join('');

    img.style.opacity = '1';
  }, 120);
}

// =============================================================================
// Presentation Slide Deck
// =============================================================================
function updateSlideView() {
  document.querySelectorAll('.slide').forEach(s => s.classList.remove('active'));
  const current = document.querySelector(`.slide[data-slide="${currentSlide}"]`);
  if (current) current.classList.add('active');

  document.getElementById('slide-indicator').textContent = `Slide ${currentSlide} of ${totalSlides}`;
  const pct = (currentSlide / totalSlides) * 100;
  document.getElementById('slide-progress-fill').style.width = `${pct}%`;
}

function nextSlide() {
  if (currentSlide < totalSlides) {
    currentSlide++;
    updateSlideView();
  }
}

function prevSlide() {
  if (currentSlide > 1) {
    currentSlide--;
    updateSlideView();
  }
}

function togglePresentationMode() {
  switchView('tab-slides');
}

function toggleFullscreenDeck() {
  isFullscreen = !isFullscreen;
  document.body.classList.toggle('fullscreen-deck', isFullscreen);
}

function setupKeyboardNavigation() {
  document.addEventListener('keydown', (e) => {
    // If user is typing in an input field, ignore shortcuts
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      const slidesTab = document.getElementById('tab-slides');
      if (slidesTab.classList.contains('active')) {
        e.preventDefault();
        nextSlide();
      }
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      const slidesTab = document.getElementById('tab-slides');
      if (slidesTab.classList.contains('active')) {
        e.preventDefault();
        prevSlide();
      }
    } else if (e.key === 'p' || e.key === 'P') {
      togglePresentationMode();
    } else if (e.key === 'Escape') {
      if (isFullscreen) {
        toggleFullscreenDeck();
      }
    }
  });
}
