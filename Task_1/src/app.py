"""
Module: app.py
Author: Devesh Kumawat
Purpose: FastAPI web application for the Iris Flower Classifier.
         Serves a premium dark-mode glassmorphic interactive dashboard with
         real-time PCA cluster visualization and live predictions via sliders.
Date: 2026-07-06
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Configure UTF-8 stdout ────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Load model pipeline ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Please run 'python src/train.py' first."
    )

pipeline = joblib.load(MODEL_PATH)
MODEL_NAME   = pipeline["model_name"]
MODEL        = pipeline["model"]
SCALER       = pipeline["scaler"]
ENCODER      = pipeline["label_encoder"]
FEATURES     = pipeline["feature_names"]
CLASSES      = pipeline["class_names"]
PCA          = pipeline["pca"]
PCA_CLUSTERS = pipeline["pca_clusters"]
PCA_EXPL     = pipeline["pca_explained"]

# ── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="Iris Flower Classifier", version="2.0.0")


class FlowerInput(BaseModel):
    """Input schema for flower measurement predictions."""
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float


@app.get("/api/model-info")
def model_info():
    """Return model metadata and PCA cluster data for the frontend chart."""
    return JSONResponse({
        "model_name":    MODEL_NAME,
        "accuracy":      round(pipeline["performance"]["accuracy"] * 100, 2),
        "f1_score":      round(pipeline["performance"]["f1_score"]  * 100, 2),
        "pca_clusters":  PCA_CLUSTERS,
        "pca_explained": PCA_EXPL,
        "classes":       CLASSES,
    })


@app.post("/api/predict")
def predict(flower: FlowerInput):
    """
    Accepts flower measurements, returns predicted species,
    class probabilities, and projected PCA (x, y) coordinates.
    """
    input_vals = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]]
    df = pd.DataFrame(input_vals, columns=FEATURES)
    scaled = SCALER.transform(df)

    pred_code  = MODEL.predict(scaled)[0]
    pred_class = ENCODER.inverse_transform([pred_code])[0]

    probs = {}
    confidence = 1.0
    if hasattr(MODEL, "predict_proba"):
        raw_probs  = MODEL.predict_proba(scaled)[0]
        confidence = float(raw_probs[pred_code])
        probs = {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, raw_probs)}

    # Project into PCA space for chart dot
    pca_point = PCA.transform(scaled)[0]

    return JSONResponse({
        "predicted_class": pred_class,
        "confidence":      round(confidence * 100, 2),
        "probabilities":   probs,
        "pca_x":           round(float(pca_point[0]), 4),
        "pca_y":           round(float(pca_point[1]), 4),
    })


# ── HTML Dashboard ────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🌸 Iris Flower Classifier</title>
<meta name="description" content="Interactive Iris Flower Classification Dashboard with real-time PCA visualization"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:         #07071a;
    --surface:    rgba(255,255,255,0.04);
    --border:     rgba(255,255,255,0.08);
    --accent1:    #7c3aed;
    --accent2:    #06b6d4;
    --accent3:    #f472b6;
    --text:       #e2e8f0;
    --text-dim:   #94a3b8;
    --c0:         #00f5d4;
    --c1:         #f72585;
    --c2:         #f8961e;
    --radius:     18px;
    --glow:       0 0 40px rgba(124,58,237,0.25);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{
    font-family:'Outfit',sans-serif;
    background:var(--bg);
    color:var(--text);
    min-height:100vh;
    background-image:
      radial-gradient(ellipse at 20% 0%,  rgba(124,58,237,0.15) 0%,transparent 60%),
      radial-gradient(ellipse at 80% 100%,rgba(6,182,212,0.12)  0%,transparent 60%);
  }

  /* ── Header ── */
  header{
    display:flex;align-items:center;gap:16px;
    padding:28px 40px;
    border-bottom:1px solid var(--border);
    backdrop-filter:blur(10px);
    position:sticky;top:0;z-index:100;
    background:rgba(7,7,26,0.75);
  }
  header h1{font-size:1.5rem;font-weight:700;letter-spacing:-0.5px}
  header h1 span{
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent
  }
  .badge{
    margin-left:auto;
    font-family:'JetBrains Mono',monospace;
    font-size:0.7rem;
    padding:4px 12px;
    border-radius:20px;
    background:rgba(124,58,237,0.2);
    border:1px solid rgba(124,58,237,0.4);
    color:#a78bfa;
  }

  /* ── Layout ── */
  .layout{display:grid;grid-template-columns:380px 1fr;gap:24px;padding:32px 40px;max-width:1400px;margin:0 auto}
  @media(max-width:900px){.layout{grid-template-columns:1fr}}

  /* ── Cards ── */
  .card{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:28px;
    backdrop-filter:blur(12px);
    box-shadow:var(--glow);
    transition:box-shadow 0.3s;
  }
  .card:hover{box-shadow:0 0 60px rgba(124,58,237,0.35)}
  .card-title{
    font-size:0.75rem;font-weight:600;letter-spacing:1.5px;
    text-transform:uppercase;color:var(--text-dim);margin-bottom:20px;
  }

  /* ── Sliders ── */
  .slider-group{margin-bottom:22px}
  .slider-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .slider-label{font-size:0.88rem;font-weight:500;color:var(--text)}
  .slider-val{
    font-family:'JetBrains Mono',monospace;font-size:0.82rem;
    color:var(--accent2);background:rgba(6,182,212,0.1);
    padding:2px 8px;border-radius:6px;border:1px solid rgba(6,182,212,0.25);
    min-width:48px;text-align:center;
  }
  input[type=range]{
    width:100%;-webkit-appearance:none;appearance:none;
    height:5px;border-radius:4px;outline:none;cursor:pointer;
    background:linear-gradient(to right, var(--accent1), var(--accent2));
  }
  input[type=range]::-webkit-slider-thumb{
    -webkit-appearance:none;width:18px;height:18px;
    border-radius:50%;background:#fff;
    box-shadow:0 0 8px rgba(124,58,237,0.7);
    transition:transform 0.15s;
  }
  input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2)}

  /* ── Predict button ── */
  #predictBtn{
    width:100%;padding:14px;border:none;border-radius:12px;
    font-family:'Outfit',sans-serif;font-size:1rem;font-weight:600;
    cursor:pointer;letter-spacing:0.5px;
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    color:#fff;
    box-shadow:0 4px 24px rgba(124,58,237,0.4);
    transition:transform 0.18s,box-shadow 0.18s;
    margin-top:8px;
  }
  #predictBtn:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(124,58,237,0.6)}
  #predictBtn:active{transform:translateY(0)}
  #predictBtn.loading{opacity:0.7;pointer-events:none}

  /* ── Result card ── */
  #resultCard{
    margin-top:22px;padding:22px;border-radius:14px;
    border:1px solid var(--border);
    background:rgba(255,255,255,0.03);
    display:none;
    animation:fadeUp 0.4s ease;
  }
  @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
  #resultCard.visible{display:block}
  .result-species{font-size:1.55rem;font-weight:700;margin-bottom:4px}
  .result-conf{font-size:0.85rem;color:var(--text-dim);margin-bottom:16px}

  /* Probability bars */
  .prob-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .prob-label{font-size:0.78rem;width:130px;flex-shrink:0;font-weight:500}
  .prob-bar-wrap{flex:1;height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden}
  .prob-bar{height:100%;border-radius:4px;transition:width 0.6s cubic-bezier(0.23,1,0.32,1)}
  .prob-pct{font-family:'JetBrains Mono',monospace;font-size:0.73rem;color:var(--text-dim);width:44px;text-align:right}

  /* ── Right panel ── */
  .right-panel{display:flex;flex-direction:column;gap:24px}

  /* ── Chart ── */
  #chartWrap{position:relative;width:100%;aspect-ratio:4/3}

  /* ── Stats strip ── */
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  .stat{text-align:center;padding:16px;border-radius:12px;border:1px solid var(--border);background:var(--surface)}
  .stat-val{font-size:1.6rem;font-weight:700;background:linear-gradient(135deg,var(--accent1),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-key{font-size:0.72rem;color:var(--text-dim);margin-top:4px;letter-spacing:0.5px;text-transform:uppercase}

  /* ── Color dots ── */
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}

  /* ── Species labels ── */
  .species-c0{color:var(--c0)}.species-c1{color:var(--c1)}.species-c2{color:var(--c2)}
</style>
</head>
<body>

<header>
  <div style="font-size:1.8rem">🌸</div>
  <h1><span>Iris</span> Flower Classifier</h1>
  <span class="badge" id="modelBadge">Loading…</span>
</header>

<div class="layout">

  <!-- LEFT: Controls -->
  <aside>
    <div class="card">
      <p class="card-title">🔬 Flower Measurements (cm)</p>

      <div class="slider-group">
        <div class="slider-header">
          <span class="slider-label">Sepal Length</span>
          <span class="slider-val" id="v-sl">5.1</span>
        </div>
        <input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1"
               oninput="document.getElementById('v-sl').textContent=parseFloat(this.value).toFixed(1)"/>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span class="slider-label">Sepal Width</span>
          <span class="slider-val" id="v-sw">3.5</span>
        </div>
        <input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5"
               oninput="document.getElementById('v-sw').textContent=parseFloat(this.value).toFixed(1)"/>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span class="slider-label">Petal Length</span>
          <span class="slider-val" id="v-pl">1.4</span>
        </div>
        <input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4"
               oninput="document.getElementById('v-pl').textContent=parseFloat(this.value).toFixed(1)"/>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span class="slider-label">Petal Width</span>
          <span class="slider-val" id="v-pw">0.2</span>
        </div>
        <input type="range" id="pw" min="0.1" max="2.5" step="0.1" value="0.2"
               oninput="document.getElementById('v-pw').textContent=parseFloat(this.value).toFixed(1)"/>
      </div>

      <button id="predictBtn" onclick="runPredict()">✨ Classify Flower</button>

      <!-- Result -->
      <div id="resultCard">
        <div id="resultSpecies" class="result-species"></div>
        <div id="resultConf"    class="result-conf"></div>
        <div id="probBars"></div>
      </div>
    </div>
  </aside>

  <!-- RIGHT: Visualizations -->
  <div class="right-panel">

    <!-- Model stats -->
    <div class="stats" id="statsStrip">
      <div class="stat"><div class="stat-val" id="acc">—</div><div class="stat-key">Test Accuracy</div></div>
      <div class="stat"><div class="stat-val" id="f1">—</div><div class="stat-key">F1-Score</div></div>
      <div class="stat"><div class="stat-val" id="pc1">—</div><div class="stat-key">PC1 Variance</div></div>
    </div>

    <!-- PCA Scatter -->
    <div class="card">
      <p class="card-title">
        📉 PCA Cluster Map
        <span style="float:right;font-weight:400;text-transform:none;letter-spacing:0">
          <span class="dot" style="background:var(--c0)"></span><span class="species-c0">Setosa</span>
          &nbsp;
          <span class="dot" style="background:var(--c1)"></span><span class="species-c1">Versicolor</span>
          &nbsp;
          <span class="dot" style="background:var(--c2)"></span><span class="species-c2">Virginica</span>
        </span>
      </p>
      <div id="chartWrap"><canvas id="pcaChart"></canvas></div>
    </div>

  </div>
</div>

<script>
const CLASS_COLORS = {"Iris-setosa":"#00f5d4","Iris-versicolor":"#f72585","Iris-virginica":"#f8961e"};
let pcaChart = null;
let clusterData = {};

async function init(){
  const res  = await fetch('/api/model-info');
  const info = await res.json();

  document.getElementById('modelBadge').textContent = info.model_name;
  document.getElementById('acc').textContent = info.accuracy + '%';
  document.getElementById('f1').textContent  = info.f1_score  + '%';
  document.getElementById('pc1').textContent = info.pca_explained[0].toFixed(1) + '%';

  clusterData = info.pca_clusters;
  buildChart(info.pca_clusters, info.pca_explained, info.classes);
}

function buildChart(clusters, explained, classes){
  const datasets = classes.map(cls => ({
    label: cls,
    data:  clusters[cls].x.map((x,i) => ({x, y: clusters[cls].y[i]})),
    backgroundColor: CLASS_COLORS[cls] + 'cc',
    borderColor:     CLASS_COLORS[cls],
    borderWidth: 1,
    pointRadius: 5,
    pointHoverRadius: 8,
  }));

  // Placeholder prediction point
  datasets.push({
    label: '⭐ Your Flower',
    data:  [],
    backgroundColor: '#ffffff',
    borderColor:     '#ffffff',
    borderWidth: 2,
    pointRadius: 12,
    pointHoverRadius: 14,
    pointStyle: 'star',
    order: -1,
  });

  const ctx = document.getElementById('pcaChart').getContext('2d');
  pcaChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 500, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          labels: { color:'#94a3b8', font:{family:'Outfit',size:12}, padding:20,
                    filter: item => item.text !== '⭐ Your Flower' }
        },
        tooltip: {
          backgroundColor:'rgba(15,15,40,0.9)',
          borderColor:'rgba(255,255,255,0.1)',borderWidth:1,
          titleColor:'#e2e8f0',bodyColor:'#94a3b8',
          callbacks:{
            label: ctx => ` PC1: ${ctx.parsed.x.toFixed(3)}, PC2: ${ctx.parsed.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          title:{display:true, text:`PC1 (${explained[0].toFixed(1)}% variance)`,color:'#64748b'},
          grid:{color:'rgba(255,255,255,0.05)'},
          ticks:{color:'#64748b',font:{family:'JetBrains Mono',size:10}}
        },
        y: {
          title:{display:true, text:`PC2 (${explained[1].toFixed(1)}% variance)`,color:'#64748b'},
          grid:{color:'rgba(255,255,255,0.05)'},
          ticks:{color:'#64748b',font:{family:'JetBrains Mono',size:10}}
        }
      }
    }
  });
}

async function runPredict(){
  const btn = document.getElementById('predictBtn');
  btn.classList.add('loading');
  btn.textContent = '⏳ Classifying…';

  const payload = {
    sepal_length: parseFloat(document.getElementById('sl').value),
    sepal_width:  parseFloat(document.getElementById('sw').value),
    petal_length: parseFloat(document.getElementById('pl').value),
    petal_width:  parseFloat(document.getElementById('pw').value),
  };

  try {
    const res  = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    showResult(data);
    updateChart(data);
  } catch(e){
    alert('Error calling prediction API: ' + e);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = '✨ Classify Flower';
  }
}

function showResult(data){
  const card = document.getElementById('resultCard');
  const col  = CLASS_COLORS[data.predicted_class] || '#e2e8f0';
  document.getElementById('resultSpecies').innerHTML =
    `<span style="color:${col}">${data.predicted_class}</span>`;
  document.getElementById('resultConf').textContent =
    `Confidence: ${data.confidence.toFixed(2)}%`;

  const barsHtml = Object.entries(data.probabilities).map(([cls, pct]) => {
    const c = CLASS_COLORS[cls] || '#7c3aed';
    return `<div class="prob-row">
      <span class="prob-label"><span class="dot" style="background:${c}"></span>${cls.replace('Iris-','')}</span>
      <div class="prob-bar-wrap"><div class="prob-bar" style="width:${pct}%;background:${c}"></div></div>
      <span class="prob-pct">${pct.toFixed(1)}%</span>
    </div>`;
  }).join('');

  document.getElementById('probBars').innerHTML = barsHtml;
  card.classList.add('visible');
}

function updateChart(data){
  if(!pcaChart) return;
  // Last dataset is the prediction star
  const starDs = pcaChart.data.datasets[pcaChart.data.datasets.length - 1];
  starDs.data = [{x: data.pca_x, y: data.pca_y}];
  const col = CLASS_COLORS[data.predicted_class] || '#ffffff';
  starDs.backgroundColor = col;
  starDs.borderColor     = col;
  pcaChart.update();
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the premium interactive Iris dashboard."""
    return HTML


if __name__ == "__main__":
    print("🌸 Starting Iris Classifier Web App at http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
