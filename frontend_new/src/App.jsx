import React, { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

function ImageSlider({ beforeSrc, afterSrc }) {
  const [sliderPos, setSliderPos] = useState(50);

  return (
    <div className="slider-wrapper">
      <p className="slider-subtext">Drag the handle over the image to compare the input and AI reconstruction.</p>

      <div className="slider-container">
        {/* 4x AI Reconstruction (Bottom Layer) */}
        <img src={afterSrc} alt="AI Reconstruction" className="slider-img" />
        <span className="slider-badge right">4× AI RECONSTRUCTION</span>

        {/* 64x64 Input (Top Layer clipped by slider percentage) */}
        <div
          className="slider-clip"
          style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
        >
          <img src={beforeSrc} alt="Original LR" className="slider-img pixelated" />
          <span className="slider-badge left">64 × 64 INPUT</span>
        </div>

        {/* Center Divider & Draggable Puck */}
        <div className="slider-divider" style={{ left: `${sliderPos}%` }}>
          <div className="slider-button">
            <span>‹</span>
            <span>›</span>
          </div>
        </div>

        {/* Range input overlay */}
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPos}
          onChange={(e) => setSliderPos(Number(e.target.value))}
          className="slider-range"
        />
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");

  // SR states
  const [srLoading, setSrLoading] = useState(false);
  const [srResult, setSrResult] = useState(null);
  const [srError, setSrError] = useState("");

  // Mineral states
  const [mineralLoading, setMineralLoading] = useState(false);
  const [mineralResult, setMineralResult] = useState(null);
  const [mineralError, setMineralError] = useState("");

  function handleFileChange(e) {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setSrResult(null);
      setMineralResult(null);
      setSrError("");
      setMineralError("");
    }
  }

  async function runPipeline() {
    if (!file) return;

    setSrLoading(true);
    setMineralLoading(true);
    setSrError("");
    setMineralError("");
    setSrResult(null);
    setMineralResult(null);

    let srData = null;

    // 1. Run Super Resolution
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/api/super-resolution`, {
        method: "POST",
        body: formData,
      });

      srData = await res.json();
      if (!res.ok) throw new Error(srData.detail || "Super resolution failed.");
      setSrResult(srData);
    } catch (err) {
      setSrError(err.message);
      setSrLoading(false);
      setMineralLoading(false);
      return;
    } finally {
      setSrLoading(false);
    }

    // 2. Feed SR output into Mineral Extraction
    const srOutputUrl = srData.output_url || srData.sr_url;
    try {
      const mineralRes = await fetch(`${API_URL}/api/mineral-from-sr`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ output_url: srOutputUrl }),
      });

      const mData = await mineralRes.json();
      if (!mineralRes.ok) throw new Error(mData.detail || "Mineral mapping failed.");
      setMineralResult(mData);
    } catch (err) {
      setMineralError(err.message);
    } finally {
      setMineralLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="navbar">
        <div className="brand">
          <div className="logo-badge">◐</div>
          <div>
            <h1>AstroNex</h1>
            <p>AI for a Clearer Lunar Tomorrow</p>
          </div>
        </div>
        <div className="nav-right">
          <strong>Chandrayaan-2 Data</strong>
          <span>Explore · Enhance · Discover</span>
        </div>
      </header>

      <section className="hero">
        <div className="badge-category">AI-POWERED LUNAR SURFACE INTELLIGENCE</div>
        <h2>From sharp images to deeper lunar insights.</h2>
        <p>Demonstrating trained super-resolution on OHRC-derived lunar imagery with automated mineral phase mapping.</p>
      </section>

      <main className="cards-grid">
        {/* LEFT: Super Resolution */}
        <section className="card">
          <div className="card-header">
            <div className="icon-box blue">🚀</div>
            <div className="card-title">
              <h3>Lunar Super Resolution</h3>
              <p>Upload a 64×64 lunar image and reconstruct a 4× higher-resolution version.</p>
            </div>
            <span className="badge blue-badge">AI POWERED</span>
          </div>

          <div className="upload-row">
            <label className="btn-upload">
              ↑ Choose Image
              <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
            </label>
            <span className="file-name">{file ? file.name : "No image selected"}</span>
          </div>
          <p className="hint">PNG, JPG, JPEG or TIFF · 64 × 64 input recommended</p>

          <button
            className="btn-primary"
            disabled={!file || srLoading || mineralLoading}
            onClick={runPipeline}
          >
            {srLoading
              ? "Reconstructing 4×..."
              : mineralLoading
              ? "Mapping Minerals..."
              : "Run Super Resolution & Mineral Mapping"}
          </button>

          {srError && <div className="error-box">{srError}</div>}

          {srResult && (
            <div className="result-container">
              <div className="result-header">
                <h4>Super Resolution Result</h4>
                <span className="badge green-badge">COMPLETED</span>
              </div>

              {/* Interactive Comparison Slider */}
              <ImageSlider
                beforeSrc={preview}
                afterSrc={`${API_URL}${srResult.output_url || srResult.sr_url}`}
              />

              <div className="stats-grid">
                <div className="stat">
                  <strong>32.9828</strong>
                  <span>PSNR (dB)</span>
                </div>
                <div className="stat">
                  <strong>0.7437</strong>
                  <span>SSIM</span>
                </div>
                <div className="stat">
                  <strong>0.032407</strong>
                  <span>RMSE</span>
                </div>
                <div className="stat">
                  <strong className="positive">+0.0467</strong>
                  <span>PSNR vs Bicubic</span>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* RIGHT: Mineral Extraction */}
        <section className="card">
          <div className="card-header">
            <div className="icon-box purple">🔬</div>
            <div className="card-title">
              <h3>Mineral Extraction</h3>
              <p>Automated spectral-morphological mapping generated from the 4× enhanced output.</p>
            </div>
            <span className="badge purple-badge">AUTO CHAINED</span>
          </div>

          {mineralLoading && (
            <div className="placeholder-box loading">
              Analyzing 4× crater morphology and extracting mineral phases...
            </div>
          )}

          {mineralError && <div className="error-box">{mineralError}</div>}

          {!mineralResult && !mineralLoading && (
            <div className="placeholder-box">
              Select an image on the left and run the pipeline.<br />
              Mineral extraction will execute automatically on the 4× reconstructed output.
            </div>
          )}

          {mineralResult && (
            <div className="result-container">
              <div className="result-header">
                <h4>Mineral Map Generated (4× Coregistered)</h4>
                <span className="badge purple-badge">RESEARCH PREVIEW</span>
              </div>

              <div className="image-box single">
                <img src={`${API_URL}${mineralResult.mineral_url}`} alt="Mineral Map" />
              </div>

              <div className="legend-row">
                <span className="legend-item"><i className="dot red"></i> Olivine (Rim slopes)</span>
                <span className="legend-item"><i className="dot green"></i> Pyroxene (Mare basalts)</span>
                <span className="legend-item"><i className="dot blue"></i> Anorthosite (Highlands)</span>
              </div>

              <div className="stats-grid three-col">
                <div className="stat">
                  <strong style={{ color: "#ef4444" }}>{mineralResult.abundances?.olivine_pct ?? 0}%</strong>
                  <span>Olivine Phase</span>
                </div>
                <div className="stat">
                  <strong style={{ color: "#22c55e" }}>{mineralResult.abundances?.pyroxene_pct ?? 0}%</strong>
                  <span>Pyroxene Basalt</span>
                </div>
                <div className="stat">
                  <strong style={{ color: "#3b82f6" }}>{mineralResult.abundances?.anorthosite_pct ?? 0}%</strong>
                  <span>Anorthosite Highlands</span>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}