import { useEffect, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

function ComparisonSlider({ before, after }) {
  const [position, setPosition] = useState(50);
  const dragging = useRef(false);

  const update = (clientX, target) => {
    const rect = target.getBoundingClientRect();
    const value = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setPosition(value);
  };

  return (
    <div
      className="comparison"
      onPointerDown={(e) => {
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        update(e.clientX, e.currentTarget);
      }}
      onPointerMove={(e) => {
        if (dragging.current) update(e.clientX, e.currentTarget);
      }}
      onPointerUp={() => { dragging.current = false; }}
      onPointerCancel={() => { dragging.current = false; }}
    >
      <img src={after} alt="4x AI reconstruction" className="comparison-image" />
      <img
        src={before}
        alt="64x64 input"
        className="comparison-image input-image"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      />

      <div className="compare-label input-label">64 × 64 INPUT</div>
      <div className="compare-label ai-label">4× AI RECONSTRUCTION</div>

      <div className="slider-line" style={{ left: `${position}%` }}>
        <div className="slider-handle">↔</div>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [inputPreview, setInputPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setInputPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setInputPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function runSuperResolution() {
    if (!file) {
      setError("Please choose a lunar image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API_URL}/api/super-resolution`, {
        method: "POST",
        body,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Processing failed.");
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <div className="brand-moon">◐</div>
          <div>
            <h1>AstroNex</h1>
            <p>AI for a Clearer Lunar Tomorrow</p>
          </div>
        </div>
        <div className="header-right">
          <strong>Chandrayaan-2 Data</strong>
          <span>Explore · Enhance · Discover</span>
        </div>
      </header>

      <main className="main">
        <section className="hero">
          <p className="eyebrow">AI-POWERED LUNAR SURFACE INTELLIGENCE</p>
          <h2>From sharp images to deeper lunar insights.</h2>
          <p>Demonstrating trained super-resolution on OHRC-derived lunar imagery with a research-preview mineral workflow.</p>
        </section>

        <div className="dashboard">
          <section className="card super-card">
            <div className="card-heading">
              <div className="heading-icon blue">🚀</div>
              <div>
                <h3>Lunar Super Resolution</h3>
                <p>Upload a 64 × 64 lunar image and reconstruct a 4× higher-resolution version.</p>
              </div>
              <span className="badge blue-badge">AI POWERED</span>
            </div>

            <div className="upload-row">
              <label className="choose blue-button">
                ↑ &nbsp; Choose Image
                <input
                  type="file"
                  hidden
                  accept=".png,.jpg,.jpeg,.tif,.tiff"
                  onChange={(e) => {
                    const selected = e.target.files?.[0];
                    if (selected) {
                      setFile(selected);
                      setResult(null);
                      setError("");
                    }
                  }}
                />
              </label>
              <div className="selected-file">
                {file ? `✓  ${file.name}` : "No image selected"}
              </div>
            </div>
            <div className="file-note">PNG, JPG, JPEG or TIFF · 64 × 64 input recommended</div>

            {file && !result && !loading && (
              <div className="preview small-preview">
                <img src={inputPreview} alt="Input preview" />
              </div>
            )}

            <button className="run-button" disabled={!file || loading} onClick={runSuperResolution}>
              {loading ? "Running AstroNex AI..." : "Run Super Resolution"}
            </button>

            {error && <div className="error">{error}</div>}

            {result && (
              <div className="result">
                <div className="result-heading-row">
                  <div>
                    <h3>Super Resolution Result</h3>
                    <p>Drag the handle across the image to compare the input and AI reconstruction.</p>
                  </div>
                  <span className="complete-badge">COMPLETED</span>
                </div>

                <ComparisonSlider
                  before={`${API_URL}${result.pixelated_url}`}
                  after={`${API_URL}${result.output_url}`}
                />

                <div className="image-info">
                  <div><strong>64 × 64</strong><span>Input</span></div>
                  <div className="arrow">→</div>
                  <div><strong>256 × 256</strong><span>4× AI Reconstruction</span></div>
                </div>

                <div className="metrics">
                  <div className="metric"><strong>{result.benchmark.astronex_psnr_db}</strong><span>PSNR (dB)</span></div>
                  <div className="metric"><strong>{result.benchmark.astronex_ssim}</strong><span>SSIM</span></div>
                  <div className="metric"><strong>{result.benchmark.astronex_rmse}</strong><span>RMSE</span></div>
                  <div className="metric improvement"><strong>+{result.benchmark.psnr_gain_db}</strong><span>PSNR vs Bicubic</span></div>
                </div>

                <div className="benchmark-note">
                  Held-out OHRC test-set benchmark · Bicubic PSNR 35.3105 dB → AstroNex 36.5302 dB
                </div>
              </div>
            )}
          </section>

          <section className="card mineral-card">
            <div className="card-heading">
              <div className="heading-icon purple">🔬</div>
              <div>
                <h3>Mineral Extraction</h3>
                <p>Explore the intended IIRS hyperspectral analysis workflow.</p>
              </div>
              <span className="badge purple-badge">DEMO</span>
            </div>

            <div className="sample-title">
              <div>
                <h3>Sample IIRS Analysis</h3>
                <p>Illustrative output for the planned mineral-analysis module.</p>
              </div>
              <span>RESEARCH PREVIEW</span>
            </div>

            <div className="sample-grid">
              <div className="sample-card">
                <div className="sample-label">Lunar Surface</div>
                <img src="/lunar_sample.png" alt="Illustrative lunar surface" />
              </div>
              <div className="sample-card">
                <div className="sample-label">Mineral Distribution</div>
                <img src="/mineral_map_sample.png" alt="Illustrative mineral distribution" />
              </div>
            </div>

            <div className="legend">
              <strong>Potential Mineral Classes</strong>
              <div className="legend-grid">
                <span><i className="dot red" />Pyroxene</span>
                <span><i className="dot green" />Olivine</span>
                <span><i className="dot blue" />Plagioclase</span>
                <span><i className="dot yellow" />Hydroxyl / H₂O Potential</span>
                <span><i className="dot purple" />Other Silicates</span>
              </div>
            </div>

            <div className="demo-note">
              <strong>DEMO / RESEARCH PREVIEW</strong>
              <p>This sample visualization is illustrative and is not a validated mineral-classification result. Replace it with calibrated IIRS-derived outputs when the spectral model is trained.</p>
            </div>
          </section>
        </div>
      </main>

      <footer>AstroNex · Chandrayaan-2 · Education & Research</footer>
    </div>
  );
}
