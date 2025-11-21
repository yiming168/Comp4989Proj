import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const formatPercent = (v) => `${(v * 100).toFixed(1)}%`;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [useGradCam, setUseGradCam] = useState(true);
  const [useSuggestions, setUseSuggestions] = useState(true);
  const [threshold, setThreshold] = useState(0.2);
  const [health, setHealth] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));

    return () => stopCamera();
  }, []);

  const handleFile = (files) => {
    if (!files?.length) return;
    const f = files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError("");
    stopCamera();
  };

  const startCamera = async () => {
    try {
      // reset previous capture when reopening camera
      setFile(null);
      setPreview(null);
      setError("");
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      setCameraOn(true);
    } catch (e) {
      setError("Camera access failed. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    streamRef.current = null;
    setCameraOn(false);
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    if (!video || !streamRef.current || !cameraOn) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const captured = new File([blob], "capture.jpg", { type: "image/jpeg" });
      setFile(captured);
      setPreview(URL.createObjectURL(blob));
      setError("");
      stopCamera(); // auto-close camera after capture
    }, "image/jpeg", 0.92);
  };

  const submit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Choose or capture an image first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("grad_cam", String(useGradCam));
      form.append("suggestions", String(useSuggestions));
      form.append("threshold", String(threshold));

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const probs = result?.probs
    ? Object.entries(result.probs).sort((a, b) => b[1] - a[1])
    : [];

  const showVideo = Boolean(streamRef.current);
  const showingVideo = cameraOn && showVideo;

  useEffect(() => {
    if (showingVideo && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [showingVideo]);

  return (
    <div className="page">
      <header className="hero">
        <h1>Nutrition Deficiency Detect</h1>
        <p className="lede">
          Share a photo of your eye, nail, or tongue, and we'll help identify possible vitamin deficiencies, plus give personalized food suggestions!
        </p>
      </header>

      <main className="layout single">
        <section className="card input-card">
          <div className="card-header center">
            <h2>Input</h2>
            <div className="pill">Device: {health?.device || "detecting..."}</div>
          </div>

          <div className="upload-controls">
            <div className="preview">
              {showingVideo ? (
                <video className="camera" ref={videoRef} playsInline muted autoPlay></video>
              ) : preview ? (
                <img src={preview} alt="preview" />
              ) : (
                <div className="placeholder">No image yet. Upload or open the camera.</div>
              )}
            </div>

            <div className="button-row">
              <label className="button ghost" htmlFor="file-upload">
                Upload image
              </label>
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                capture="environment"
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files)}
              />
              <button className="button secondary" type="button" onClick={startCamera}>
                Open camera
              </button>
              <button className="button" type="button" onClick={capturePhoto} disabled={!cameraOn}>
                Capture photo
              </button>
              <button className="link" type="button" onClick={stopCamera} disabled={!cameraOn}>
                Stop
              </button>
            </div>

            <div className="toggles">
              <label className="switch">
                <input
                  type="checkbox"
                  checked={useGradCam}
                  onChange={(e) => setUseGradCam(e.target.checked)}
                />
                <span>Save Grad-CAM overlay</span>
              </label>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={useSuggestions}
                  onChange={(e) => setUseSuggestions(e.target.checked)}
                />
                <span>Gemini food suggestions</span>
              </label>
            </div>

            <label className="slider-label">
              Suggestion threshold ({threshold.toFixed(2)})
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>

            <button className="button primary" type="button" onClick={submit} disabled={loading}>
              {loading ? "Running..." : "Run prediction"}
            </button>

            {error && <div className="alert error">{error}</div>}
          </div>
        </section>

        <section className="card result-card">
          <div className="card-header center">
            <h2>Prediction & Suggestion</h2>
          </div>

          {!result && <p className="muted">Results will appear here after you run a prediction.</p>}

          {result && (
            <div className="result">
              <div className="summary column">
                <div>
                  <p className="muted">Predicted class</p>
                  <h3>{result.pred_class}</h3>
                  <p className="confidence">Confidence {formatPercent(result.confidence)}</p>
                  {result.deficiency_info && (
                    <p className="muted">
                      Potential deficiency: {result.deficiency_info.deficiency} - {result.deficiency_info.symptom}
                    </p>
                  )}
                  {result.food_suggestions_error && (
                    <div className="alert error small">{result.food_suggestions_error}</div>
                  )}
                  {result.food_suggestions_skipped && (
                    <div className="alert warn small">{result.food_suggestions_skipped}</div>
                  )}
                </div>
              </div>

              {probs.length > 0 && (
                <div className="probabilities">
                  {probs.map(([cls, p]) => (
                    <div key={cls} className="prob-row">
                      <span>{cls}</span>
                      <div className="bar">
                        <div className="fill" style={{ width: `${Math.max(p * 100, 2)}%` }}></div>
                      </div>
                      <span className="percent">{formatPercent(p)}</span>
                    </div>
                  ))}
                </div>
              )}

              {result.grad_cam_image && (
                <div className="gradcam">
                  <p className="label">Grad-CAM overlay</p>
                  <img src={result.grad_cam_image} alt="Grad-CAM" />
                </div>
              )}

              {result.food_suggestions && (
                <div className="suggestions">
                  <div className="label">Gemini food suggestions</div>
                  <pre>{result.food_suggestions}</pre>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <p>
          This tool is a research demo, not a medical device. For health concerns, consult a healthcare professional.
        </p>
        <p className="muted">API: {API_URL} - Model: best_model.pth - Grad-CAM optional</p>
      </footer>
    </div>
  );
}

export default App;
