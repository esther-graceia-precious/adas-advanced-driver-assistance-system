import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";

const API_URL = "http://localhost:5000";

const css = `
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }
  .slide-in { animation: slideIn 0.4s ease forwards; }
  .pulse { animation: pulse 1s ease infinite; }
`;

const playAlert = () => {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const gongInterval = 1.5;
    const gongCount = Math.floor(15 / gongInterval);
    const playGong = (t) => {
      [[220,1.5,1.2],[440,0.8,1.0],[660,0.5,0.8]].forEach(([freq,vol,dur]) => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.frequency.value = freq; o.type = 'sine';
        g.gain.setValueAtTime(vol, t);
        g.gain.exponentialRampToValueAtTime(0.001, t + dur);
        o.start(t); o.stop(t + dur);
      });
    };
    for (let i = 0; i < gongCount; i++) playGong(ctx.currentTime + i * gongInterval);
  } catch(e) {}
};

const playBeep = () => {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.connect(g); g.connect(ctx.destination);
    o.frequency.value = 880; o.type = 'sine';
    g.gain.setValueAtTime(0.3, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    o.start(ctx.currentTime); o.stop(ctx.currentTime + 0.3);
  } catch(e) {}
};

const riskStyle = (level) => {
  if (level === 'HIGH')   return { color: '#c53030', bg: '#fff5f5', border: '#feb2b2' };
  if (level === 'MEDIUM') return { color: '#d97706', bg: '#fffbeb', border: '#fcd34d' };
  return                         { color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' };
};

const gradeStyle = (grade) => {
  if (grade === 'A') return { color: '#276749', bg: '#f0fff4', border: '#9ae6b4' };
  if (grade === 'B') return { color: '#2b6cb0', bg: '#ebf8ff', border: '#90cdf4' };
  if (grade === 'C') return { color: '#d97706', bg: '#fffbeb', border: '#fcd34d' };
  return                    { color: '#c53030', bg: '#fff5f5', border: '#feb2b2' };
};

// ================================
// MULTISTREAM TAGS COMPONENT
// ================================
function MultiStreamTags({ ms }) {
  if (!ms) return null;
  const faceFound = ms.head && ms.head !== 'N/A';
  return (
    <div style={{ marginTop: '0.75rem', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
      {faceFound ? (
        <>
          <span style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#1d4ed8' }}>
            👤 {ms.head}
          </span>
          <span style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#166534' }}>
            👁️ {ms.eye}
          </span>
          <span style={{ background: '#fdf4ff', border: '1px solid #e9d5ff', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#7e22ce' }}>
            👄 {ms.mouth}
          </span>
        </>
      ) : (
        <span style={{ background: '#fff9f5', border: '1px solid #ff6600', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#ff6600' }}>
          👤 Face not detected in this frame
        </span>
      )}
      {ms.reasons && ms.reasons.length > 0 && ms.reasons.map((r, i) => (
        <span key={i} style={{ background: '#fff5f5', border: '1px solid #feb2b2', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#c53030', fontWeight: 600 }}>
          ⚠️ {r}
        </span>
      ))}
    </div>
  );
}

// ================================
// SESSION SUMMARY MODAL
// Shown when user stops live session
// ================================
function SessionSummaryModal({ summary, onClose }) {
  if (!summary) return null;
  const { summary: s, safety, fatigue } = summary;
  const gs = gradeStyle(safety.grade);
  const rs = riskStyle(s.risk_level);

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000, padding: '1rem'
    }}>
      <div className="slide-in" style={{
        background: '#fff', borderRadius: '16px', padding: '2rem',
        maxWidth: '480px', width: '100%', boxShadow: '0 20px 60px rgba(0,0,0,0.2)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#111', marginBottom: '2px' }}>Session Summary</h2>
            <p style={{ fontSize: '0.78rem', color: '#aaa' }}>{s.total_frames_analyzed} frames analyzed</p>
          </div>
          <button onClick={onClose} style={{
            background: '#f5f5f5', border: 'none', borderRadius: '8px',
            padding: '6px 14px', cursor: 'pointer', fontSize: '0.85rem', color: '#555'
          }}>Close</button>
        </div>

        {/* Safety Grade — big hero number */}
        <div style={{
          background: gs.bg, border: `2px solid ${gs.border}`,
          borderRadius: '12px', padding: '1.5rem', textAlign: 'center', marginBottom: '1.25rem'
        }}>
          <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '6px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Safety Grade</div>
          <div style={{ fontSize: '4rem', fontWeight: 800, color: gs.color, lineHeight: 1 }}>{safety.grade}</div>
          <div style={{ fontSize: '0.85rem', color: gs.color, marginTop: '6px', fontWeight: 600 }}>{safety.message}</div>
          <div style={{ fontSize: '0.78rem', color: '#888', marginTop: '4px' }}>Score: {safety.score} / 100</div>
        </div>

        {/* Stats grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1.25rem' }}>
          {[
            { label: 'Attentive',  value: `${s.attentive_pct}%`,  color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' },
            { label: 'Distracted', value: `${s.distracted_pct}%`, color: '#c53030', bg: '#fff5f5', border: '#feb2b2' },
            { label: 'Risk Level', value: s.risk_level,            color: rs.color,  bg: rs.bg,    border: rs.border },
          ].map(({ label, value, color, bg, border }) => (
            <div key={label} style={{ background: bg, border: `1px solid ${border}`, borderRadius: '10px', padding: '0.9rem 1rem' }}>
              <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px', fontWeight: 500 }}>{label}</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
            </div>
          ))}
        </div>

        {/* Fatigue */}
        <div style={{ background: fatigue.event_count > 0 ? '#fffbeb' : '#f0fff4', border: `1px solid ${fatigue.event_count > 0 ? '#fcd34d' : '#9ae6b4'}`, borderRadius: '10px', padding: '1rem 1.25rem' }}>
          <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Fatigue Events</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '2rem', fontWeight: 700, color: fatigue.event_count > 0 ? '#d97706' : '#38a169' }}>
              {fatigue.event_count}
            </span>
            <span style={{ fontSize: '0.8rem', color: '#888' }}>
              {fatigue.event_count === 0
                ? 'No fatigue detected — great session!'
                : `events · longest streak: ${fatigue.max_duration_frames} frames`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ================================
// IMAGE MODE  (unchanged)
// ================================
function ImageMode() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFile = (f) => {
    setFile(f); setResult(null); setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    const fd = new FormData();
    fd.append('image', file);
    try {
      const res = await axios.post(`${API_URL}/analyze_image`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
      if (res.data.label === 'Distracted') playAlert();
    } catch {
      setError('Could not connect to backend.');
    } finally { setLoading(false); }
  };

  const bad = result?.label === 'Distracted';
  const rs = result ? riskStyle(result.risk_level) : null;

  return (
    <div>
      <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e8e8e8', padding: '1.5rem', marginBottom: '1.25rem' }}>
        <p style={{ fontSize: '0.8rem', fontWeight: 600, color: '#555', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Upload Image</p>
        <label style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          border: '2px dashed #ddd', borderRadius: '10px', padding: '2rem', cursor: 'pointer',
          background: file ? '#fff9f5' : '#fafafa', borderColor: file ? '#ff6600' : '#ddd',
          marginBottom: '1rem', gap: '8px'
        }}>
          <input type="file" accept="image/*" onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
          {preview ? (
            <img src={preview} alt="preview" style={{ maxHeight: '180px', borderRadius: '8px', objectFit: 'contain' }} />
          ) : (
            <>
              <span style={{ fontSize: '2rem' }}>🖼️</span>
              <span style={{ fontWeight: 500, color: '#555', fontSize: '0.9rem' }}>Click to upload an image</span>
              <span style={{ color: '#aaa', fontSize: '0.8rem' }}>JPG, PNG supported</span>
            </>
          )}
        </label>
        <button onClick={analyze} disabled={!file || loading} style={{
          width: '100%', padding: '12px',
          background: !file || loading ? '#f0f0f0' : '#ff6600',
          color: !file || loading ? '#aaa' : '#fff',
          border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.95rem',
          cursor: !file || loading ? 'not-allowed' : 'pointer'
        }}>
          {loading ? 'Analyzing...' : 'Analyze Image'}
        </button>
        {error && <p style={{ color: '#e53e3e', fontSize: '0.82rem', marginTop: '10px' }}>{error}</p>}
      </div>

      {result && (
        <div className="slide-in">
          <div style={{
            background: bad ? '#fff5f5' : '#f0fff4', border: `1px solid ${bad ? '#feb2b2' : '#9ae6b4'}`,
            borderRadius: '10px', padding: '14px 18px', display: 'flex', alignItems: 'center', gap: '10px',
            marginBottom: '1.25rem', animation: bad ? 'blink 0.8s ease infinite' : 'none'
          }}>
            <span style={{ fontSize: '1.4rem' }}>{bad ? '⚠️' : '✅'}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: bad ? '#c53030' : '#276749' }}>
                {bad ? 'Driver is Distracted' : 'Driver is Attentive'}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Confidence: {result.confidence}%</div>
              {result.multistream && <MultiStreamTags ms={result.multistream} />}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
            <div style={{ background: bad ? '#fff5f5' : '#f0fff4', border: `1px solid ${bad ? '#feb2b2' : '#9ae6b4'}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
              <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Result</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 700, color: bad ? '#c53030' : '#38a169', lineHeight: 1 }}>{result.label}</div>
            </div>
            <div style={{ background: rs.bg, border: `1px solid ${rs.border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
              <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Risk Level</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 700, color: rs.color, lineHeight: 1 }}>{result.risk_level}</div>
            </div>
          </div>

          {result.gradcam && (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Explainability — Grad-CAM</p>
              <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>{result.confidence}% confidence. Red areas show model focus.</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <p style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px' }}>Original Image</p>
                  <img src={`data:image/jpeg;base64,${result.gradcam.original}`} alt="Original" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e8e8e8' }} />
                </div>
                <div>
                  <p style={{ fontSize: '0.75rem', color: '#ff6600', marginBottom: '6px' }}>Grad-CAM Heatmap</p>
                  <img src={`data:image/jpeg;base64,${result.gradcam.gradcam}`} alt="Grad-CAM" style={{ width: '100%', borderRadius: '8px', border: '1px solid #fcd34d' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ================================
// VIDEO MODE  (unchanged)
// ================================
function VideoMode() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);

  const handleFile = (f) => { setFile(f); setResult(null); setError(null); };
  const onDrop = e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith('video/')) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setProgress(0);
    const fd = new FormData();
    fd.append('video', file);
    try {
      const res = await axios.post(`${API_URL}/analyze`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: e => setProgress(Math.round(e.loaded * 100 / e.total))
      });
      setResult(res.data);
      setCurrentFrame(0);
      if (res.data.summary.distracted_pct > 50) playAlert();
    } catch {
      setError('Could not connect to backend. Make sure Flask is running.');
    } finally { setLoading(false); }
  };

  const chart = result?.frame_data?.timestamps.map((t, i) => ({
    time: `${t}s`,
    confidence: parseFloat((result.frame_data.confidences[i] * 100).toFixed(1))
  }));

  const bad = result?.summary?.overall_status === 'DISTRACTED';
  const risk = result?.summary?.risk_level;
  const rs = risk ? riskStyle(risk) : { color: '#888', bg: '#fafafa', border: '#e8e8e8' };
  const gs = result?.safety ? gradeStyle(result.safety.grade) : null;

  const currentMs = result?.multistream ? {
    head:    result.multistream.head?.[currentFrame]    || 'N/A',
    eye:     result.multistream.eye?.[currentFrame]     || 'N/A',
    mouth:   result.multistream.mouth?.[currentFrame]   || 'N/A',
    reasons: result.multistream.reasons?.[currentFrame] || []
  } : null;

  return (
    <div>
      <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e8e8e8', padding: '1.5rem', marginBottom: '1.25rem' }}>
        <p style={{ fontSize: '0.8rem', fontWeight: 600, color: '#555', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Upload Video</p>
        <label onDragOver={e => e.preventDefault()} onDrop={onDrop} style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          border: '2px dashed #ddd', borderRadius: '10px', padding: '2rem', cursor: 'pointer',
          background: file ? '#fff9f5' : '#fafafa', borderColor: file ? '#ff6600' : '#ddd',
          marginBottom: '1rem', gap: '8px'
        }}>
          <input type="file" accept="video/*" onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
          {file ? (
            <>
              <span style={{ fontSize: '2rem' }}>🎬</span>
              <span style={{ fontWeight: 600, color: '#ff6600', fontSize: '0.9rem' }}>{file.name}</span>
              <span style={{ color: '#aaa', fontSize: '0.8rem' }}>{(file.size/1024/1024).toFixed(1)} MB · click to change</span>
            </>
          ) : (
            <>
              <span style={{ fontSize: '2rem' }}>📁</span>
              <span style={{ fontWeight: 500, color: '#555', fontSize: '0.9rem' }}>Click to upload or drag a video here</span>
              <span style={{ color: '#aaa', fontSize: '0.8rem' }}>MP4, AVI, MOV supported</span>
            </>
          )}
        </label>

        {loading && (
          <div style={{ marginBottom: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#888', marginBottom: '5px' }}>
              <span>{progress < 100 ? 'Uploading...' : 'Analyzing frames...'}</span>
              <span>{progress < 100 ? `${progress}%` : 'please wait'}</span>
            </div>
            <div style={{ height: '6px', background: '#f0f0f0', borderRadius: '99px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${progress}%`, background: '#ff6600', borderRadius: '99px', transition: 'width 0.3s' }} />
            </div>
          </div>
        )}

        <button onClick={analyze} disabled={!file || loading} style={{
          width: '100%', padding: '12px',
          background: !file || loading ? '#f0f0f0' : '#ff6600',
          color: !file || loading ? '#aaa' : '#fff',
          border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.95rem',
          cursor: !file || loading ? 'not-allowed' : 'pointer'
        }}>
          {loading ? 'Analyzing...' : 'Analyze Video'}
        </button>
        {error && <p style={{ color: '#e53e3e', fontSize: '0.82rem', marginTop: '10px' }}>{error}</p>}
      </div>

      {result && (
        <div className="slide-in">
          <div style={{
            background: bad ? '#fff5f5' : '#f0fff4', border: `1px solid ${bad ? '#feb2b2' : '#9ae6b4'}`,
            borderRadius: '10px', padding: '14px 18px', display: 'flex', alignItems: 'center', gap: '10px',
            marginBottom: '1.25rem', animation: bad ? 'blink 0.8s ease infinite' : 'none'
          }}>
            <span style={{ fontSize: '1.4rem' }}>{bad ? '⚠️' : '✅'}</span>
            <div>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: bad ? '#c53030' : '#276749' }}>
                {bad ? 'Driver Distraction Detected' : 'Driver is Attentive'}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>
                {result.summary.duration_seconds}s video · {result.summary.total_frames_analyzed} frames analyzed
              </div>
            </div>
          </div>

          {result.processed_frames && result.processed_frames.length > 0 && (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Video Analysis Frames</p>
              <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1rem' }}>Frame {currentFrame + 1} of {result.processed_frames.length}</p>
              <img src={`data:image/jpeg;base64,${result.processed_frames[currentFrame]}`} alt={`Frame ${currentFrame}`}
                style={{ width: '100%', borderRadius: '8px', border: '1px solid #e8e8e8', marginBottom: '0.75rem' }} />
              {currentMs && <MultiStreamTags ms={currentMs} />}
              <input type="range" min={0} max={result.processed_frames.length - 1} value={currentFrame}
                onChange={e => setCurrentFrame(Number(e.target.value))}
                style={{ width: '100%', accentColor: '#ff6600', marginTop: '0.75rem' }} />
              <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                <button onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}
                  style={{ flex: 1, padding: '6px', background: '#f5f5f5', border: '1px solid #e8e8e8', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem' }}>← Prev</button>
                <button onClick={() => setCurrentFrame(Math.min(result.processed_frames.length - 1, currentFrame + 1))}
                  style={{ flex: 1, padding: '6px', background: '#f5f5f5', border: '1px solid #e8e8e8', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem' }}>Next →</button>
              </div>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
            {[
              { label: 'Attentive',  value: `${result.summary.attentive_pct}%`,  color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' },
              { label: 'Distracted', value: `${result.summary.distracted_pct}%`, color: bad ? '#c53030' : '#555', bg: bad ? '#fff5f5' : '#fafafa', border: bad ? '#feb2b2' : '#e8e8e8' },
              { label: 'Alerts',     value: result.summary.alert_count,           color: '#d97706', bg: '#fffbeb', border: '#fcd34d' },
              { label: 'Risk Level', value: risk,                                  color: rs.color,  bg: rs.bg,    border: rs.border },
            ].map(({ label, value, color, bg, border }) => (
              <div key={label} style={{ background: bg, border: `1px solid ${border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>{label}</div>
                <div style={{ fontSize: label === 'Risk Level' ? '1.1rem' : '2rem', fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Safety Grade for video */}
          {result.safety && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ background: gs.bg, border: `1px solid ${gs.border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Safety Grade</div>
                <div style={{ fontSize: '2rem', fontWeight: 700, color: gs.color, lineHeight: 1 }}>{result.safety.grade}</div>
                <div style={{ fontSize: '0.72rem', color: gs.color, marginTop: '4px' }}>{result.safety.message}</div>
              </div>
              <div style={{ background: '#fafafa', border: '1px solid #e8e8e8', borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Safety Score</div>
                <div style={{ fontSize: '2rem', fontWeight: 700, color: '#222', lineHeight: 1 }}>{result.safety.score}</div>
                <div style={{ fontSize: '0.72rem', color: '#aaa', marginTop: '4px' }}>out of 100</div>
              </div>
              <div style={{ background: result.fatigue?.event_count > 0 ? '#fffbeb' : '#f0fff4', border: `1px solid ${result.fatigue?.event_count > 0 ? '#fcd34d' : '#9ae6b4'}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Fatigue Events</div>
                <div style={{ fontSize: '2rem', fontWeight: 700, color: result.fatigue?.event_count > 0 ? '#d97706' : '#38a169', lineHeight: 1 }}>{result.fatigue?.event_count ?? 0}</div>
                <div style={{ fontSize: '0.72rem', color: '#aaa', marginTop: '4px' }}>detected this session</div>
              </div>
            </div>
          )}

          <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
            <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Confidence Over Time</p>
            <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>Above 50% = Distracted · Below 50% = Attentive</p>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chart} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#aaa' }} interval={Math.floor((chart?.length || 1) / 7)} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#aaa' }} />
                <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '8px', fontSize: '0.78rem' }}
                  formatter={v => [`${v}%`, 'Distraction']} />
                <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="4 3" />
                <Line type="monotone" dataKey="confidence" stroke="#ff6600" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {result.alerts.length > 0 && (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '1rem' }}>Distraction Events ({result.alerts.length})</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '8px' }}>
                {result.alerts.map((a, i) => (
                  <div key={i} style={{ background: '#fff5f5', border: '1px solid #fed7d7', borderRadius: '8px', padding: '10px 12px' }}>
                    <div style={{ fontSize: '0.68rem', color: '#aaa', marginBottom: '3px' }}>Event {i + 1}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#c53030' }}>{a.start}s → {a.end}s</div>
                    <div style={{ fontSize: '0.72rem', color: '#aaa' }}>{(a.end - a.start).toFixed(1)}s</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.gradcam && (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Explainability — Grad-CAM</p>
              <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>Most distracted frame ({result.gradcam.confidence}% confidence). Red areas show model focus.</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <p style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px' }}>Original Frame</p>
                  <img src={`data:image/jpeg;base64,${result.gradcam.original}`} alt="Original" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e8e8e8' }} />
                </div>
                <div>
                  <p style={{ fontSize: '0.75rem', color: '#ff6600', marginBottom: '6px' }}>Grad-CAM Heatmap</p>
                  <img src={`data:image/jpeg;base64,${result.gradcam.gradcam}`} alt="Grad-CAM" style={{ width: '100%', borderRadius: '8px', border: '1px solid #fcd34d' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ================================
// LIVE MODE  — session-based
// ================================
function LiveMode() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const streamRef   = useRef(null);
  const intervalRef = useRef(null);
  const sessionRef  = useRef(null);   // stores session_id from /live_start

  const [isRunning,            setIsRunning]            = useState(false);
  const [result,               setResult]               = useState(null);
  const [error,                setError]                = useState(null);
  const [frameCount,           setFrameCount]           = useState(0);
  const [consecutiveDistracted,setConsecutiveDistracted] = useState(0);
  const [liveHistory,          setLiveHistory]          = useState([]);
  const [sessionSummary,       setSessionSummary]       = useState(null); // shown in modal after stop

  // ── Stop: ends session, fetches summary, cleans up ──
  const stopCamera = useCallback(async () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsRunning(false);

    // Call /live_end to get session summary
    if (sessionRef.current) {
      try {
        const res = await axios.post(`${API_URL}/live_end`, { session_id: sessionRef.current });
        setSessionSummary(res.data);
      } catch(e) {
        console.error('live_end failed:', e);
      }
      sessionRef.current = null;
    }
  }, []);

  // ── Capture frame → POST to /analyze_live ──
  const captureAndAnalyze = useCallback(async () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !sessionRef.current) return;

    const ctx = canvas.getContext('2d');
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    // Convert canvas to base64 JPEG
    const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

    try {
      const res = await axios.post(`${API_URL}/analyze_live`, {
        session_id: sessionRef.current,
        image: base64
      });
      const data = res.data;
      setResult(data);
      setFrameCount(prev => prev + 1);

      if (data.label === 'Distracted') {
        setConsecutiveDistracted(prev => {
          const n = prev + 1;
          if (n === 3) playBeep();
          if (n === 5) playAlert();
          return n;
        });
      } else {
        setConsecutiveDistracted(0);
      }

      setLiveHistory(prev => {
        const updated = [...prev, {
          time: `${prev.length + 1}`,
          confidence: data.confidence
        }];
        return updated.slice(-30);
      });

    } catch(e) {
      console.error('analyze_live error:', e);
    }
  }, []);

  // ── Start: creates session then opens camera ──
  const startCamera = async () => {
    setError(null);
    setResult(null);
    setFrameCount(0);
    setConsecutiveDistracted(0);
    setLiveHistory([]);
    setSessionSummary(null);

    // 1. Start session on backend
    try {
      const res = await axios.post(`${API_URL}/live_start`);
      sessionRef.current = res.data.session_id;
    } catch(e) {
      setError('Could not start session. Make sure Flask is running.');
      return;
    }

    // 2. Open webcam
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsRunning(true);
      intervalRef.current = setInterval(captureAndAnalyze, 1000);
    } catch(e) {
      setError('Camera access denied. Please allow camera permissions and try again.');
      sessionRef.current = null;
    }
  };

  useEffect(() => { return () => stopCamera(); }, [stopCamera]);

  const bad = result?.label === 'Distracted';
  const rs  = result ? riskStyle(result.risk_level) : null;

  // Session stats come from result.session (live route returns running totals)
  const session = result?.session;

  return (
    <div>
      {/* Session Summary Modal */}
      {sessionSummary && (
        <SessionSummaryModal
          summary={sessionSummary}
          onClose={() => setSessionSummary(null)}
        />
      )}

      {/* Camera Feed */}
      <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e8e8e8', padding: '1.4rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '2px' }}>Live Camera Feed</p>
            <p style={{ fontSize: '0.75rem', color: '#aaa' }}>
              {isRunning ? `Analyzing · ${frameCount} frames processed` : 'Click Start to begin monitoring'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {!isRunning ? (
              <button onClick={startCamera} style={{
                padding: '8px 20px', background: '#ff6600', color: '#fff',
                border: 'none', borderRadius: '8px', fontWeight: 600,
                fontSize: '0.85rem', cursor: 'pointer'
              }}>▶️ Start</button>
            ) : (
              <button onClick={stopCamera} style={{
                padding: '8px 20px', background: '#e53e3e', color: '#fff',
                border: 'none', borderRadius: '8px', fontWeight: 600,
                fontSize: '0.85rem', cursor: 'pointer'
              }}>⏹️ Stop</button>
            )}
          </div>
        </div>

        <div style={{ position: 'relative', background: '#000', borderRadius: '10px', overflow: 'hidden', marginBottom: '1rem' }}>
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: '100%', maxHeight: '400px', objectFit: 'cover', display: 'block' }} />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {isRunning && result && (
            <div style={{
              position: 'absolute', top: '12px', left: '12px',
              background: 'rgba(0,0,0,0.75)', borderRadius: '8px',
              padding: '8px 14px', backdropFilter: 'blur(4px)'
            }}>
              <div style={{
                fontWeight: 700, fontSize: '1rem',
                color: bad ? '#fc8181' : '#68d391',
                animation: bad ? 'blink 0.8s ease infinite' : 'none'
              }}>
                {bad ? '⚠️ DISTRACTED' : '✅ ATTENTIVE'} ({result.confidence}%)
              </div>
              {/* Show safety grade live on overlay */}
              {session && (
                <div style={{ fontSize: '0.72rem', color: '#e2e8f0', marginTop: '3px' }}>
                  Grade: {session.safety_grade} · Score: {session.safety_score}
                </div>
              )}
              {consecutiveDistracted >= 3 && (
                <div style={{ fontSize: '0.72rem', color: '#fbd38d', marginTop: '3px' }}>
                  ⚠️ Distracted for {consecutiveDistracted}s — Alert!
                </div>
              )}
            </div>
          )}

          {!isRunning && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
              justifyContent: 'center', flexDirection: 'column', gap: '8px'
            }}>
              <span style={{ fontSize: '3rem' }}>📷</span>
              <span style={{ color: '#aaa', fontSize: '0.9rem' }}>Camera not started</span>
            </div>
          )}
        </div>

        {error && (
          <div style={{ background: '#fff5f5', border: '1px solid #feb2b2', borderRadius: '8px', padding: '10px 14px', fontSize: '0.82rem', color: '#c53030' }}>
            {error}
          </div>
        )}
      </div>

      {/* Live Status */}
      {isRunning && result && (
        <div className="slide-in">
          {/* Status banner */}
          <div style={{
            background: bad ? '#fff5f5' : '#f0fff4', border: `1px solid ${bad ? '#feb2b2' : '#9ae6b4'}`,
            borderRadius: '10px', padding: '14px 18px', display: 'flex', alignItems: 'center', gap: '10px',
            marginBottom: '1.25rem', animation: bad ? 'blink 0.8s ease infinite' : 'none'
          }}>
            <span style={{ fontSize: '1.4rem' }}>{bad ? '⚠️' : '✅'}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: bad ? '#c53030' : '#276749' }}>
                {bad ? 'Driver is Distracted' : 'Driver is Attentive'}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Confidence: {result.confidence}%</div>
              {result.multistream && <MultiStreamTags ms={result.multistream} />}
            </div>
          </div>

          {/* Stats row — now includes Safety Grade and Fatigue from session */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
            {[
              { label: 'Frames',     value: frameCount,                  color: '#3b5bdb', bg: '#edf2ff', border: '#bac8ff' },
              { label: 'Attentive',  value: session ? `${100 - session.distracted_pct}%` : '—', color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' },
              { label: 'Distracted', value: session ? `${session.distracted_pct}%` : '—', color: bad ? '#c53030' : '#555', bg: bad ? '#fff5f5' : '#fafafa', border: bad ? '#feb2b2' : '#e8e8e8' },
            ].map(({ label, value, color, bg, border }) => (
              <div key={label} style={{ background: bg, border: `1px solid ${border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>{label}</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Safety Grade + Fatigue row */}
          {session && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
              {(() => {
                const g = gradeStyle(session.safety_grade);
                return (
                  <div style={{ background: g.bg, border: `1px solid ${g.border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                    <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Safety Grade</div>
                    <div style={{ fontSize: '2rem', fontWeight: 700, color: g.color, lineHeight: 1 }}>{session.safety_grade}</div>
                    <div style={{ fontSize: '0.7rem', color: g.color, marginTop: '4px' }}>{session.safety_message}</div>
                  </div>
                );
              })()}
              <div style={{ background: '#fafafa', border: '1px solid #e8e8e8', borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Safety Score</div>
                <div style={{ fontSize: '2rem', fontWeight: 700, color: '#222', lineHeight: 1 }}>{session.safety_score}</div>
                <div style={{ fontSize: '0.7rem', color: '#aaa', marginTop: '4px' }}>out of 100</div>
              </div>
              <div style={{ background: session.fatigue_events > 0 ? '#fffbeb' : '#f0fff4', border: `1px solid ${session.fatigue_events > 0 ? '#fcd34d' : '#9ae6b4'}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Fatigue Events</div>
                <div style={{ fontSize: '2rem', fontWeight: 700, color: session.fatigue_events > 0 ? '#d97706' : '#38a169', lineHeight: 1 }}>{session.fatigue_events}</div>
                <div style={{ fontSize: '0.7rem', color: '#aaa', marginTop: '4px' }}>this session</div>
              </div>
            </div>
          )}

          {/* Live chart */}
          {liveHistory.length > 2 && (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Live Confidence</p>
              <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>Real-time distraction probability</p>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={liveHistory} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#aaa' }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#aaa' }} />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '8px', fontSize: '0.78rem' }}
                    formatter={v => [`${v}%`, 'Distraction']} />
                  <ReferenceLine y={75} stroke="#e53e3e" strokeDasharray="4 3" label={{ value: 'Live threshold', fontSize: 9, fill: '#e53e3e' }} />
                  <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="4 3" />
                  <Line type="monotone" dataKey="confidence" stroke="#ff6600" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ================================
// MAIN APP
// ================================
export default function App() {
  const [mode, setMode] = useState('video');

  const tabs = [
    { key: 'image', label: '🖼️ Image' },
    { key: 'video', label: '🎬 Video' },
    { key: 'live',  label: '📷 Live'  },
  ];

  return (
    <div style={{ fontFamily: "'Segoe UI', Tahoma, sans-serif", background: '#f5f5f5', minHeight: '100vh', color: '#222' }}>
      <style>{css}</style>

      <div style={{ background: '#fff', borderBottom: '1px solid #e0e0e0', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '32px', height: '32px', background: '#ff6600', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 700, fontSize: '1rem' }}>A</div>
          <span style={{ fontWeight: 600, fontSize: '1rem', color: '#222' }}>ADAS Driver Monitor</span>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          {[['Binary', '97%'], ['Multiclass', '94%'], ['Video', '95%']].map(([k, v]) => (
            <div key={k} style={{ fontSize: '0.75rem', color: '#888', textAlign: 'center' }}>
              <span style={{ display: 'block', fontWeight: 600, color: '#ff6600' }}>{v}</span>
              {k}
            </div>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: '860px', margin: '0 auto', padding: '2rem 1.5rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: '#111', marginBottom: '6px' }}>Driver Distraction Detection</h1>
          <p style={{ color: '#666', fontSize: '0.9rem' }}>Analyze driver attention using MobileNetV2 — upload an image, video, or use live monitoring.</p>
        </div>

        <div style={{ display: 'flex', gap: '8px', marginBottom: '1.5rem', background: '#fff', padding: '6px', borderRadius: '10px', border: '1px solid #e8e8e8', width: 'fit-content' }}>
          {tabs.map(tab => (
            <button key={tab.key} onClick={() => setMode(tab.key)} style={{
              padding: '8px 20px',
              background: mode === tab.key ? '#ff6600' : 'transparent',
              color: mode === tab.key ? '#fff' : '#666',
              border: 'none', borderRadius: '8px',
              fontWeight: mode === tab.key ? 600 : 400,
              fontSize: '0.88rem', cursor: 'pointer', transition: 'all 0.2s'
            }}>
              {tab.label}
            </button>
          ))}
        </div>

        {mode === 'image' && <ImageMode />}
        {mode === 'video' && <VideoMode />}
        {mode === 'live'  && <LiveMode />}

        <div style={{ marginTop: '2.5rem', paddingTop: '1.25rem', borderTop: '1px solid #e8e8e8', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#bbb' }}>
          <span>ADAS Driver Distraction Detection</span>
          <span>MobileNetV2 · TensorFlow 2.13 · NST Augmentation</span>
        </div>
      </div>
    </div>
  );
}