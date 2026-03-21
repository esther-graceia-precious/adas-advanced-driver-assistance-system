import { useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";

const API_URL = "http://127.0.0.1:5000";

const css = `
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
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

const riskStyle = (level) => {
  if (level === 'HIGH')   return { color: '#c53030', bg: '#fff5f5', border: '#feb2b2' };
  if (level === 'MEDIUM') return { color: '#d97706', bg: '#fffbeb', border: '#fcd34d' };
  return                         { color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' };
};

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFile = f => { setFile(f); setResult(null); setError(null); };
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

  return (
    <div style={{ fontFamily: "'Segoe UI', Tahoma, sans-serif", background: '#f5f5f5', minHeight: '100vh', color: '#222' }}>
      <style>{css}</style>

      {/* Navbar */}
      <div style={{ background: '#fff', borderBottom: '1px solid #e0e0e0', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '32px', height: '32px', background: '#ff6600', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 700, fontSize: '1rem' }}>A</div>
          <span style={{ fontWeight: 600, fontSize: '1rem', color: '#222' }}>ADAS Driver Monitor</span>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          {[['Binary', '97%'], ['Multiclass', '94%'], ['Video', '82%']].map(([k, v]) => (
            <div key={k} style={{ fontSize: '0.75rem', color: '#888', textAlign: 'center' }}>
              <span style={{ display: 'block', fontWeight: 600, color: '#ff6600' }}>{v}</span>
              {k}
            </div>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: '860px', margin: '0 auto', padding: '2rem 1.5rem' }}>

        {/* Title */}
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: '#111', marginBottom: '6px' }}>
            Driver Distraction Detection
          </h1>
          <p style={{ color: '#666', fontSize: '0.9rem' }}>
            Upload a driving video to analyze driver attention using MobileNetV2.
          </p>
        </div>

        {/* Upload card */}
        <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e8e8e8', padding: '1.5rem', marginBottom: '1.5rem' }}>
          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: '#555', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Upload Video
          </p>

          <label
            onDragOver={e => e.preventDefault()}
            onDrop={onDrop}
            style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              border: '2px dashed #ddd', borderRadius: '10px',
              padding: '2rem', cursor: 'pointer',
              background: file ? '#fff9f5' : '#fafafa',
              borderColor: file ? '#ff6600' : '#ddd',
              marginBottom: '1rem', transition: 'all 0.2s',
              gap: '8px'
            }}
          >
            <input type="file" accept="video/*" onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
            {file ? (
              <>
                <span style={{ fontSize: '2rem' }}>🎬</span>
                <span style={{ fontWeight: 600, color: '#ff6600', fontSize: '0.9rem' }}>{file.name}</span>
                <span style={{ color: '#aaa', fontSize: '0.8rem' }}>{(file.size / 1024 / 1024).toFixed(1)} MB · click to change</span>
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

          <button
            onClick={analyze}
            disabled={!file || loading}
            style={{
              width: '100%', padding: '12px',
              background: !file || loading ? '#f0f0f0' : '#ff6600',
              color: !file || loading ? '#aaa' : '#fff',
              border: 'none', borderRadius: '8px',
              fontWeight: 600, fontSize: '0.95rem',
              cursor: !file || loading ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s'
            }}
          >
            {loading ? 'Analyzing...' : 'Analyze Video'}
          </button>

          {error && <p style={{ color: '#e53e3e', fontSize: '0.82rem', marginTop: '10px' }}>{error}</p>}
        </div>

        {/* Results */}
        {result && (
          <div>

            {/* Status banner */}
            <div style={{
              background: bad ? '#fff5f5' : '#f0fff4',
              border: `1px solid ${bad ? '#feb2b2' : '#9ae6b4'}`,
              borderRadius: '10px', padding: '14px 18px',
              display: 'flex', alignItems: 'center', gap: '10px',
              marginBottom: '1.25rem',
              animation: bad ? 'blink 0.8s ease infinite' : 'none'
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

            {/* Stat cards — 4 columns now */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
              {[
                {
                  label: 'Attentive',
                  value: `${result.summary.attentive_pct}%`,
                  color: '#38a169', bg: '#f0fff4', border: '#9ae6b4'
                },
                {
                  label: 'Distracted',
                  value: `${result.summary.distracted_pct}%`,
                  color: bad ? '#c53030' : '#555',
                  bg: bad ? '#fff5f5' : '#fafafa',
                  border: bad ? '#feb2b2' : '#e8e8e8'
                },
                {
                  label: 'Alerts',
                  value: result.summary.alert_count,
                  color: '#d97706', bg: '#fffbeb', border: '#fcd34d'
                },
                {
                  label: 'Risk Level',
                  value: risk,
                  color: rs.color, bg: rs.bg, border: rs.border
                },
              ].map(({ label, value, color, bg, border }) => (
                <div key={label} style={{ background: bg, border: `1px solid ${border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>{label}</div>
                  <div style={{ fontSize: label === 'Risk Level' ? '1.1rem' : '2rem', fontWeight: 700, color, lineHeight: 1, wordBreak: 'break-word' }}>{value}</div>
                </div>
              ))}
            </div>

            {/* Chart */}
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Confidence Over Time</p>
              <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>Above 50% = Distracted · Below 50% = Attentive</p>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chart} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#aaa' }} interval={Math.floor((chart?.length || 1) / 7)} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#aaa' }} />
                  <Tooltip
                    contentStyle={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '8px', fontSize: '0.78rem' }}
                    formatter={v => [`${v}%`, 'Distraction']}
                  />
                  <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="4 3" />
                  <Line type="monotone" dataKey="confidence" stroke="#ff6600" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Alerts */}
            {result.alerts.length > 0 && (
              <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
                <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '1rem' }}>
                  Distraction Events ({result.alerts.length})
                </p>
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

            {/* Grad-CAM */}
            {result.gradcam && (
              <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
                <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Explainability — Grad-CAM</p>
                <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>
                  Most distracted frame ({result.gradcam.confidence}% confidence). Red areas show model focus.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <p style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px' }}>Original Frame</p>
                    <img src={`data:image/jpeg;base64,${result.gradcam.original}`} alt="Original"
                      style={{ width: '100%', borderRadius: '8px', border: '1px solid #e8e8e8' }} />
                  </div>
                  <div>
                    <p style={{ fontSize: '0.75rem', color: '#ff6600', marginBottom: '6px' }}>Grad-CAM Heatmap</p>
                    <img src={`data:image/jpeg;base64,${result.gradcam.gradcam}`} alt="Grad-CAM"
                      style={{ width: '100%', borderRadius: '8px', border: '1px solid #fcd34d' }} />
                  </div>
                </div>
              </div>
            )}

          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: '2.5rem', paddingTop: '1.25rem', borderTop: '1px solid #e8e8e8', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#bbb' }}>
          <span>ADAS Driver Distraction Detection</span>
          <span>MobileNetV2 · TensorFlow 2.13</span>
        </div>

      </div>
    </div>
  );
}