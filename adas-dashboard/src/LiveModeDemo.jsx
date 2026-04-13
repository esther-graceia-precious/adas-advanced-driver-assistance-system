import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";

const API_URL = "http://localhost:5000";

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

// Reason tag gets its own color based on type
const reasonStyle = (reason) => {
  if (reason.includes('Phone') || reason.includes('\uD83D\uDCF1'))
    return { bg: '#fdf4ff', border: '#e9d5ff', color: '#7e22ce' };
  if (reason.includes('Fatigue') || reason.includes('Eyes Closed'))
    return { bg: '#fff7ed', border: '#fed7aa', color: '#c2410c' };
  if (reason.includes('Yawning'))
    return { bg: '#fffbeb', border: '#fcd34d', color: '#92400e' };
  if (reason.includes('Looking'))
    return { bg: '#eff6ff', border: '#bfdbfe', color: '#1d4ed8' };
  if (reason.includes('Drinking') || reason.includes('\uD83E\uDD64'))
    return { bg: '#f0fdf4', border: '#bbf7d0', color: '#166534' };
  return { bg: '#fff5f5', border: '#feb2b2', color: '#c53030' };
};

// ── MultiStream Tags ──────────────────────────────────────────────────
function MultiStreamTags({ ms }) {
  if (!ms) return null;
  const faceFound = ms.head && ms.head !== 'N/A';
  return (
    <div style={{ marginTop: '0.75rem', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
      {faceFound ? (
        <>
          <span style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#1d4ed8' }}>
            {'\uD83D\uDC64'} {ms.head}
          </span>
          <span style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#166534' }}>
            {'\uD83D\uDC41\uFE0F'} {ms.eye}
          </span>
          <span style={{ background: '#fdf4ff', border: '1px solid #e9d5ff', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#7e22ce' }}>
            {'\uD83D\uDC44'} {ms.mouth}
          </span>
        </>
      ) : (
        <span style={{ background: '#fff9f5', border: '1px solid #ff6600', borderRadius: '6px', padding: '3px 10px', fontSize: '0.72rem', color: '#ff6600' }}>
          Face not detected
        </span>
      )}
      {ms.reasons?.map((r, i) => {
        const rs = reasonStyle(r);
        return (
          <span key={i} style={{
            background: rs.bg, border: `1px solid ${rs.border}`,
            borderRadius: '6px', padding: '3px 10px',
            fontSize: '0.72rem', color: rs.color, fontWeight: 600
          }}>
            {'\u26A0\uFE0F'} {r}
          </span>
        );
      })}
    </div>
  );
}

// ── Video Editor ──────────────────────────────────────────────────────
function VideoEditor({ filename, duration, onTrimmed, onClose }) {
  const [segments,   setSegments]   = useState([]);
  const [start,      setStart]      = useState('');
  const [end,        setEnd]        = useState('');
  const [trimming,   setTrimming]   = useState(false);
  const [trimResult, setTrimResult] = useState(null);
  const [error,      setError]      = useState(null);

  const addSegment = () => {
    const s = parseFloat(start), e = parseFloat(end);
    if (isNaN(s) || isNaN(e) || s >= e || s < 0 || e > duration) {
      setError(`Invalid range. Enter seconds between 0 and ${duration}.`);
      return;
    }
    setError(null);
    setSegments(prev => [...prev, [s, e]].sort((a, b) => a[0] - b[0]));
    setStart(''); setEnd('');
  };

  const removeSegment = (idx) => setSegments(prev => prev.filter((_, i) => i !== idx));

  const doTrim = async () => {
    if (!segments.length) { setError('Add at least one segment to keep.'); return; }
    setTrimming(true); setError(null);
    try {
      const res = await axios.post(`${API_URL}/trim_video`, { filename, keep_segments: segments });
      setTrimResult(res.data);
      onTrimmed && onTrimmed(res.data.trimmed_filename);
    } catch {
      setError('Trim failed. Check backend.');
    } finally { setTrimming(false); }
  };

  return (
    <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
        <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', margin: 0 }}>
          {'\u2702\uFE0F'} Video Editor — {filename}
        </p>
        {onClose && (
          <button onClick={onClose}
            style={{ background: 'none', border: 'none', color: '#aaa', fontSize: '1.2rem', cursor: 'pointer', lineHeight: 1, padding: '0 4px' }}>
            {'\u00D7'}
          </button>
        )}
      </div>
      <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1rem' }}>
        Total duration: {duration}s — Add segments to <strong>keep</strong>, then trim.
      </p>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <input value={start} onChange={e => setStart(e.target.value)} placeholder="Start (s)"
          style={{ width: '90px', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '0.82rem' }} />
        <span style={{ color: '#aaa', fontSize: '0.8rem' }}>{'\u2192'}</span>
        <input value={end} onChange={e => setEnd(e.target.value)} placeholder="End (s)"
          style={{ width: '90px', padding: '6px 10px', border: '1px solid #ddd', borderRadius: '6px', fontSize: '0.82rem' }} />
        <button onClick={addSegment}
          style={{ padding: '6px 16px', background: '#3b5bdb', color: '#fff', border: 'none', borderRadius: '6px', fontSize: '0.82rem', cursor: 'pointer' }}>
          + Add
        </button>
        <button onClick={() => setSegments([[0, duration]])}
          style={{ padding: '6px 12px', background: '#f0f0f0', color: '#555', border: '1px solid #ddd', borderRadius: '6px', fontSize: '0.78rem', cursor: 'pointer' }}>
          Keep All
        </button>
      </div>
      {error && <p style={{ color: '#c53030', fontSize: '0.78rem', marginBottom: '8px' }}>{error}</p>}
      {segments.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Segments to keep:</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {segments.map(([s, e], i) => (
              <div key={i} style={{ background: '#f0fff4', border: '1px solid #9ae6b4', borderRadius: '6px', padding: '4px 10px', fontSize: '0.78rem', color: '#276749', display: 'flex', alignItems: 'center', gap: '6px' }}>
                {s}s {'\u2192'} {e}s
                <button onClick={() => removeSegment(i)}
                  style={{ background: 'none', border: 'none', color: '#c53030', cursor: 'pointer', fontSize: '0.85rem', padding: 0, lineHeight: 1 }}>
                  {'\u00D7'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      <button onClick={doTrim} disabled={trimming || !segments.length}
        style={{ padding: '8px 20px', background: trimming || !segments.length ? '#f0f0f0' : '#ff6600', color: trimming || !segments.length ? '#aaa' : '#fff', border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.85rem', cursor: trimming || !segments.length ? 'not-allowed' : 'pointer' }}>
        {trimming ? 'Trimming...' : 'Trim Video'}
      </button>
      {trimResult && (
        <div style={{ marginTop: '1rem', background: '#f0fff4', border: '1px solid #9ae6b4', borderRadius: '8px', padding: '10px 14px' }}>
          <p style={{ fontSize: '0.82rem', color: '#276749', fontWeight: 600, marginBottom: '6px' }}>Trimmed successfully!</p>
          <a href={`${API_URL}${trimResult.download_url}`} target="_blank" rel="noreferrer"
            style={{ display: 'inline-block', padding: '6px 16px', background: '#276749', color: '#fff', borderRadius: '6px', fontSize: '0.82rem', textDecoration: 'none' }}>
            Download Trimmed Video
          </a>
        </div>
      )}
    </div>
  );
}

// ── Session Report Card ───────────────────────────────────────────────
function SessionReportCard({ summary, safety, fatigue, gradcam }) {
  const gs = gradeStyle(safety.grade);
  const rs = riskStyle(summary.risk_level);

  return (
    <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.5rem', marginBottom: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#111', marginBottom: '4px' }}>Session Report</h2>
          <p style={{ fontSize: '0.78rem', color: '#aaa' }}>{summary.total_frames_analyzed} frames analyzed</p>
        </div>
      </div>

      <div style={{ background: gs.bg, border: `2px solid ${gs.border}`, borderRadius: '12px', padding: '1.5rem', textAlign: 'center', marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '6px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Safety Grade</div>
        <div style={{ fontSize: '3.5rem', fontWeight: 800, color: gs.color, lineHeight: 1 }}>{safety.grade}</div>
        <div style={{ fontSize: '0.85rem', color: gs.color, marginTop: '6px', fontWeight: 600 }}>{safety.message}</div>
        <div style={{ fontSize: '0.78rem', color: '#888', marginTop: '4px' }}>Score: {safety.score} / 100</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1.25rem' }}>
        {[
          { label: 'Attentive',  value: `${summary.attentive_pct}%`,  color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' },
          { label: 'Distracted', value: `${summary.distracted_pct}%`, color: '#c53030', bg: '#fff5f5', border: '#feb2b2' },
          { label: 'Risk Level', value: summary.risk_level,            color: rs.color,  bg: rs.bg,    border: rs.border },
        ].map(({ label, value, color, bg, border }) => (
          <div key={label} style={{ background: bg, border: `1px solid ${border}`, borderRadius: '10px', padding: '0.9rem 1rem' }}>
            <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px', fontWeight: 500 }}>{label}</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ background: fatigue.event_count > 0 ? '#fffbeb' : '#f0fff4', border: `1px solid ${fatigue.event_count > 0 ? '#fcd34d' : '#9ae6b4'}`, borderRadius: '10px', padding: '1rem 1.25rem', marginBottom: '1.25rem' }}>
        <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Fatigue Events</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
          <span style={{ fontSize: '2rem', fontWeight: 700, color: fatigue.event_count > 0 ? '#d97706' : '#38a169' }}>{fatigue.event_count}</span>
          <span style={{ fontSize: '0.8rem', color: '#888' }}>
            {fatigue.event_count === 0 ? 'No fatigue detected — great session!' : `events · longest streak: ${fatigue.max_duration_frames} frames`}
          </span>
        </div>
      </div>

      {gradcam && (
        <div style={{ marginBottom: '1.25rem' }}>
          <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Grad-CAM — Most Distracted Moment</p>
          <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '0.75rem' }}>{gradcam.confidence}% confidence. Red = model focus.</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div>
              <p style={{ fontSize: '0.72rem', color: '#888', marginBottom: '5px' }}>Original</p>
              <img src={`data:image/jpeg;base64,${gradcam.original}`} alt="Original" style={{ width: '100%', borderRadius: '8px', border: '1px solid #e8e8e8' }} />
            </div>
            <div>
              <p style={{ fontSize: '0.72rem', color: '#ff6600', marginBottom: '5px' }}>Heatmap</p>
              <img src={`data:image/jpeg;base64,${gradcam.gradcam}`} alt="Grad-CAM" style={{ width: '100%', borderRadius: '8px', border: '1px solid #fcd34d' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ================================
// MAIN LiveModeDemo
// ================================
export default function LiveModeDemo() {
  const videoRef            = useRef(null);
  const canvasRef           = useRef(null);
  const streamRef           = useRef(null);
  const intervalRef         = useRef(null);
  const sessionRef          = useRef(null);
  const recordingSessionRef = useRef(null);

  const [isRunning,             setIsRunning]             = useState(false);
  const [result,                setResult]                = useState(null);
  const [error,                 setError]                 = useState(null);
  const [frameCount,            setFrameCount]            = useState(0);
  const [consecutiveDistracted, setConsecutiveDistracted] = useState(0);
  const [liveHistory,           setLiveHistory]           = useState([]);
  const [sessionSummary,        setSessionSummary]        = useState(null);

  // Recording — fully independent of camera lifecycle
  const [isRecording,     setIsRecording]     = useState(false);
  const [recError,        setRecError]        = useState(null);

  // Recordings library — persists across sessions, capped at 10
  const [recordings,      setRecordings]      = useState([]);
  const [selectedForTrim, setSelectedForTrim] = useState(null);

  // ── Stop Recording ────────────────────────────────────────────────
  const stopRecording = useCallback(async () => {
    const sid = recordingSessionRef.current;
    if (!sid) {
      setRecError('No session ID found. Was recording started?');
      return;
    }
    setRecError(null);
    try {
      const res = await axios.post(`${API_URL}/recording_stop`, { session_id: sid });
      if (res.data && res.data.video_filename) {
        setIsRecording(false);
        // Capture sid here before any possible cleanup
        const entry = {
          filename:  res.data.video_filename,
          duration:  res.data.duration_seconds,
          sessionId: sid,
          timestamp: new Date().toLocaleTimeString(),
        };
        setRecordings(prev => [entry, ...prev].slice(0, 10));
      }
    } catch(e) {
      console.error('recording_stop error:', e);
      setRecError('Recording stop failed. Check backend.');
    }
  }, []); // stable — no camera deps

  // ── Stop Camera (does NOT touch recording) ────────────────────────
  const stopCamera = useCallback(async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    const sessionToEnd = sessionRef.current;
    sessionRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsRunning(false);

    await new Promise(r => setTimeout(r, 600));

    if (sessionToEnd) {
      try {
        const res = await axios.post(`${API_URL}/live_end`, { session_id: sessionToEnd });
        if (!res.data.stop) setSessionSummary(res.data);
      } catch(e) {
        console.error('live_end failed:', e);
      }
    }
  }, []); // stable

  // ── Capture & Analyze ─────────────────────────────────────────────
  const captureAndAnalyze = useCallback(async () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !sessionRef.current) return;

    const currentSession = sessionRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

    try {
      const res = await axios.post(`${API_URL}/analyze_live`, {
        session_id: currentSession,
        image: base64
      });
      const data = res.data;
      if (data.stop) return;

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
        const updated = [...prev, { time: `${prev.length + 1}`, confidence: data.confidence }];
        return updated.slice(-30);
      });
    } catch(e) {
      console.error('analyze_live error:', e);
    }
  }, []);

  // ── Start Camera ──────────────────────────────────────────────────
  const startCamera = async () => {
    // Clear per-session state but NOT recordings library
    setError(null); setResult(null); setFrameCount(0);
    setConsecutiveDistracted(0); setLiveHistory([]);
    setSessionSummary(null); setRecError(null);

    try {
      const res = await axios.post(`${API_URL}/live_start`);
      sessionRef.current = res.data.session_id;
      recordingSessionRef.current = res.data.session_id;
    } catch(e) {
      setError('Could not start session. Make sure Flask is running.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play(); }
      setIsRunning(true);
      intervalRef.current = setInterval(captureAndAnalyze, 1000);
    } catch(e) {
      setError('Camera access denied. Please allow camera permissions and try again.');
      sessionRef.current = null;
      recordingSessionRef.current = null;
    }
  };

  // ── Start Recording ───────────────────────────────────────────────
  const startRecording = async () => {
    const sid = sessionRef.current;
    if (!sid) {
      setRecError('No active session. Start camera first.');
      return;
    }
    if (isRecording) {
      setRecError('Already recording. Click "Stop Rec" first.');
      return;
    }
    setRecError(null);
    try {
      const res = await axios.post(`${API_URL}/recording_start`, { session_id: sid });
      if (res.status === 200 || res.status === 201) {
        setIsRecording(true);
        recordingSessionRef.current = sid;
      }
    } catch(e) {
      console.error('recording_start error:', e);
      setRecError(`Recording start failed: ${e.response?.data?.error || e.message}`);
    }
  };

  useEffect(() => { return () => stopCamera(); }, [stopCamera]);

  const bad     = result?.label === 'Distracted';
  const session = result?.session;
  const gs      = session ? gradeStyle(session.safety_grade) : null;

  return (
    <div>
      {/* ── Camera Card ── */}
      <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e8e8e8', padding: '1.4rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '2px' }}>Live Camera Feed</p>
            <p style={{ fontSize: '0.75rem', color: '#aaa' }}>
              {isRunning ? `Analyzing · ${frameCount} frames` : 'Click Start to begin'}
              {isRecording && <span style={{ color: '#e53e3e', fontWeight: 600, marginLeft: '8px' }}>{'\u25CF'} REC</span>}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {!isRunning ? (
              <button onClick={startCamera}
                style={{ padding: '8px 20px', background: '#ff6600', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer' }}>
                Start
              </button>
            ) : (
              <>
                <button onClick={stopCamera}
                  style={{ padding: '8px 20px', background: '#e53e3e', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer' }}>
                  Stop{isRecording && <span style={{ fontSize: '0.7rem', opacity: 0.8, marginLeft: '4px' }}>(stop rec first)</span>}
                </button>
                {!isRecording ? (
                  <button onClick={startRecording}
                    style={{ padding: '8px 16px', background: '#c53030', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer' }}>
                    {'\u23FA'} Record
                  </button>
                ) : (
                  <button onClick={stopRecording}
                    style={{ padding: '8px 16px', background: '#744210', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer' }}>
                    Stop Rec
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        <div style={{ position: 'relative', background: '#000', borderRadius: '10px', overflow: 'hidden', marginBottom: '1rem' }}>
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: '100%', maxHeight: '400px', objectFit: 'cover', display: 'block' }} />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {isRunning && result && (
            <div style={{ position: 'absolute', top: '12px', left: '12px', background: 'rgba(0,0,0,0.75)', borderRadius: '8px', padding: '8px 14px', backdropFilter: 'blur(4px)' }}>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: bad ? '#fc8181' : '#68d391' }}>
                {bad ? 'DISTRACTED' : 'ATTENTIVE'} ({result.confidence}%)
              </div>
              {session && (
                <div style={{ fontSize: '0.72rem', color: '#e2e8f0', marginTop: '3px' }}>
                  Grade: {session.safety_grade} · Score: {session.safety_score}
                </div>
              )}
              {isRecording && (
                <div style={{ fontSize: '0.72rem', color: '#fc8181', marginTop: '3px', fontWeight: 600 }}>{'\u25CF'} RECORDING</div>
              )}
              {consecutiveDistracted >= 3 && (
                <div style={{ fontSize: '0.72rem', color: '#fbd38d', marginTop: '3px' }}>Distracted for {consecutiveDistracted}s</div>
              )}
            </div>
          )}

          {!isRunning && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '8px' }}>
              <span style={{ fontSize: '3rem' }}>📷</span>
              <span style={{ color: '#aaa', fontSize: '0.9rem' }}>Camera not started</span>
            </div>
          )}
        </div>

        {error    && <div style={{ background: '#fff5f5', border: '1px solid #feb2b2', borderRadius: '8px', padding: '10px 14px', fontSize: '0.82rem', color: '#c53030' }}>{error}</div>}
        {recError && <div style={{ background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: '8px', padding: '10px 14px', fontSize: '0.82rem', color: '#92400e', marginTop: '6px' }}>{recError}</div>}
      </div>

      {/* ── Live Status ── */}
      {isRunning && result && (
        <>
          <div style={{ background: bad ? '#fff5f5' : '#f0fff4', border: `1px solid ${bad ? '#feb2b2' : '#9ae6b4'}`, borderRadius: '10px', padding: '14px 18px', display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
            <span style={{ fontSize: '1.4rem' }}>{bad ? '\u26A0\uFE0F' : '\u2705'}</span>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: '1rem', color: bad ? '#c53030' : '#276749' }}>
                {bad ? 'Driver is Distracted' : 'Driver is Attentive'}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Confidence: {result.confidence}%</div>
              {result.multistream && <MultiStreamTags ms={result.multistream} />}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
            {[
              { label: 'Frames',     value: frameCount,                                         color: '#3b5bdb', bg: '#edf2ff', border: '#bac8ff' },
              { label: 'Attentive',  value: session ? `${100 - session.distracted_pct}%` : '\u2014', color: '#38a169', bg: '#f0fff4', border: '#9ae6b4' },
              { label: 'Distracted', value: session ? `${session.distracted_pct}%` : '\u2014',       color: bad ? '#c53030' : '#555', bg: bad ? '#fff5f5' : '#fafafa', border: bad ? '#feb2b2' : '#e8e8e8' },
            ].map(({ label, value, color, bg, border }) => (
              <div key={label} style={{ background: bg, border: `1px solid ${border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>{label}</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
              </div>
            ))}
          </div>

          {session && gs && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ background: gs.bg, border: `1px solid ${gs.border}`, borderRadius: '10px', padding: '1.1rem 1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '6px', fontWeight: 500 }}>Safety Grade</div>
                <div style={{ fontSize: '2rem', fontWeight: 700, color: gs.color, lineHeight: 1 }}>{session.safety_grade}</div>
                <div style={{ fontSize: '0.7rem', color: gs.color, marginTop: '4px' }}>{session.safety_message}</div>
              </div>
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

          {liveHistory.length > 2 && (
            <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
              <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', marginBottom: '4px' }}>Live Confidence</p>
              <p style={{ fontSize: '0.78rem', color: '#aaa', marginBottom: '1.1rem' }}>Real-time distraction probability</p>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={liveHistory} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#aaa' }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#aaa' }} />
                  <Tooltip
                    contentStyle={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '8px', fontSize: '0.78rem' }}
                    formatter={v => [`${v}%`, 'Distraction']}
                  />
                  <ReferenceLine y={75} stroke="#e53e3e" strokeDasharray="4 3" label={{ value: 'threshold', fontSize: 9, fill: '#e53e3e' }} />
                  <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="4 3" />
                  <Line type="monotone" dataKey="confidence" stroke="#ff6600" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* ── Recordings Library — always visible, survives new sessions ── */}
      {recordings.length > 0 && (
        <div style={{ background: '#fff', border: '1px solid #e8e8e8', borderRadius: '12px', padding: '1.4rem', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <p style={{ fontWeight: 600, fontSize: '0.85rem', color: '#444', margin: 0 }}>
              Recordings ({recordings.length}/10)
            </p>
            <span style={{ fontSize: '0.72rem', color: '#bbb' }}>Oldest auto-removed after 10</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {recordings.map((rec, i) => (
              <div key={`${rec.filename}-${i}`} style={{
                background: selectedForTrim?.filename === rec.filename ? '#fff9f5' : '#fafafa',
                border: `1px solid ${selectedForTrim?.filename === rec.filename ? '#ff6600' : '#e8e8e8'}`,
                borderRadius: '8px', padding: '10px 14px',
                display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap'
              }}>
                <div style={{ flex: 1, minWidth: '120px' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#333' }}>
                    #{recordings.length - i} · {rec.filename}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#aaa', marginTop: '2px' }}>
                    {rec.duration}s · {rec.timestamp}
                  </div>
                </div>
                <a
                  href={`${API_URL}/download_video/${rec.filename}`}
                  target="_blank" rel="noreferrer"
                  style={{ padding: '4px 12px', background: '#3b5bdb', color: '#fff', borderRadius: '6px', fontSize: '0.78rem', textDecoration: 'none', whiteSpace: 'nowrap' }}
                >
                  Download Video
                </a>
                {rec.sessionId && (
                  <a
                    href={`${API_URL}/download_report/${rec.sessionId}`}
                    target="_blank" rel="noreferrer"
                    style={{ padding: '4px 12px', background: '#ff6600', color: '#fff', borderRadius: '6px', fontSize: '0.78rem', textDecoration: 'none', whiteSpace: 'nowrap' }}
                  >
                    PDF Report
                  </a>
                )}
                <button
                  onClick={() => setSelectedForTrim(
                    selectedForTrim?.filename === rec.filename ? null : rec
                  )}
                  style={{
                    padding: '4px 12px',
                    background: selectedForTrim?.filename === rec.filename ? '#ff6600' : '#f0f0f0',
                    color:      selectedForTrim?.filename === rec.filename ? '#fff' : '#555',
                    border: '1px solid #ddd', borderRadius: '6px',
                    fontSize: '0.78rem', cursor: 'pointer', whiteSpace: 'nowrap'
                  }}
                >
                  Trim
                </button>
                <button
                  onClick={() => {
                    if (selectedForTrim?.filename === rec.filename) setSelectedForTrim(null);
                    setRecordings(prev => prev.filter(r => r.filename !== rec.filename));
                  }}
                  style={{ padding: '4px 10px', background: 'none', border: '1px solid #e0e0e0', borderRadius: '6px', fontSize: '0.78rem', color: '#bbb', cursor: 'pointer' }}
                >
                  {'\u00D7'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Video Editor — shown when a recording row's Trim is clicked ── */}
      {selectedForTrim && (
        <VideoEditor
          filename={selectedForTrim.filename}
          duration={selectedForTrim.duration}
          onTrimmed={(newFilename) => {
            // Update the filename in the library
            setRecordings(prev => prev.map(r =>
              r.filename === selectedForTrim.filename
                ? { ...r, filename: newFilename }
                : r
            ));
            setSelectedForTrim(null);
          }}
          onClose={() => setSelectedForTrim(null)}
        />
      )}

      {/* ── Session Report — shown after session ends ── */}
      {sessionSummary && !isRunning && (
        <SessionReportCard
          summary={sessionSummary.summary}
          safety={sessionSummary.safety}
          fatigue={sessionSummary.fatigue}
          gradcam={sessionSummary.gradcam}
        />
      )}
    </div>
  );
}