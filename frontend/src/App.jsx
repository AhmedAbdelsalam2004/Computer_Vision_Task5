import { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Activity, Cpu, Image as ImageIcon, CheckCircle, XCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [status, setStatus] = useState('offline');
  const [rocData, setRocData] = useState(null);
  const [eigenfaces, setEigenfaces] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  
  const fileInputRef = useRef(null);

  const fetchRoc = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/roc`);
      if (!res.data.error) {
        const { fpr, tpr, auc } = res.data;
        const formattedData = fpr.map((f, i) => ({
          fpr: f,
          tpr: tpr[i]
        }));
        setRocData({ data: formattedData, auc });
      }
    } catch {
      console.error("Error fetching ROC curve.");
    }
  }, []);

  const fetchEigenfaces = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/eigenfaces`);
      if (!res.data.error) {
        setEigenfaces(res.data);
      }
    } catch {
      console.error("Error fetching Eigenfaces.");
    }
  }, []);

  const checkServer = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/status`);
      setStatus('online');
      if (res.data.trained && !rocData) {
        fetchRoc();
      }
      if (res.data.trained && !eigenfaces) {
        fetchEigenfaces();
      }
    } catch {
      setStatus('offline');
    }
  }, [rocData, fetchRoc, eigenfaces, fetchEigenfaces]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    checkServer();
    const interval = setInterval(checkServer, 5000);
    return () => clearInterval(interval);
  }, [checkServer]);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setPreviewLoading(true);
      
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await axios.post(`${API_BASE}/preview`, formData);
        if (!res.data.error) {
          setPreviewUrl(`data:image/png;base64,${res.data.image_b64}`);
        }
      } catch {
        console.error("Preview error. Ensure backend is running latest code.");
      } finally {
        setPreviewLoading(false);
      }
    }
  };

  const handleRecognize = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await axios.post(`${API_BASE}/recognize`, formData);
      setResult(res.data);
    } catch {
      setResult({ error: "Failed to connect to server" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={24} color="var(--accent-color)" />
            FaceID Pro
          </h1>
          <p style={{ fontSize: '0.875rem', marginTop: '8px' }}>
            Eigenfaces Recognition System
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <h3 style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
            SYSTEM STATUS
          </h3>
          <div className={`status-badge ${status}`}>
            {status === 'online' ? <CheckCircle size={16} /> : <XCircle size={16} />}
            {status.toUpperCase()}
          </div>
        </div>
        {eigenfaces && (
          <div className="glass-panel" style={{ padding: '16px', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ImageIcon size={16} /> EXTRACTED EIGENFACES
            </h3>
            
            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ fontSize: '0.75rem', marginBottom: '8px', color: 'var(--text-secondary)' }}>Mean Face</h4>
              <img 
                src={`data:image/png;base64,${eigenfaces.mean_face}`} 
                alt="Mean Face" 
                style={{ width: '80px', height: '100px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--glass-border)' }} 
              />
            </div>
            
            <div style={{ flex: 1, overflowY: 'auto', paddingRight: '4px' }} className="custom-scrollbar">
              <h4 style={{ fontSize: '0.75rem', marginBottom: '8px', color: 'var(--text-secondary)' }}>Top Eigenfaces</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                {eigenfaces.eigenfaces.map((ef, idx) => (
                  <div key={idx} style={{ textAlign: 'center' }}>
                    <img 
                      src={`data:image/png;base64,${ef}`} 
                      alt={`Eigenface ${idx + 1}`} 
                      style={{ width: '100%', height: '100px', objectFit: 'cover', borderRadius: '8px', border: '1px solid var(--glass-border)' }} 
                    />
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '4px' }}>EF {idx + 1}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </aside>

      <main className="main-content">
        <section className="glass-panel" style={{ padding: '32px' }}>
          <h2><ImageIcon size={20} /> Face Recognition</h2>
          <p style={{ marginBottom: '24px' }}>Upload an image to detect and recognize the subject using custom PCA.</p>
          
          <div className="result-card">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div 
                className="upload-area"
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept="image/*" 
                  style={{ display: 'none' }} 
                />
                <Upload size={32} color="var(--text-secondary)" style={{ margin: '0 auto 12px' }} />
                <div style={{ fontWeight: 500 }}>Click to browse</div>
                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>JPG, PNG or PGM</div>
              </div>

              <button 
                className="glass-button" 
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={handleRecognize}
                disabled={!selectedFile || loading || status === 'offline'}
              >
                {loading ? 'Processing...' : 'Run Recognition'}
              </button>
            </div>

            <div className="glass-panel" style={{ padding: '24px', background: 'rgba(0,0,0,0.2)' }}>
              {result && !result.error && (
                <>
                  <div className="result-image-container">
                    <img 
                      src={`data:image/png;base64,${result.image_b64}`} 
                      alt="Result" 
                      className="result-image"
                    />
                  </div>
                  <div className="metric-grid">
                    <div className="metric-box">
                      <div className="metric-label">Predicted Subject</div>
                      <div className="metric-value">s{result.subject_id}</div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-label">Euclidean Distance</div>
                      <div className="metric-value">{result.distance.toFixed(2)}</div>
                    </div>
                  </div>
                </>
              )}
              {result && result.error && (
                <div style={{ color: 'var(--danger)', padding: '16px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>
                  {result.error}
                </div>
              )}
              {!result && previewUrl && !previewLoading && (
                <div className="result-image-container">
                  <img src={previewUrl} alt="Preview" className="result-image" />
                </div>
              )}
              {!result && previewLoading && (
                <div className="result-image-container" style={{ color: 'var(--text-secondary)' }}>
                  Loading preview...
                </div>
              )}
              {!result && !previewUrl && !previewLoading && (
                <div className="result-image-container" style={{ color: 'var(--text-secondary)' }}>
                  Awaiting image upload...
                </div>
              )}
            </div>
          </div>
        </section>



        {rocData && (
          <section className="glass-panel" style={{ padding: '32px' }}>
            <h2><Activity size={20} /> Performance Metrics (ROC Curve)</h2>
            <div className="metric-grid" style={{ marginBottom: '24px' }}>
              <div className="metric-box">
                <div className="metric-label">Area Under Curve (AUC)</div>
                <div className="metric-value" style={{ color: 'var(--accent-color)' }}>
                  {rocData.auc.toFixed(4)}
                </div>
              </div>
            </div>
            
            <div style={{ width: '100%', height: 400, background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rocData.data} margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="fpr" 
                    type="number" 
                    domain={[0, 1]} 
                    tick={{ fill: 'var(--text-secondary)' }}
                    label={{ value: 'False Positive Rate', position: 'bottom', fill: 'var(--text-secondary)', offset: 0 }}
                  />
                  <YAxis 
                    type="number" 
                    domain={[0, 1]} 
                    tick={{ fill: 'var(--text-secondary)' }}
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
                  />
                  <Tooltip 
                    contentStyle={{ background: 'var(--bg-color)', border: '1px solid var(--glass-border)', borderRadius: '8px' }}
                    formatter={(value) => value.toFixed(3)}
                  />
                  <Line 
                    type="stepAfter" 
                    dataKey="tpr" 
                    stroke="var(--accent-color)" 
                    strokeWidth={3} 
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line 
                    type="linear" 
                    dataKey="fpr" 
                    stroke="var(--text-secondary)" 
                    strokeWidth={2}
                    strokeDasharray="5 5" 
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
