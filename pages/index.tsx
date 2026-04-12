import Head from 'next/head';
import { useEffect, useRef, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';

// ─── Types ────────────────────────────────────────────────────────────────────
interface Dataset {
  id: number;
  name: string;
  uploaded_at: string;
  row_count: number;
}

interface BinSeries { x: number[]; y: number[] }

interface Bins {
  dileptonE: BinSeries;
  dileptonM: BinSeries;
  fourEE: BinSeries;
  fourMM: BinSeries;
  fourME: BinSeries;
  diphoton: BinSeries;
}

interface StatRow { range: string; events: number; mean: number | 'N/A' }

interface Stats {
  electrons: StatRow[];
  muons: StatRow[];
  photons: StatRow[];
}

type ViewMode = 'all' | 'dilepton' | 'fourlepton' | 'diphoton';
type DatasetMode = 'latest' | 'selected' | 'cumulative';

// ─── Helpers ──────────────────────────────────────────────────────────────────
const PLOT_CONFIG = { responsive: true, displayModeBar: true };

const COLORS = {
  electron: '#38bdf8',
  muon: '#f472b6',
  fourEE: '#818cf8',
  fourMM: '#fb923c',
  fourME: '#34d399',
  photon: '#facc15',
};

function buildPlotlyTrace(label: string, series: BinSeries, color: string, logX: boolean) {
  if (!series.x.length) return null;
  return {
    type: 'bar',
    name: label,
    x: series.x,
    y: series.y,
    marker: { color, opacity: 0.82 },
  };
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function Home() {
  // State
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [datasetMode, setDatasetMode] = useState<DatasetMode>('latest');
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const [numBins, setNumBins] = useState(50);
  const [logX, setLogX] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [bins, setBins] = useState<Bins | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [totalEvents, setTotalEvents] = useState(0);
  const [plotsReady, setPlotsReady] = useState(false);

  // Refs for plot divs
  const dileptonRef = useRef<HTMLDivElement>(null);
  const fourleptonRef = useRef<HTMLDivElement>(null);
  const diphotonRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Mounted guard — prevents server/client HTML mismatch (hydration error)
  const [mounted, setMounted] = useState(false);

  // Load Plotly dynamically (client-only)
  const [Plotly, setPlotly] = useState<any>(null);
  useEffect(() => {
    setMounted(true);
    import('plotly.js-dist-min').then((P) => setPlotly(P.default ?? P));
  }, []);

  // Fetch dataset list
  const fetchDatasets = useCallback(async () => {
    const r = await fetch('/api/datasets');
    const j = await r.json();
    setDatasets(j.datasets ?? []);
  }, []);

  useEffect(() => { fetchDatasets(); }, [fetchDatasets]);

  // ── Upload ──
  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const text = await file.text();
      const r = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, content: text }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error);
      await fetchDatasets();
      // Auto-select the newly uploaded dataset
      setSelectedIds([j.dataset.id]);
      setDatasetMode('latest');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  // ── Delete dataset ──
  async function handleDelete(id: number) {
    if (!confirm('Delete this dataset?')) return;
    await fetch(`/api/datasets/${id}`, { method: 'DELETE' });
    await fetchDatasets();
    setSelectedIds((prev) => prev.filter((x) => x !== id));
    setBins(null);
    setStats(null);
  }

  // ── Analyze ──
  async function handleAnalyze() {
    setLoading(true);
    setError('');
    setPlotsReady(false);
    try {
      let datasetIds: number[] | 'all';
      if (datasetMode === 'cumulative') {
        datasetIds = 'all';
      } else if (datasetMode === 'latest') {
        if (datasets.length === 0) throw new Error('No datasets available');
        datasetIds = [datasets[0].id];
      } else {
        if (selectedIds.length === 0) throw new Error('Select at least one dataset');
        datasetIds = selectedIds;
      }

      const r = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetIds, numBins }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error);
      setBins(j.bins);
      setStats(j.stats);
      setTotalEvents(j.totalEvents);
      setPlotsReady(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Render plots whenever bins/Plotly/viewMode/logX change ──
  useEffect(() => {
    if (!Plotly || !bins || !plotsReady) return;

    const xaxis: any = {
      title: 'Invariant Mass (GeV)',
      type: logX ? 'log' : 'linear',
      color: '#94a3b8',
      gridcolor: '#1e293b',
      zerolinecolor: '#334155',
    };
    const yaxis = {
      title: 'Events',
      color: '#94a3b8',
      gridcolor: '#1e293b',
      zerolinecolor: '#334155',
    };
    const layout = (title: string, barmode = 'overlay') => ({
      title: { text: title, font: { color: '#e2e8f0', size: 16, family: 'JetBrains Mono, monospace' } },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis,
      yaxis,
      barmode,
      legend: { font: { color: '#94a3b8' } },
      margin: { t: 50, r: 20, b: 60, l: 60 },
    });

    // Dilepton
    if ((viewMode === 'all' || viewMode === 'dilepton') && dileptonRef.current) {
      const traces = [
        buildPlotlyTrace('Electrons', bins.dileptonE, COLORS.electron, logX),
        buildPlotlyTrace('Muons', bins.dileptonM, COLORS.muon, logX),
      ].filter(Boolean);
      Plotly.react(dileptonRef.current, traces, layout('Dilepton Invariant Mass Distribution', 'stack'));
    }

    // Four lepton
    if ((viewMode === 'all' || viewMode === 'fourlepton') && fourleptonRef.current) {
      const traces = [
        buildPlotlyTrace('4e', bins.fourEE, COLORS.fourEE, logX),
        buildPlotlyTrace('4μ', bins.fourMM, COLORS.fourMM, logX),
        buildPlotlyTrace('2e2μ', bins.fourME, COLORS.fourME, logX),
      ].filter(Boolean);
      Plotly.react(fourleptonRef.current, traces, layout('Four-Lepton Invariant Mass Distribution'));
    }

    // Diphoton
    if ((viewMode === 'all' || viewMode === 'diphoton') && diphotonRef.current) {
      const traces = [
        buildPlotlyTrace('Photons', bins.diphoton, COLORS.photon, logX),
      ].filter(Boolean);
      Plotly.react(diphotonRef.current, traces, layout('Diphoton Invariant Mass Distribution'));
    }
  }, [Plotly, bins, plotsReady, viewMode, logX]);

  const show = (mode: ViewMode) => viewMode === 'all' || viewMode === mode;

  // ─── Render ────────────────────────────────────────────────────────────────
  // Suppress hydration mismatch: render nothing on server, full UI on client
  if (!mounted) return null;

  return (
    <>
      <Head>
        <title>HPlot — Invariant Mass Explorer</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap" rel="stylesheet" />
      </Head>

      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #050d1a;
          --panel: #0c1929;
          --border: #1e3a5f;
          --accent: #0ea5e9;
          --accent2: #f472b6;
          --text: #e2e8f0;
          --muted: #64748b;
          --success: #10b981;
          --danger: #ef4444;
          --mono: 'JetBrains Mono', monospace;
          --sans: 'Space Grotesk', sans-serif;
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: var(--sans);
          min-height: 100vh;
        }

        /* Grid background */
        body::before {
          content: '';
          position: fixed;
          inset: 0;
          background-image:
            linear-gradient(rgba(14, 165, 233, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(14, 165, 233, 0.04) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
          z-index: 0;
        }

        .page { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 24px 20px 60px; }

        /* ── Header ── */
        .header {
          display: flex;
          align-items: baseline;
          gap: 16px;
          margin-bottom: 36px;
          padding-bottom: 20px;
          border-bottom: 1px solid var(--border);
        }
        .header h1 {
          font-family: var(--mono);
          font-size: 2.4rem;
          font-weight: 700;
          letter-spacing: -2px;
          color: var(--accent);
          text-shadow: 0 0 40px rgba(14,165,233,0.4);
        }
        .header-tag {
          font-family: var(--mono);
          font-size: 0.75rem;
          color: var(--muted);
          letter-spacing: 2px;
          text-transform: uppercase;
        }

        /* ── Panel ── */
        .panel {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 24px;
          margin-bottom: 20px;
        }
        .panel-title {
          font-family: var(--mono);
          font-size: 0.7rem;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: var(--accent);
          margin-bottom: 16px;
        }

        /* ── Controls grid ── */
        .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }

        .ctrl label {
          display: block;
          font-family: var(--mono);
          font-size: 0.7rem;
          letter-spacing: 1px;
          text-transform: uppercase;
          color: var(--muted);
          margin-bottom: 6px;
        }

        select, input[type="number"] {
          width: 100%;
          background: #0a1628;
          border: 1px solid var(--border);
          border-radius: 6px;
          color: var(--text);
          padding: 9px 12px;
          font-family: var(--mono);
          font-size: 0.85rem;
          outline: none;
          transition: border-color .2s;
        }
        select:focus, input[type="number"]:focus { border-color: var(--accent); }

        .toggle-wrap { display: flex; align-items: center; gap: 10px; padding-top: 22px; }
        .toggle-wrap input { width: 18px; height: 18px; accent-color: var(--accent); cursor: pointer; }
        .toggle-wrap span { font-family: var(--mono); font-size: 0.8rem; color: var(--muted); }

        /* ── Buttons ── */
        .btn-primary {
          background: var(--accent);
          color: #000;
          border: none;
          border-radius: 8px;
          padding: 12px 28px;
          font-family: var(--mono);
          font-size: 0.85rem;
          font-weight: 700;
          letter-spacing: 1px;
          cursor: pointer;
          transition: box-shadow .2s, transform .1s;
        }
        .btn-primary:hover { box-shadow: 0 0 20px rgba(14,165,233,0.5); transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

        .btn-danger {
          background: transparent;
          color: var(--danger);
          border: 1px solid var(--danger);
          border-radius: 6px;
          padding: 4px 10px;
          font-family: var(--mono);
          font-size: 0.7rem;
          cursor: pointer;
          transition: background .2s;
        }
        .btn-danger:hover { background: rgba(239,68,68,0.1); }

        /* ── Upload area ── */
        .upload-zone {
          border: 2px dashed var(--border);
          border-radius: 10px;
          padding: 28px;
          text-align: center;
          cursor: pointer;
          transition: border-color .2s, background .2s;
          position: relative;
        }
        .upload-zone:hover { border-color: var(--accent); background: rgba(14,165,233,0.04); }
        .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .upload-zone p { font-family: var(--mono); font-size: 0.85rem; color: var(--muted); }
        .upload-zone .u-icon { font-size: 2rem; margin-bottom: 8px; }

        /* ── Dataset list ── */
        .dataset-list { display: flex; flex-direction: column; gap: 8px; max-height: 240px; overflow-y: auto; }
        .dataset-item {
          display: flex;
          align-items: center;
          gap: 12px;
          background: #0a1628;
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 10px 14px;
          cursor: pointer;
          transition: border-color .15s;
        }
        .dataset-item.active { border-color: var(--accent); }
        .dataset-item input[type="checkbox"] { accent-color: var(--accent); width: 16px; height: 16px; }
        .ds-name { font-family: var(--mono); font-size: 0.8rem; color: var(--text); flex: 1; }
        .ds-meta { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); }
        .ds-badge {
          background: rgba(14,165,233,0.15);
          color: var(--accent);
          border-radius: 4px;
          padding: 2px 7px;
          font-family: var(--mono);
          font-size: 0.65rem;
        }

        /* ── Mode pills ── */
        .pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
        .pill {
          background: transparent;
          border: 1px solid var(--border);
          border-radius: 100px;
          padding: 5px 14px;
          font-family: var(--mono);
          font-size: 0.72rem;
          letter-spacing: 1px;
          color: var(--muted);
          cursor: pointer;
          transition: all .15s;
        }
        .pill.active { background: var(--accent); border-color: var(--accent); color: #000; font-weight: 700; }
        .pill:hover:not(.active) { border-color: var(--accent); color: var(--accent); }

        /* ── Error ── */
        .error-box {
          background: rgba(239,68,68,0.1);
          border: 1px solid var(--danger);
          border-radius: 8px;
          padding: 14px 18px;
          font-family: var(--mono);
          font-size: 0.82rem;
          color: var(--danger);
          margin-top: 16px;
        }

        /* ── Spinner ── */
        .spinner {
          width: 36px; height: 36px;
          border: 3px solid var(--border);
          border-top-color: var(--accent);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          margin: 0 auto;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Plots ── */
        .plots-section { margin-top: 24px; }
        .plot-card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 12px;
          margin-bottom: 20px;
          overflow: hidden;
          padding: 8px;
        }

        /* ── Stats ── */
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-top: 24px; }
        .stat-card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 20px;
        }
        .stat-card h3 {
          font-family: var(--mono);
          font-size: 0.75rem;
          letter-spacing: 3px;
          text-transform: uppercase;
          margin-bottom: 14px;
        }
        .stat-card h3.e { color: ${COLORS.electron}; }
        .stat-card h3.m { color: ${COLORS.muon}; }
        .stat-card h3.g { color: ${COLORS.photon}; }
        table { width: 100%; border-collapse: collapse; }
        th, td {
          padding: 8px 10px;
          text-align: left;
          font-family: var(--mono);
          font-size: 0.75rem;
          border-bottom: 1px solid var(--border);
        }
        th { color: var(--muted); font-weight: 400; }
        td { color: var(--text); }
        tr:last-child td { border-bottom: none; }
        .num { text-align: right; }

        /* ── Total badge ── */
        .total-badge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: rgba(14,165,233,0.1);
          border: 1px solid var(--accent);
          border-radius: 8px;
          padding: 8px 16px;
          font-family: var(--mono);
          font-size: 0.8rem;
          color: var(--accent);
          margin-bottom: 20px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--panel); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

        @media (max-width: 700px) {
          .header h1 { font-size: 1.8rem; }
          .controls { grid-template-columns: 1fr 1fr; }
        }
      `}</style>

      <div className="page">
        {/* Header */}
        <div className="header">
          <h1>HPlot</h1>
          <span className="header-tag">Invariant Mass Explorer · LHC Open Data</span>
        </div>

        {/* Upload panel */}
        <div className="panel">
          <div className="panel-title">▸ Upload Dataset</div>
          <div className="upload-zone">
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.txt"
              onChange={handleUpload}
              disabled={uploading}
            />
            <div className="u-icon">⬆</div>
            <p>{uploading ? 'Uploading…' : 'Drop a .csv or .txt file here, or click to browse'}</p>
            <p style={{ fontSize: '0.7rem', marginTop: 4 }}>Format: invariant_mass, particle_type, combination</p>
          </div>
          {error && <div className="error-box">⚠ {error}</div>}
        </div>

        {/* Dataset selector */}
        <div className="panel">
          <div className="panel-title">▸ Datasets</div>
          <div className="pills">
            {(['latest','selected','cumulative'] as DatasetMode[]).map((m) => (
              <button key={m} className={`pill ${datasetMode === m ? 'active' : ''}`} onClick={() => setDatasetMode(m)}>
                {m === 'latest' ? '◉ Latest upload' : m === 'selected' ? '☑ Selected' : '∑ Cumulative (all)'}
              </button>
            ))}
          </div>

          {datasets.length === 0
            ? <p style={{ fontFamily: 'var(--mono)', fontSize: '0.8rem', color: 'var(--muted)' }}>No datasets yet — upload one above.</p>
            : <div className="dataset-list">
                {datasets.map((ds) => {
                  const isChecked = selectedIds.includes(ds.id);
                  return (
                    <div
                      key={ds.id}
                      className={`dataset-item ${isChecked && datasetMode === 'selected' ? 'active' : ''}`}
                      onClick={() => {
                        if (datasetMode !== 'selected') return;
                        setSelectedIds((prev) =>
                          prev.includes(ds.id) ? prev.filter((x) => x !== ds.id) : [...prev, ds.id]
                        );
                      }}
                    >
                      {datasetMode === 'selected' && (
                        <input type="checkbox" checked={isChecked} readOnly />
                      )}
                      <span className="ds-name">{ds.name}</span>
                      <span className="ds-badge">{ds.row_count} events</span>
                      <span className="ds-meta">{new Date(ds.uploaded_at).toLocaleDateString()}</span>
                      <button className="btn-danger" onClick={(e) => { e.stopPropagation(); handleDelete(ds.id); }}>✕</button>
                    </div>
                  );
                })}
              </div>
          }
        </div>

        {/* Analysis controls */}
        <div className="panel">
          <div className="panel-title">▸ Analysis Options</div>
          <div className="controls">
            <div className="ctrl">
              <label>View Mode</label>
              <select value={viewMode} onChange={(e) => setViewMode(e.target.value as ViewMode)}>
                <option value="all">All Plots</option>
                <option value="dilepton">Dilepton</option>
                <option value="fourlepton">Four-Lepton</option>
                <option value="diphoton">Diphoton</option>
              </select>
            </div>
            <div className="ctrl">
              <label>Number of Bins</label>
              <input
                type="number"
                value={numBins}
                min={5}
                max={500}
                step={5}
                onChange={(e) => setNumBins(Number(e.target.value))}
              />
            </div>
            <div className="ctrl">
              <div className="toggle-wrap">
                <input type="checkbox" id="logx" checked={logX} onChange={(e) => setLogX(e.target.checked)} />
                <span>Log X axis</span>
              </div>
            </div>
            <div className="ctrl" style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button
                className="btn-primary"
                onClick={handleAnalyze}
                disabled={loading || datasets.length === 0}
              >
                {loading ? 'Analyzing…' : '▶ Analyze'}
              </button>
            </div>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <div className="spinner" />
            <p style={{ fontFamily: 'var(--mono)', fontSize: '0.8rem', color: 'var(--muted)', marginTop: 14 }}>
              Processing events…
            </p>
          </div>
        )}

        {/* Plots */}
        {plotsReady && bins && (
          <div className="plots-section">
            <div className="total-badge">
              <span>∑</span>
              <span>{totalEvents.toLocaleString()} events analysed</span>
            </div>

            {show('dilepton') && (
              <div className="plot-card"><div ref={dileptonRef} style={{ height: 420 }} /></div>
            )}
            {show('fourlepton') && (
              <div className="plot-card"><div ref={fourleptonRef} style={{ height: 420 }} /></div>
            )}
            {show('diphoton') && (
              <div className="plot-card"><div ref={diphotonRef} style={{ height: 420 }} /></div>
            )}
          </div>
        )}

        {/* Statistics */}
        {plotsReady && stats && (
          <div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: '0.7rem', letterSpacing: '3px', color: 'var(--accent)', textTransform: 'uppercase', marginBottom: 16, marginTop: 8 }}>
              ▸ Statistical Summary by Energy Range
            </div>
            <div className="stats-grid">
              {([
                { key: 'electrons', label: 'Electrons', cls: 'e' },
                { key: 'muons',     label: 'Muons',     cls: 'm' },
                { key: 'photons',   label: 'Photons',   cls: 'g' },
              ] as const).map(({ key, label, cls }) => (
                <div key={key} className="stat-card">
                  <h3 className={cls}>{label}</h3>
                  <table>
                    <thead><tr><th>Range</th><th className="num">Events</th><th className="num">Mean (GeV)</th></tr></thead>
                    <tbody>
                      {(stats[key as keyof Stats] as StatRow[]).map((row) => (
                        <tr key={row.range}>
                          <td>{row.range}</td>
                          <td className="num">{row.events}</td>
                          <td className="num">{row.mean}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}