import Head from 'next/head';
import { useEffect, useState } from 'react';

export default function About() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);
  if (!mounted) return null;

  return (
    <>
      <Head>
        <title>HPlot — About</title>
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
          --mono: 'JetBrains Mono', monospace;
          --sans: 'Space Grotesk', sans-serif;
        }

        body {
          background: var(--bg);
          color: var(--text);
          font-family: var(--sans);
          min-height: 100vh;
        }

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
          text-decoration: none;
        }
        .header h1 a { color: inherit; text-decoration: none; }
        .header-tag {
          font-family: var(--mono);
          font-size: 0.75rem;
          color: var(--muted);
          letter-spacing: 2px;
          text-transform: uppercase;
        }
        .nav-link {
          font-family: var(--mono);
          font-size: 0.72rem;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: var(--accent);
          text-decoration: none;
          border: 1px solid var(--accent);
          border-radius: 100px;
          padding: 5px 14px;
          margin-left: auto;
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

        /* ── About content ── */
        .about-text {
          font-family: var(--sans);
          font-size: 1rem;
          line-height: 1.8;
          color: var(--text);
          max-width: 720px;
        }
        .about-text .highlight {
          color: var(--accent);
          font-weight: 600;
        }
        .about-text .highlight2 {
          color: var(--accent2);
          font-weight: 600;
        }

        /* ── Footer ── */
        .footer {
          position: relative; z-index: 1;
          text-align: center;
          padding: 24px 20px;
          font-family: var(--mono);
          font-size: 0.7rem;
          color: var(--muted);
          border-top: 1px solid var(--border);
          letter-spacing: 1px;
        }

        @media (max-width: 700px) {
          .header h1 { font-size: 1.8rem; }
        }
      `}</style>

      <div className="page">
        {/* Header */}
        <div className="header">
          <h1><a href="/">HPlot</a></h1>
          <span className="header-tag">Invariant Mass Explorer · LHC Open Data</span>
          <a href="/" className="nav-link">← Back</a>
        </div>

        {/* About panel */}
        <div className="panel">
          <div className="panel-title">▸ About</div>
          <p className="about-text">
            <span className="highlight">HPlot</span> is a modern alternative of{' '}
            <span className="highlight2">OPlot</span>, which is the plotting-tool for hands-on
            CERN masterclasses, providing the same functions with a more responsive and
            user-friendly approach.
          </p>
        </div>
      </div>

      <footer className="footer">© 2026 HPlot</footer>
    </>
  );
}
