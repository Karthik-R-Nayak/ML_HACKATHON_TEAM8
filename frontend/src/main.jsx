import React from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const API = import.meta.env.VITE_API_BASE_URL || ''

function App() {
  const [sequence, setSequence] = React.useState('')
  const [alphabet, setAlphabet] = React.useState('any')
  const [result, setResult] = React.useState(null)
  const [error, setError] = React.useState('')
  const [busy, setBusy] = React.useState(false)
  const [ready, setReady] = React.useState(null)

  React.useEffect(() => {
    fetch(`${API}/v1/capabilities`).then(r => r.ok ? r.json() : null).then(setReady).catch(() => setReady(null))
  }, [])

  async function submit(event) {
    event.preventDefault()
    setBusy(true); setError(''); setResult(null)
    try {
      const response = await fetch(`${API}/v1/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sequence, alphabet }) })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail?.message || body.detail || `Request failed (${response.status})`)
      setResult(body)
    } catch (e) { setError(String(e.message || e)) }
    finally { setBusy(false) }
  }

  return <main className="shell">
    <header className="top"><a className="brand" href="/" aria-label="Sequence Workbench home"><span className="mark">S</span><span>sequence<span className="brand-light"> / workbench</span></span></a><div className="service"><i className={ready?.pipeline_ready ? 'on' : ''}/>{ready?.pipeline_ready ? 'Pipeline ready' : 'Adapters pending'}</div></header>
    <section className="intro"><div className="eyebrow">SEQUENCE ANALYSIS PLATFORM <span>•</span> API V1</div><h1>Explore a sequence.<br/><em>Trace every decision.</em></h1><p className="lede">Submit a sequence to the analysis pipeline. Each stage is modular, versioned, and visible.</p></section>
    <section className="workspace">
      <form className="panel input-panel" onSubmit={submit}>
        <div className="panel-head"><div><div className="step">01 / INPUT</div><h2>Sequence input</h2></div><span className="quiet">{sequence.replace(/\s/g, '').length.toLocaleString()} symbols</span></div>
        <label htmlFor="sequence">PASTE OR TYPE A SEQUENCE</label>
        <textarea id="sequence" value={sequence} onChange={e => setSequence(e.target.value)} placeholder="Paste sequence here…" maxLength={100000} required spellCheck="false" />
        <div className="row"><div><label htmlFor="alphabet">ALPHABET</label><select id="alphabet" value={alphabet} onChange={e => setAlphabet(e.target.value)}><option value="any">Auto / unspecified</option><option value="dna">DNA</option><option value="rna">RNA</option><option value="protein">Protein</option></select></div><span className="limit">Maximum 100,000 symbols</span></div>
        <button className="submit" disabled={busy || !sequence.trim()}>{busy ? 'Running pipeline…' : 'Run analysis'}<span>↗</span></button>
        <p className="fine">Analysis is unavailable until real adapters are connected. Test fixture mode is labelled in every response.</p>
      </form>
      <section className="panel result-panel"><div className="panel-head"><div><div className="step">02 / RESULTS</div><h2>Analysis output</h2></div><span className="quiet">{result ? 'COMPLETE' : 'AWAITING INPUT'}</span></div>
        {!result && !error && <div className="empty"><div className="orbit"><span>···</span></div><h3>Results will appear here</h3><p>Normalized sequence, candidate spans, scores, and decoded path.</p>{ready && !ready.pipeline_ready && <div className="pending">Adapter implementations are not registered yet.</div>}</div>}
        {error && <div className="error"><b>Analysis unavailable</b><p>{error}</p><small>Check API connectivity and adapter readiness.</small></div>}
        {result && <div className="results">{result.is_fixture && <div className="fixture">⚠ {result.notice}</div>}<div className="summary"><div><small>SEQUENCE LENGTH</small><b>{result.sequence_length}</b></div><div><small>CANDIDATES</small><b>{result.candidates.length}</b></div><div><small>MODEL</small><b>{result.model_name} <small>{result.model_version}</small></b></div></div><div className="result-block"><small>NORMALIZED SEQUENCE</small><code>{result.normalized_sequence}</code></div><div className="result-block"><small>DECODED PATH</small><code>{result.decoded_path.join(' · ')}</code></div><div className="result-block"><small>FEATURES</small><div className="chips">{result.feature_names.map(x => <span key={x}>{x}</span>)}</div></div></div>}
      </section>
    </section>
    <section className="pipeline"><div className="pipeline-title"><span className="step">PIPELINE STATUS</span><span className="quiet">IMPLEMENTATION INTERFACES</span></div><div className="stages">{[['01','Preprocessing','preprocessing'],['02','Candidate generation','candidate_generation'],['03','Feature extraction','feature_extraction'],['04','ML scoring','ml_model'],['05','HMM / Viterbi','hmm_viterbi']].map(([n,label,key],i) => <div className="stage" key={key}><span className="stage-num">{n}</span><b>{label}</b><span className={ready?.[key]?.ready ? 'stage-state ready' : 'stage-state'}>{ready?.[key]?.ready ? ready[key].version : 'Awaiting adapter'}</span>{i<4&&<span className="arrow">→</span>}</div>)}</div></section>
    <footer><span>SEQUENCE WORKBENCH <span className="brand-light">· Modular inference</span></span><span>API <a href={`${API}/docs`}>OpenAPI docs ↗</a></span></footer>
  </main>
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>)
