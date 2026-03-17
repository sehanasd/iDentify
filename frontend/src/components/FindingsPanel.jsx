import './FindingsPanel.css'

const STATUS_CONFIG = {
  idle       : { icon: '🔍', title: 'No image loaded',      sub: 'Upload an OPG to begin analysis.' },
  validating : { icon: '⏳', title: 'Validating image…',    sub: 'Checking image format and dimensions.' },
  ready      : { icon: '✅', title: 'Image ready',          sub: 'Click "Analyse OPG" to start the pipeline.' },
  analysing  : { icon: '🧠', title: 'Running AI pipeline…', sub: 'YOLO detection → EfficientNet classification → Grad-CAM.' },
  error      : { icon: '⚠', title: 'Analysis failed',      sub: 'See the error message on the left.' },
}

const SEVERITY_ORDER = ['infection', 'fractured', 'caries', 'bdc_bdr', 'impacted']

function confidenceColor(c) {
  if (c >= 0.85) return '#22c55e'
  if (c >= 0.70) return '#f59e0b'
  return '#ef4444'
}

export default function FindingsPanel({ status, result }) {
  if (status !== 'done') {
    const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.idle
    return (
      <div className="findings-panel findings-empty">
        <div className="fp-header"><h2>Findings</h2></div>
        <div className="fp-placeholder">
          <span className="fp-placeholder-icon">{cfg.icon}</span>
          <p className="fp-placeholder-title">{cfg.title}</p>
          <p className="fp-placeholder-sub">{cfg.sub}</p>
        </div>
        <div className="fp-pipeline-info">
          <h3>Detection Pipeline</h3>
          <div className="pipeline-step"><span className="ps-num">1</span><span>YOLOv11m — Object Detection</span></div>
          <div className="pipeline-step"><span className="ps-num">2</span><span>EfficientNet-B0 — Classification</span></div>
          <div className="pipeline-step"><span className="ps-num">3</span><span>Grad-CAM — XAI Heatmap</span></div>
        </div>
      </div>
    )
  }

  const { detections, total_found } = result
  const sorted = [...detections].sort((a, b) => {
    const ai = SEVERITY_ORDER.indexOf(a.class_name)
    const bi = SEVERITY_ORDER.indexOf(b.class_name)
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi)
  })
  const counts = {}
  detections.forEach(d => { counts[d.class_name] = (counts[d.class_name] || 0) + 1 })

  return (
    <div className="findings-panel">
      <div className="fp-header">
        <h2>Findings</h2>
        <span className="fp-badge">{total_found} detection{total_found !== 1 ? 's' : ''}</span>
      </div>

      {total_found === 0 ? (
        <div className="fp-placeholder">
          <span className="fp-placeholder-icon">✅</span>
          <p className="fp-placeholder-title">No pathologies detected</p>
          <p className="fp-placeholder-sub">The AI found no regions of concern in this OPG at the current detection threshold.</p>
        </div>
      ) : (
        <>
          <div className="fp-summary">
            {Object.entries(counts).map(([cls, count]) => {
              const d = sorted.find(x => x.class_name === cls)
              return (
                <div key={cls} className="fp-summary-badge" style={{ '--accent': d?.color_hex || '#4d9ef7' }}>
                  <span className="fp-sb-dot" style={{ background: d?.color_hex || '#4d9ef7' }} />
                  <span className="fp-sb-label">{d?.label || cls}</span>
                  <span className="fp-sb-count">{count}</span>
                </div>
              )
            })}
          </div>

          <div className="fp-list">
            {sorted.map((d, i) => (
              <div key={i} className="fp-item">
                <div className="fp-item-left">
                  <span className="fp-color-bar" style={{ background: d.color_hex }} />
                  <div>
                    <p className="fp-item-name">{d.label}</p>
                    <p className="fp-item-box">Box: [{d.box.join(', ')}]</p>
                  </div>
                </div>
                <div className="fp-item-conf" style={{ color: confidenceColor(d.confidence) }}>
                  {d.confidence_pct}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="fp-disclaimer">
        ⚠ For research and educational use only. Not a substitute for clinical diagnosis.
      </div>
    </div>
  )
}