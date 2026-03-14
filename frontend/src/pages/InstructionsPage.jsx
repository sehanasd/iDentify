import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import './InstructionsPage.css'

const STEPS = [
  {
    n: '01', title: 'Prepare your OPG image',
    points: [
      'Use a standard panoramic dental radiograph (OPG / orthopantomogram)',
      'Accepted formats: JPG, JPEG, PNG, BMP, TIFF',
      'Minimum resolution: 400 × 200 pixels',
      'The image must be landscape orientation (wider than tall)',
      'The image should be greyscale or near-greyscale — colour photographs will be rejected',
    ],
  },
  {
    n: '02', title: 'Upload the image',
    points: [
      'Navigate to the Analyse page',
      'Drag and drop your OPG into the upload area, or click to browse',
      'The system will automatically validate the image before analysis',
      'If the image fails validation, a clear reason will be shown',
    ],
  },
  {
    n: '03', title: 'Run the analysis',
    points: [
      'Click the "Analyse OPG" button to start',
      'The AI model scans the full radiograph and marks all detected regions',
      'Each region is classified into one of five pathology categories',
      'Processing typically takes a few seconds',
    ],
  },
  {
    n: '04', title: 'Review the results',
    points: [
      'The annotated OPG is displayed with colour-coded boxes around each finding',
      'Toggle to the "XAI Heatmap" view to see which areas of the radiograph influenced each finding',
      'The Findings Panel lists every detected pathology with its confidence level',
      'Use the download buttons to save the annotated image, heatmap view, or a text report',
    ],
  },
]

const CLASSES = [
  { name: 'Dental Caries',          hex: '#FFA500', note: 'Tooth decay' },
  { name: 'Periapical Infection',   hex: '#FF4444', note: 'Infection or bone loss at root tip' },
  { name: 'Impacted Tooth',         hex: '#4d9ef7', note: 'Unerupted or misaligned tooth' },
  { name: 'Fractured Tooth',        hex: '#FFFF44', note: 'Crack or structural break' },
  { name: 'Broken Down Crown/Root', hex: '#FF44FF', note: 'Severely decayed or retained root' },
]

const NOTES = [
  'iDentify is a decision-support tool and does not replace clinical diagnosis by a qualified dental professional.',
  'The system is trained on panoramic radiographs only. Bitewing, periapical, or CBCT images are not supported.',
  'Only the five pathology classes listed above are detectable. Other conditions will not be flagged.',
  'Prediction confidence scores are shown for each detection — low-confidence findings should be interpreted with care.',
]

export default function InstructionsPage() {
  useEffect(() => { window.scrollTo(0, 0) }, [])

  return (
    <div className="instructions-page page">
      <div className="instr-inner">

        <div className="instr-header">
          <div className="hero-badge">User Guide</div>
          <h1>How to Use iDentify</h1>
          <p>Follow the steps below to upload an OPG and receive an AI-assisted pathology analysis.</p>
        </div>

        <div className="instr-steps">
          {STEPS.map(s => (
            <div key={s.n} className="instr-step">
              <div className="instr-step-num">{s.n}</div>
              <div className="instr-step-body">
                <h2>{s.title}</h2>
                <ul>
                  {s.points.map(p => <li key={p}>{p}</li>)}
                </ul>
              </div>
            </div>
          ))}
        </div>

        <div className="instr-section">
          <h2>Detection Colour Legend</h2>
          <p>Each pathology class is assigned a unique bounding-box colour for easy identification.</p>
          <div className="legend-grid">
            {CLASSES.map(c => (
              <div key={c.name} className="legend-item">
                <span className="legend-dot" style={{ background: c.hex }} />
                <div>
                  <span className="legend-name">{c.name}</span>
                  <span className="legend-note">{c.note}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="instr-section instr-notes">
          <h2>⚠ Important Notes</h2>
          <ul>
            {NOTES.map(n => <li key={n}>{n}</li>)}
          </ul>
        </div>

        <div className="instr-cta">
          <Link to="/predict" className="btn btn-primary" style={{ fontSize: '16px', padding: '14px 36px' }}>
            Start Analysis →
          </Link>
        </div>

      </div>
    </div>
  )
}