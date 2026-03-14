import { Link } from 'react-router-dom'
import './LandingPage.css'

const PATHOLOGIES = [
  { icon: '🦠', name: 'Dental Caries',         desc: 'Spots early and advanced tooth decay across the dentition on a single panoramic view.', color: '#FFA500' },
  { icon: '🔴', name: 'Periapical Infection',   desc: 'Flags infection and bone loss around root tips that may require root canal treatment or extraction.', color: '#FF4444' },
  { icon: '📐', name: 'Impacted Teeth',         desc: 'Locates unerupted or misaligned teeth that may need surgical intervention.', color: '#4d9ef7' },
  { icon: '⚡', name: 'Fractured Teeth',        desc: 'Identifies cracks or structural breaks in teeth caused by trauma or heavy bite forces.', color: '#FFFF44' },
  { icon: '🔩', name: 'Broken Down Crown/Root', desc: 'Highlights severely decayed crowns or retained roots that are no longer restorable.', color: '#FF44FF' },
]

const STEPS = [
  { n: '01', title: 'Upload OPG',        desc: 'Upload a panoramic dental radiograph (OPG). The system checks the image automatically before analysis.' },
  { n: '02', title: 'AI Detection',      desc: 'A deep learning model scans the full OPG and marks all regions of clinical interest.' },
  { n: '03', title: 'Classification',    desc: 'Each marked region is classified into one of five pathology categories with a confidence score.' },
  { n: '04', title: 'Visual Explanation',desc: 'A heatmap overlay highlights the exact areas of the radiograph that influenced each finding — supporting clinical transparency.' },
]

const STATS = [
  { value: '90.7%', label: 'Accuracy' },
  { value: '95.9%', label: 'Detection Rate' },
  { value: '0.982', label: 'AUC-ROC' },
  { value: '5',     label: 'Pathology Classes' },
]

export default function LandingPage() {
  return (
    <div className="landing page">

      <section className="hero">
        <div className="hero-bg-grid" />
        <div className="hero-glow" />
        <div className="hero-content">
          <div className="hero-badge">AI-Assisted · Explainable · OPG Radiographs</div>
          <h1 className="hero-title">
            AI-Powered<br />
            <span className="hero-accent">Dental Pathology</span><br />
            Detection
          </h1>
          <p className="hero-sub">
            iDentify analyses panoramic dental radiographs using deep learning
            to automatically detect and classify dental pathologies —
            with visual explanations to support clinical decision-making.
          </p>
          <div className="hero-actions">
            <Link to="/predict" className="btn btn-primary">Start Analysis →</Link>
            <button className="btn btn-ghost" onClick={() => document.getElementById('how-it-works').scrollIntoView({ behavior: 'smooth' })}>How It Works</button>
          </div>
          <div className="hero-stats">
            {STATS.map(s => (
              <div key={s.label} className="stat-item">
                <span className="stat-value">{s.value}</span>
                <span className="stat-label">{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-inner">
          <div className="section-header">
            <h2>Detectable Pathologies</h2>
            <p>iDentify is trained to identify five clinically significant dental conditions from panoramic radiographs.</p>
          </div>
          <div className="pathology-grid">
            {PATHOLOGIES.map(p => (
              <div key={p.name} className="pathology-card" style={{ '--accent': p.color }}>
                <span className="p-icon">{p.icon}</span>
                <h3 className="p-name">{p.name}</h3>
                <p className="p-desc">{p.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="how-it-works" className="section section-dark">
        <div className="section-inner">
          <div className="section-header">
            <h2>How It Works</h2>
            <p>From upload to explained findings in seconds.</p>
          </div>
          <div className="steps-grid">
            {STEPS.map(s => (
              <div key={s.n} className="step-card">
                <span className="step-num">{s.n}</span>
                <h3 className="step-title">{s.title}</h3>
                <p className="step-desc">{s.desc}</p>
              </div>
            ))}
          </div>
          <div className="cta-row">
            <Link to="/instructions" className="btn btn-ghost">Read Full Instructions</Link>
            <Link to="/predict"      className="btn btn-primary">Try It Now →</Link>
          </div>
        </div>
      </section>

      <footer className="footer">
        <span>iDentify — Final Year Project · IIT in collaboration with University of Westminster</span>
      </footer>

    </div>
  )
}