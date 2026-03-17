import { Link, useLocation } from 'react-router-dom'
import './Navbar.css'

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span className="brand-icon">🦷</span>
        <span className="brand-name">iDentify</span>
      </Link>

      <div className="navbar-links">
        <Link to="/"             className={`nav-link ${pathname === '/' ? 'active' : ''}`}>Home</Link>
        <Link to="/instructions" className={`nav-link ${pathname === '/instructions' ? 'active' : ''}`}>Instructions</Link>
        <Link to="/predict"      className={`nav-link ${pathname === '/predict' ? 'active' : ''}`}>Analyse</Link>
      </div>

      <Link to="/predict" className="btn btn-primary nav-cta">
        Start Analysis
      </Link>
    </nav>
  )
}