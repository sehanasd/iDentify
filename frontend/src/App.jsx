import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import LandingPage from './pages/LandingPage'
import InstructionsPage from './pages/InstructionsPage'
import PredictionPage from './pages/PredictionPage'

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"             element={<LandingPage />} />
        <Route path="/instructions" element={<InstructionsPage />} />
        <Route path="/predict"      element={<PredictionPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App