import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import PatientForm from './components/PatientForm'
import ResultDisplay from './components/ResultDisplay'
import LiveMonitor from './components/LiveMonitor'

const API_URL = 'http://localhost:5000/api'

function TriageAssessment() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (patientData) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(
          data.details
            ? Array.isArray(data.details) ? data.details.join(' ') : data.details
            : data.error || 'Prediction failed'
        )
      }
      setResult(data)
    } catch (err) {
      setError(err.message || 'Unable to reach prediction service.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="triage-page">
      <div className="triage-page__header">
        <h2 className="triage-page__title">Patient Triage Assessment</h2>
        <p className="triage-page__subtitle">
          NEWS2 rule-based assessment · ML stacking ensemble
        </p>
      </div>

      {!result && (
        <PatientForm onSubmit={handleSubmit} loading={loading} />
      )}

      {error && (
        <div className="error-alert">{error}</div>
      )}

      {result && (
        <ResultDisplay result={result} onReset={handleReset} />
      )}
    </div>
  )
}

export default function App() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('theme')
    if (saved) return saved === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div className="app">
      <Navbar dark={dark} onToggleTheme={() => setDark(d => !d)} />

      <main className="app-main">
        <Routes>
          <Route path="/" element={<TriageAssessment />} />
          <Route path="/monitor" element={<LiveMonitor />} />
        </Routes>
      </main>

      <footer className="app-footer">
        Triage Command System — For clinical decision support only
      </footer>
    </div>
  )
}
