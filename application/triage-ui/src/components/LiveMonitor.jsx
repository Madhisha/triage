import { useState, useEffect, useRef, useCallback } from 'react'
import { initializeSoldiers, simulateVitals } from '../data/mockSoldiers'
import SoldierCard from './SoldierCard'
import SoldierDetail from './SoldierDetail'

const API_URL = 'http://localhost:5000/api'
const POLL_INTERVAL = 4000 // 4 seconds
const MAX_HISTORY = 20

export default function LiveMonitor() {
  const [soldiers, setSoldiers] = useState(() => initializeSoldiers())
  const [selectedSoldier, setSelectedSoldier] = useState(null)
  const [connected, setConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [error, setError] = useState(null)
  const [isPaused, setIsPaused] = useState(false)
  const timerRef = useRef(null)

  // Count acuity levels
  const counts = soldiers.reduce(
    (acc, s) => {
      if (s.prediction) {
        const cls = s.prediction.prediction_class
        if (cls === 1) acc.critical++
        else if (cls === 2) acc.urgent++
        else acc.stable++
      } else {
        acc.pending++
      }
      return acc
    },
    { critical: 0, urgent: 0, stable: 0, pending: 0 }
  )

  const isPausedRef = useRef(isPaused)
  useEffect(() => {
    isPausedRef.current = isPaused
  }, [isPaused])

  // Track the absolute newest soldiers state to use inside tick without closure staleness
  const soldiersRef = useRef(soldiers)
  useEffect(() => {
    soldiersRef.current = soldiers
  }, [soldiers])

  const tick = useCallback(async () => {
    if (isPausedRef.current) return

    // 1. Simulate new vitals for all soldiers
    const updated = soldiersRef.current.map(s => {
      const newVitals = simulateVitals(s.vitals, s.baseline)
      return {
        ...s,
        vitals: newVitals,
        history: [...s.history.slice(-(MAX_HISTORY - 1)), { ...s.vitals }],
      }
    })
    
    // Visually update the UI right away
    setSoldiers(updated)

    // 2. Send to backend for prediction (await the result to avoid overlap)
    try {
      const patients = updated.map(s => ({
        ...s.vitals,
        chiefcomplaint: s.chiefcomplaint || 'none',
      }))

      const res = await fetch(`${API_URL}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patients }),
      })
      
      const data = await res.json()
      
      if (data.results) {
        setSoldiers(curr =>
          curr.map((s, i) => ({
            ...s,
            prediction: data.results[i]?.error ? s.prediction : data.results[i],
            lastUpdated: new Date(),
          }))
        )
        setConnected(true)
        setLastUpdate(new Date())
        setError(null)
      }
    } catch (err) {
      console.error('Batch prediction error:', err)
      setError('Cannot reach prediction service')
      setConnected(false)
    } finally {
      // Schedule next tick immediately after processing finishes (no arbitrary wait limit)
      if (!isPausedRef.current) {
        timerRef.current = setTimeout(tick, 100) // Small 100ms breather for React to render
      }
    }
  }, [])

  // Start/stop polling
  useEffect(() => {
    if (isPaused) {
      if (timerRef.current) clearTimeout(timerRef.current)
    } else {
      // Start polling
      tick()
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [isPaused, tick])

  // Keep selectedSoldier in sync with latest data
  useEffect(() => {
    if (selectedSoldier) {
      const updated = soldiers.find(s => s.id === selectedSoldier.id)
      if (updated) setSelectedSoldier(updated)
    }
  }, [soldiers])

  return (
    <div className="monitor" id="live-monitor">
      {/* Header Bar */}
      <div className="monitor__header">
        <div className="monitor__header-left">
          <h2 className="monitor__title">
            <span className={`monitor__live-dot ${connected ? 'monitor__live-dot--active' : ''}`} />
            FIELD MONITORING — LIVE
          </h2>
          {lastUpdate && (
            <span className="monitor__timestamp">
              Last update: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
        <div className="monitor__header-right">
          <button
            className={`monitor__pause-btn ${isPaused ? 'monitor__pause-btn--paused' : ''}`}
            onClick={() => setIsPaused(p => !p)}
            id="pause-btn"
          >
            {isPaused ? '▶ Resume' : '⏸ Pause'}
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="monitor__stats">
        <div className="monitor__stat monitor__stat--critical">
          <span className="monitor__stat-count">{counts.critical}</span>
          <span className="monitor__stat-label">Critical</span>
        </div>
        <div className="monitor__stat monitor__stat--urgent">
          <span className="monitor__stat-count">{counts.urgent}</span>
          <span className="monitor__stat-label">Urgent</span>
        </div>
        <div className="monitor__stat monitor__stat--stable">
          <span className="monitor__stat-count">{counts.stable}</span>
          <span className="monitor__stat-label">Stable</span>
        </div>
        <div className="monitor__stat monitor__stat--pending">
          <span className="monitor__stat-count">{counts.pending}</span>
          <span className="monitor__stat-label">Pending</span>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="monitor__error">
          ⚠ {error} — Make sure the Flask API is running on port 5000
        </div>
      )}

      {/* Soldier Cards Grid */}
      <div className="monitor__grid">
        {soldiers.map(soldier => (
          <SoldierCard
            key={soldier.id}
            soldier={soldier}
            onClick={() => setSelectedSoldier(soldier)}
          />
        ))}
      </div>

      {/* Detail Panel */}
      {selectedSoldier && (
        <SoldierDetail
          soldier={selectedSoldier}
          onClose={() => setSelectedSoldier(null)}
        />
      )}
    </div>
  )
}
