import NEWS2Breakdown from './NEWS2Breakdown'

const SEVERITY_MAP = {
  1: { key: 'critical', label: 'Critical', color: 'var(--critical)' },
  2: { key: 'urgent', label: 'Urgent', color: 'var(--urgent)' },
  3: { key: 'nonurgent', label: 'Non-Urgent', color: 'var(--safe)' },
}

const VITAL_CONFIG = [
  { key: 'heartrate', label: 'Heart Rate', unit: 'bpm', icon: '❤️', warnLow: 50, warnHigh: 100 },
  { key: 'o2sat', label: 'SpO₂', unit: '%', icon: '🫁', warnLow: 95, warnHigh: 999 },
  { key: 'sbp', label: 'Systolic BP', unit: 'mmHg', icon: '🩸', warnLow: 90, warnHigh: 160 },
  { key: 'dbp', label: 'Diastolic BP', unit: 'mmHg', icon: '🩸', warnLow: 40, warnHigh: 100 },
  { key: 'temperature', label: 'Temperature', unit: '°F', icon: '🌡️', warnLow: 96, warnHigh: 100.4 },
  { key: 'resprate', label: 'Resp Rate', unit: '/min', icon: '🌬️', warnLow: 10, warnHigh: 22 },
  { key: 'pain', label: 'Pain Level', unit: '/10', icon: '⚡', warnLow: -1, warnHigh: 7 },
]

export default function SoldierDetail({ soldier, onClose }) {
  if (!soldier) return null

  const pred = soldier.prediction
  const severity = pred ? (SEVERITY_MAP[pred.prediction_class] || SEVERITY_MAP[3]) : null
  const confidencePct = pred ? Math.round(pred.confidence * 100) : 0
  const individualScores = pred ? parseNEWS2Scores(pred.explanation) : null

  return (
    <div className="detail-overlay" onClick={onClose} id="soldier-detail-overlay">
      <div className="detail-panel" onClick={e => e.stopPropagation()} id="soldier-detail-panel">
        {/* Close Button */}
        <button className="detail-close" onClick={onClose} id="detail-close-btn">✕</button>

        {/* Soldier Profile Header */}
        <div className="detail-profile">
          <div className="detail-avatar">
            {soldier.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
          </div>
          <div className="detail-profile__info">
            <h2 className="detail-profile__name">{soldier.name}</h2>
            <div className="detail-profile__meta">
              <span>{soldier.rank}</span>
              <span className="detail-sep">•</span>
              <span>{soldier.callsign}</span>
            </div>
          </div>
          {severity && (
            <div className={`detail-acuity detail-acuity--${severity.key}`}>
              {severity.label}
            </div>
          )}
        </div>

        {/* Info Tags */}
        <div className="detail-tags">
          <div className="detail-tag">
            <span className="detail-tag__label">Unit</span>
            <span className="detail-tag__value">{soldier.unit}</span>
          </div>
          <div className="detail-tag">
            <span className="detail-tag__label">Blood Type</span>
            <span className="detail-tag__value">{soldier.bloodType}</span>
          </div>
          <div className="detail-tag">
            <span className="detail-tag__label">Age</span>
            <span className="detail-tag__value">{soldier.age}</span>
          </div>
          <div className="detail-tag">
            <span className="detail-tag__label">Status</span>
            <span className="detail-tag__value">{soldier.missionStatus}</span>
          </div>
        </div>

        {/* Chief Complaint */}
        {soldier.chiefcomplaint && soldier.chiefcomplaint !== 'none' && (
          <div className="detail-complaint">
            <div className="detail-section-title">Chief Complaint</div>
            <div className="detail-complaint__text">{soldier.chiefcomplaint}</div>
          </div>
        )}

        {/* Prediction Info */}
        {pred && (
          <div className="detail-prediction">
            <div className="detail-section-title">Prediction Details</div>
            <div className="detail-prediction__grid">
              <div className="detail-prediction__item">
                <span className="detail-prediction__label">Confidence</span>
                <span className="detail-prediction__value">{confidencePct}%</span>
                <div className="detail-confidence-bar">
                  <div
                    className={`detail-confidence-fill detail-confidence-fill--${severity.key}`}
                    style={{ width: `${confidencePct}%` }}
                  />
                </div>
              </div>
              <div className="detail-prediction__item">
                <span className="detail-prediction__label">Source</span>
                <span className="detail-prediction__value">{pred.prediction_source}</span>
              </div>
              <div className="detail-prediction__item">
                <span className="detail-prediction__label">NEWS2 Score</span>
                <span className="detail-prediction__value">{pred.news2_score}</span>
              </div>
            </div>
          </div>
        )}

        {/* Vitals Grid */}
        <div className="detail-vitals-section">
          <div className="detail-section-title">Current Vitals</div>
          <div className="detail-vitals-grid">
            {VITAL_CONFIG.map(v => {
              const val = soldier.vitals[v.key]
              const isWarn = val < v.warnLow || val > v.warnHigh
              return (
                <div className={`detail-vital ${isWarn ? 'detail-vital--warn' : 'detail-vital--ok'}`} key={v.key}>
                  <div className="detail-vital__icon">{v.icon}</div>
                  <div className="detail-vital__info">
                    <div className="detail-vital__label">{v.label}</div>
                    <div className="detail-vital__value">
                      {val}
                      <span className="detail-vital__unit">{v.unit}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* NEWS2 Breakdown */}
        {individualScores && (
          <div className="detail-news2">
            <NEWS2Breakdown
              news2Score={pred.news2_score}
              individualScores={individualScores}
            />
          </div>
        )}

        {/* Vitals History */}
        {soldier.history && soldier.history.length > 0 && (
          <div className="detail-history">
            <div className="detail-section-title">Recent Vitals History ({soldier.history.length} readings)</div>
            <div className="detail-history-table-wrap">
              <table className="detail-history-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>HR</th>
                    <th>SpO₂</th>
                    <th>BP</th>
                    <th>Temp</th>
                    <th>RR</th>
                    <th>Pain</th>
                  </tr>
                </thead>
                <tbody>
                  {soldier.history.slice(-10).reverse().map((h, i) => (
                    <tr key={i}>
                      <td className="detail-history-idx">{soldier.history.length - i}</td>
                      <td>{h.heartrate}</td>
                      <td>{h.o2sat}%</td>
                      <td>{h.sbp}/{h.dbp}</td>
                      <td>{h.temperature}</td>
                      <td>{h.resprate}</td>
                      <td>{h.pain}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Clinical Explanation */}
        {pred && pred.explanation && (
          <div className="detail-explanation">
            <div className="detail-section-title">Clinical Explanation</div>
            <div className="explanation-box"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(pred.explanation) }}
            />
          </div>
        )}
      </div>
    </div>
  )
}

function renderMarkdown(text) {
  if (!text) return ''
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^[•\-]\s+(.+)$/gm, '<li>$1</li>')
    .replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
    .replace(/\n/g, '<br/>')
    .replace(/<br\/>\s*<ul>/g, '<ul>')
    .replace(/<\/ul>\s*<br\/>/g, '</ul>')
    .replace(/<\/li><br\/>/g, '</li>')
}

function parseNEWS2Scores(explanation) {
  if (!explanation) return null
  const scores = {}
  const patterns = [
    { key: 'resp_rate', regex: /Respiration Rate.*?Score:\s*(\d+)/i },
    { key: 'o2sat', regex: /Oxygen Saturation.*?Score:\s*(\d+)/i },
    { key: 'temperature', regex: /Temperature.*?Score:\s*(\d+)/i },
    { key: 'sbp', regex: /Systolic Blood Pressure.*?Score:\s*(\d+)/i },
    { key: 'heart_rate', regex: /Heart Rate.*?Score:\s*(\d+)/i },
  ]
  let found = false
  for (const { key, regex } of patterns) {
    const m = explanation.match(regex)
    if (m) { scores[key] = parseInt(m[1], 10); found = true }
  }
  return found ? scores : null
}
