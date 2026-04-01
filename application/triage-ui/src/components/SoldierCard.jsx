const SEVERITY_MAP = {
  1: { key: 'critical', label: 'CRITICAL', emoji: '🔴' },
  2: { key: 'urgent', label: 'URGENT', emoji: '🟠' },
  3: { key: 'nonurgent', label: 'STABLE', emoji: '🟢' },
}

export default function SoldierCard({ soldier, onClick }) {
  const pred = soldier.prediction
  const severity = pred ? (SEVERITY_MAP[pred.prediction_class] || SEVERITY_MAP[3]) : null
  const vitals = soldier.vitals

  const confidencePct = pred ? Math.round(pred.confidence * 100) : 0

  return (
    <div
      className={`soldier-card ${severity ? `soldier-card--${severity.key}` : 'soldier-card--loading'}`}
      onClick={onClick}
      id={`soldier-card-${soldier.id}`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
    >
      {/* Status indicator strip */}
      <div className={`soldier-card__strip ${severity ? `soldier-card__strip--${severity.key}` : ''}`} />

      <div className="soldier-card__body">
        {/* Header */}
        <div className="soldier-card__header">
          <div className="soldier-card__identity">
            <div className="soldier-card__name">{soldier.name}</div>
            <div className="soldier-card__callsign">{soldier.callsign}</div>
          </div>
          {severity && (
            <div className={`soldier-card__badge soldier-card__badge--${severity.key}`}>
              {severity.label}
            </div>
          )}
        </div>

        {/* Vitals Grid */}
        <div className="soldier-card__vitals">
          <VitalItem label="HR" value={vitals.heartrate} unit="bpm" warn={vitals.heartrate > 100 || vitals.heartrate < 50} />
          <VitalItem label="SpO₂" value={vitals.o2sat} unit="%" warn={vitals.o2sat < 95} />
          <VitalItem label="BP" value={`${vitals.sbp}/${vitals.dbp}`} unit="" warn={vitals.sbp < 90 || vitals.sbp > 160} />
          <VitalItem label="Temp" value={vitals.temperature} unit="°F" warn={vitals.temperature > 100.4 || vitals.temperature < 96} />
          <VitalItem label="RR" value={vitals.resprate} unit="/min" warn={vitals.resprate > 22 || vitals.resprate < 10} />
          <VitalItem label="Pain" value={vitals.pain} unit="/10" warn={vitals.pain >= 7} />
        </div>

        {/* Footer */}
        <div className="soldier-card__footer">
          {pred && (
            <div className="soldier-card__confidence">
              <div className="soldier-card__confidence-bar">
                <div
                  className={`soldier-card__confidence-fill soldier-card__confidence-fill--${severity.key}`}
                  style={{ width: `${confidencePct}%` }}
                />
              </div>
              <span className="soldier-card__confidence-text">{confidencePct}%</span>
            </div>
          )}
          <div className="soldier-card__mission">{soldier.missionStatus}</div>
        </div>
      </div>
    </div>
  )
}

function VitalItem({ label, value, unit, warn }) {
  return (
    <div className={`soldier-vital ${warn ? 'soldier-vital--warn' : ''}`}>
      <span className="soldier-vital__label">{label}</span>
      <span className="soldier-vital__value">
        {value}
        <span className="soldier-vital__unit">{unit}</span>
      </span>
    </div>
  )
}
