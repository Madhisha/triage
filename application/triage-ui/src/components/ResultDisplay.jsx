import NEWS2Breakdown from './NEWS2Breakdown'

const SEVERITY_MAP = {
    1: { key: 'critical', label: 'Critical' },
    2: { key: 'urgent', label: 'Urgent' },
    3: { key: 'nonurgent', label: 'Non-Urgent' },
}

const VITAL_LABELS = {
    temperature: 'Temperature',
    heartrate: 'Heart Rate',
    resprate: 'Resp Rate',
    o2sat: 'O₂ Saturation',
    sbp: 'Systolic BP',
    dbp: 'Diastolic BP',
    pain: 'Pain Level',
    chiefcomplaint: 'Chief Complaint',
}

const VITAL_UNITS = {
    temperature: ' °F',
    heartrate: ' bpm',
    resprate: ' br/min',
    o2sat: '%',
    sbp: ' mmHg',
    dbp: ' mmHg',
    pain: '/10',
    chiefcomplaint: '',
}

export default function ResultDisplay({ result, onReset }) {
    if (!result) return null

    const cls = result.prediction_class
    const severity = SEVERITY_MAP[cls] || SEVERITY_MAP[3]
    const confidencePct = Math.round(result.confidence * 100)
    const individualScores = parseNEWS2Scores(result.explanation)

    return (
        <div className="result-panel">
            <div className="card">
                {/* Triage Badge */}
                <div className={`triage-badge triage-badge--${severity.key}`}>
                    <div className="triage-badge__indicator" />
                    <div className="triage-badge__text">
                        <div className="triage-badge__label">{result.prediction_label}</div>
                        <div className="triage-badge__class">Triage Level {cls}</div>
                    </div>
                </div>

                {/* Metrics */}
                <div className="result-meta">
                    <div className="meta-item">
                        <div className="meta-item__label">Confidence</div>
                        <div className="meta-item__value">{confidencePct}%</div>
                        <div className="confidence-bar">
                            <div
                                className={`confidence-bar__fill confidence-bar__fill--${severity.key}`}
                                style={{ width: `${confidencePct}%` }}
                            />
                        </div>
                    </div>
                    <div className="meta-item">
                        <div className="meta-item__label">Source</div>
                        <div className="meta-item__value">{result.prediction_source}</div>
                    </div>
                    <div className="meta-item">
                        <div className="meta-item__label">NEWS2 Score</div>
                        <div className="meta-item__value">{result.news2_score}</div>
                    </div>
                </div>

                <NEWS2Breakdown
                    news2Score={result.news2_score}
                    individualScores={individualScores}
                />
            </div>

            {/* Explanation */}
            <div className="card explanation-section">
                <div className="card-title">Clinical Explanation</div>
                <div className="explanation-box"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(result.explanation) }}
                />
            </div>

            {/* Patient Summary */}
            {result.patient_summary && (
                <div className="card patient-summary">
                    <div className="card-title">Patient Summary</div>
                    <div className="summary-grid">
                        {Object.entries(result.patient_summary).map(([key, value]) => (
                            <div className="summary-item" key={key}>
                                <span className="summary-item__label">{VITAL_LABELS[key] || key}</span>
                                <span className="summary-item__value">
                                    {value}{VITAL_UNITS[key] || ''}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <button className="reset-btn" onClick={onReset}>
                ← New Assessment
            </button>
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
