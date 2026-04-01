const SCORE_LABELS = [
    { key: 'resp_rate', label: 'Resp Rate' },
    { key: 'o2sat', label: 'O₂ Sat' },
    { key: 'temperature', label: 'Temp' },
    { key: 'sbp', label: 'Sys BP' },
    { key: 'heart_rate', label: 'Heart Rate' },
]

export default function NEWS2Breakdown({ news2Score, individualScores }) {
    if (!individualScores) return null

    return (
        <div className="news2-section">
            <div className="card-title">NEWS2 Score Breakdown</div>
            <div className="news2-grid">
                {SCORE_LABELS.map(({ key, label }) => {
                    const score = individualScores[key] ?? 0
                    return (
                        <div className="news2-item" key={key}>
                            <div className="news2-item__name">{label}</div>
                            <div className={`news2-item__score news2-item__score--${Math.min(score, 3)}`}>
                                {score}
                            </div>
                        </div>
                    )
                })}
                <div className="news2-item" style={{ border: '1px solid #d5d9e3' }}>
                    <div className="news2-item__name">Total</div>
                    <div className={`news2-item__score news2-item__score--${news2Score >= 7 ? 3 : news2Score >= 5 ? 2 : news2Score >= 1 ? 1 : 0}`}>
                        {news2Score}
                    </div>
                </div>
            </div>
        </div>
    )
}
