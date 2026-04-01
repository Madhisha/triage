import { useState } from 'react'

const FIELDS = [
    { name: 'temperature', label: 'Temperature', unit: '°F', min: 91.4, max: 107.6, step: 0.1, placeholder: '98.6' },
    { name: 'heartrate', label: 'Heart Rate', unit: 'bpm', min: 10, max: 300, step: 1, placeholder: '75' },
    { name: 'resprate', label: 'Respiratory Rate', unit: 'breaths/min', min: 3, max: 60, step: 1, placeholder: '16' },
    { name: 'o2sat', label: 'O₂ Saturation', unit: '%', min: 60, max: 100, step: 1, placeholder: '98' },
    { name: 'sbp', label: 'Systolic BP', unit: 'mmHg', min: 30, max: 300, step: 1, placeholder: '120' },
    { name: 'dbp', label: 'Diastolic BP', unit: 'mmHg', min: 30, max: 300, step: 1, placeholder: '80' },
]

function getPainColor(value) {
    if (value <= 3) return '#1a8a5c'
    if (value <= 6) return '#c27816'
    return '#d63a3a'
}

export default function PatientForm({ onSubmit, loading }) {
    const [values, setValues] = useState({
        temperature: '', heartrate: '', resprate: '', o2sat: '',
        sbp: '', dbp: '', pain: 0, chiefcomplaint: ''
    })
    const [errors, setErrors] = useState({})

    const handleChange = (name, value) => {
        setValues(prev => ({ ...prev, [name]: value }))
        if (errors[name]) {
            setErrors(prev => { const n = { ...prev }; delete n[name]; return n })
        }
    }

    const validate = () => {
        const newErrors = {}
        for (const field of FIELDS) {
            const val = values[field.name]
            if (val === '' || val === undefined || val === null) {
                newErrors[field.name] = 'Required'
                continue
            }
            const num = parseFloat(val)
            if (isNaN(num)) { newErrors[field.name] = 'Must be a number'; continue }
            if (num < field.min || num > field.max) {
                newErrors[field.name] = `Range: ${field.min}–${field.max}`
            }
        }
        setErrors(newErrors)
        return Object.keys(newErrors).length === 0
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        if (!validate()) return
        const data = {}
        for (const field of FIELDS) data[field.name] = parseFloat(values[field.name])
        data.pain = Number(values.pain)
        data.chiefcomplaint = values.chiefcomplaint || 'none'
        onSubmit(data)
    }

    const painVal = Number(values.pain)
    const painColor = getPainColor(painVal)

    return (
        <form className="card" onSubmit={handleSubmit} id="patient-form">
            <div className="card-title">Vital Signs</div>

            <div className="form-grid">
                {FIELDS.map(field => (
                    <div className="form-group" key={field.name}>
                        <label className="form-label" htmlFor={field.name}>
                            {field.label}
                            <span className="form-label__unit">{field.unit}</span>
                        </label>
                        <input
                            id={field.name}
                            className={`form-input${errors[field.name] ? ' form-input--error' : ''}`}
                            type="number"
                            step={field.step}
                            min={field.min}
                            max={field.max}
                            placeholder={field.placeholder}
                            value={values[field.name]}
                            onChange={e => handleChange(field.name, e.target.value)}
                            disabled={loading}
                        />
                        {errors[field.name]
                            ? <span className="form-error">{errors[field.name]}</span>
                            : <span className="form-range-hint">{field.min} – {field.max}</span>
                        }
                    </div>
                ))}

                {/* Pain Level — Slider */}
                <div className="form-group pain-slider-group">
                    <div className="pain-slider-header">
                        <label className="form-label" htmlFor="pain">
                            Pain Level
                            <span className="form-label__unit">0–10</span>
                        </label>
                        <span className="pain-slider-value" style={{ color: painColor }}>{painVal}</span>
                    </div>
                    <div className="pain-slider-wrapper">
                        <div className="pain-slider-track">
                            <div
                                className="pain-slider-fill"
                                style={{ width: `${painVal * 10}%`, background: painColor }}
                            />
                        </div>
                        <input
                            id="pain"
                            type="range"
                            className="pain-slider-input"
                            min={0}
                            max={10}
                            step={1}
                            value={values.pain}
                            onChange={e => handleChange('pain', e.target.value)}
                            disabled={loading}
                        />
                    </div>
                    <div className="pain-ticks">
                        {[...Array(11)].map((_, i) => (
                            <span
                                key={i}
                                className={`pain-tick${i === painVal ? ' pain-tick--active' : ''}`}
                                style={i === painVal ? { color: painColor, fontWeight: 700 } : undefined}
                                onClick={() => !loading && handleChange('pain', i)}
                            >
                                {i}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Chief Complaint */}
                <div className="form-group form-group--full">
                    <label className="form-label" htmlFor="chiefcomplaint">
                        Chief Complaint
                        <span className="form-label__unit">optional</span>
                    </label>
                    <input
                        id="chiefcomplaint"
                        className="form-input"
                        type="text"
                        placeholder="e.g., chest pain, shortness of breath"
                        value={values.chiefcomplaint}
                        onChange={e => handleChange('chiefcomplaint', e.target.value)}
                        disabled={loading}
                    />
                </div>
            </div>

            <button type="submit" className="submit-btn" id="submit-prediction" disabled={loading}>
                {loading
                    ? <><span className="spinner" /> Analyzing...</>
                    : 'Run Triage Assessment'
                }
            </button>
        </form>
    )
}
