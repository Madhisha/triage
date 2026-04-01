"""
Flask REST API for Patient Triage Prediction

Endpoints:
    POST /api/predict  - Make triage prediction for patient data
    GET  /api/health   - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from application.prediction_service import TriagePredictionService

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# Initialize prediction service once at startup
print("Initializing Triage Prediction Service...")
service = TriagePredictionService()
print("✓ Service ready!\n")

# Validation ranges (from ml_preprocess.py VALID_RANGES)
VALID_RANGES = {
    'temperature': (91.4, 107.6, '°F'),
    'heartrate': (10, 300, 'bpm'),
    'resprate': (3, 60, 'breaths/min'),
    'o2sat': (60, 100, '%'),
    'sbp': (30, 300, 'mmHg'),
    'dbp': (30, 300, 'mmHg'),
    'pain': (0, 10, 'scale'),
}

REQUIRED_FIELDS = list(VALID_RANGES.keys())


def validate_patient_data(data):
    """Validate patient input data. Returns (cleaned_data, errors)."""
    errors = []
    cleaned = {}

    for field in REQUIRED_FIELDS:
        value = data.get(field)
        if value is None or value == '':
            errors.append(f"'{field}' is required.")
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be a number.")
            continue

        low, high, unit = VALID_RANGES[field]
        if not (low <= value <= high):
            errors.append(f"'{field}' must be between {low}–{high} {unit}.")
            continue

        cleaned[field] = value

    # Chief complaint is optional, defaults to 'none'
    cleaned['chiefcomplaint'] = str(data.get('chiefcomplaint', 'none')).strip() or 'none'

    return cleaned, errors


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make a triage prediction.

    Expects JSON body:
    {
        "temperature": 98.6,
        "heartrate": 75,
        "resprate": 16,
        "o2sat": 98,
        "sbp": 120,
        "dbp": 80,
        "pain": 3,
        "chiefcomplaint": "headache"
    }
    """
    import time

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    print("\n" + "=" * 60)
    print(f"📥 PREDICTION REQUEST RECEIVED")
    print(f"   Input: {data}")
    print("=" * 60)

    cleaned, errors = validate_patient_data(data)
    if errors:
        print(f"❌ Validation failed: {errors}")
        return jsonify({'error': 'Validation failed.', 'details': errors}), 422

    try:
        start = time.time()
        result = service.predict_with_explanation(cleaned)
        elapsed = time.time() - start

        print(f"✅ Prediction: {result['prediction_label']} (Class {result['prediction_class']})")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Source: {result['prediction_source']}")
        print(f"   NEWS2 Score: {result['news2_score']}")
        print(f"   ⏱  Time: {elapsed:.3f}s")
        print("=" * 60 + "\n")

        return jsonify(result), 200
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': 'Prediction failed.', 'details': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make triage predictions for multiple patients at once.

    Expects JSON body:
    {
        "patients": [
            { "temperature": 98.6, "heartrate": 75, ... },
            ...
        ]
    }

    Returns:
    {
        "results": [ { prediction result }, ... ]
    }
    """
    import time

    data = request.get_json(silent=True)
    if not data or 'patients' not in data:
        return jsonify({'error': 'Request body must be JSON with a "patients" array.'}), 400

    patients = data['patients']
    if not isinstance(patients, list) or len(patients) == 0:
        return jsonify({'error': '"patients" must be a non-empty array.'}), 400

    if len(patients) > 20:
        return jsonify({'error': 'Maximum 20 patients per batch request.'}), 400

    print(f"\n{'=' * 60}")
    print(f"📥 BATCH PREDICTION REQUEST — {len(patients)} patients")
    print(f"{'=' * 60}")

    results = []
    start = time.time()

    for i, patient in enumerate(patients):
        cleaned, errors = validate_patient_data(patient)
        if errors:
            results.append({'error': 'Validation failed.', 'details': errors, 'index': i})
            continue

        try:
            result = service.predict_with_explanation(cleaned)
            results.append(result)
        except Exception as e:
            results.append({'error': 'Prediction failed.', 'details': str(e), 'index': i})

    elapsed = time.time() - start
    print(f"✅ Batch complete — {len(results)} results in {elapsed:.3f}s")
    print(f"{'=' * 60}\n")

    return jsonify({'results': results}), 200


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': service.ml_model is not None,
        'scaler_loaded': service.scaler is not None,
        'tfidf_loaded': service.tfidf_vectorizer is not None,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
