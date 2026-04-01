"""
Patient Triage Prediction Service with Explanations

This module provides the core logic for:
1. Taking patient input data (vitals + chief complaint)
2. Making triage predictions using the hybrid system
3. Generating explanations for the predictions

The hybrid system uses:
- Rule-based NEWS2 scoring (for critical cases)
- ML Stacking Ensemble (for non-critical cases)
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not available — explanations will not include SHAP analysis.")

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rule_based.rule_based_triage import rule_based_triage, calculate_news2_score
from ml_model.ml_preprocess import clean_text


class TriagePredictionService:
    """
    Service class for making patient triage predictions with explanations
    """
    
    def __init__(self, model_path=None, tfidf_path=None, scaler_path=None, shap_background_path=None, feature_order_path=None):
        """
        Initialize the prediction service
        
        Args:
            model_path: Path to the trained stacking ensemble model
            tfidf_path: Path to the saved TF-IDF vectorizer
            scaler_path: Path to the saved StandardScaler
            shap_background_path: Path to background data for SHAP (optional)
            feature_order_path: Path to the saved feature order (optional)
        """
        # Default paths
        base_dir = os.path.dirname(__file__)
        if model_path is None:
            model_path = os.path.join(base_dir, '../ml_model/ensemble_model/stacking_lr_ensemble.pkl')
        if tfidf_path is None:
            tfidf_path = os.path.join(base_dir, '../ml_model/ml_processed_data/tfidf_vectorizer.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(base_dir, '../ml_model/ml_processed_data/scaler.pkl')
        if shap_background_path is None:
            shap_background_path = os.path.join(base_dir, '../hybrid_triage/xai_outputs/shap_background.csv')
        if feature_order_path is None:
            feature_order_path = os.path.join(base_dir, '../ml_model/ml_processed_data/feature_order.pkl')
        
        self.model_path = model_path
        self.tfidf_path = tfidf_path
        self.scaler_path = scaler_path
        self.shap_background_path = shap_background_path
        self.feature_order_path = feature_order_path
        self.ml_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.shap_explainer = None
        self.shap_background_data = None
        self.feature_order = None
        
        # Class labels mapping
        self.class_labels = {
            1: 'Critical',
            2: 'Urgent',
            3: 'Non-Urgent'
        }
        
        # Expected input features (from dataset)
        self.vital_features = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        
        # Load ML model, scaler, TF-IDF vectorizer, feature order, and SHAP background
        self._load_ml_model()
        self._load_scaler()
        self._load_tfidf_vectorizer()
        self._load_feature_order()
        self._load_shap_background()
    
    
    def _load_ml_model(self):
        """Load the pre-trained ML stacking ensemble model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                print(f"✓ Loaded ML model from: {self.model_path}")
            else:
                print(f"⚠ Warning: ML model not found at {self.model_path}")
                print("  The service will only use rule-based predictions.")
        except Exception as e:
            print(f"⚠ Error loading ML model: {e}")
            print("  The service will only use rule-based predictions.")
    
    
    def _load_scaler(self):
        """Load the pre-fitted StandardScaler"""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Loaded scaler from: {self.scaler_path}")
            else:
                print(f"⚠ Warning: Scaler not found at {self.scaler_path}")
                print("  ML predictions may be unreliable without proper scaling.")
        except Exception as e:
            print(f"⚠ Error loading scaler: {e}")
    
    
    def _load_tfidf_vectorizer(self):
        """Load the pre-fitted TF-IDF vectorizer"""
        try:
            if os.path.exists(self.tfidf_path):
                with open(self.tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                print(f"✓ Loaded TF-IDF vectorizer from: {self.tfidf_path}")
            else:
                print(f"⚠ Warning: TF-IDF vectorizer not found at {self.tfidf_path}")
        except Exception as e:
            print(f"⚠ Error loading TF-IDF vectorizer: {e}")
    
    
    def _load_feature_order(self):
        """Load the saved feature order to ensure consistency"""
        try:
            if os.path.exists(self.feature_order_path):
                with open(self.feature_order_path, 'rb') as f:
                    self.feature_order = pickle.load(f)
                print(f"✓ Loaded feature order ({len(self.feature_order)} features) from: {self.feature_order_path}")
            else:
                print(f"⚠ Warning: Feature order not found at {self.feature_order_path}")
                print("  Predictions may be unreliable if feature order doesn't match training.")
        except Exception as e:
            print(f"⚠ Error loading feature order: {e}")
    
    
    def _load_shap_background(self):
        """Load background data for SHAP explanations"""
        if not SHAP_AVAILABLE:
            print("⚠ SHAP library not available — skipping SHAP background loading.")
            return
        try:
            if os.path.exists(self.shap_background_path):
                self.shap_background_data = pd.read_csv(self.shap_background_path)
                # Remove acuity column if present
                if 'acuity' in self.shap_background_data.columns:
                    self.shap_background_data = self.shap_background_data.drop(columns=['acuity'])
                print(f"✓ Loaded SHAP background data: {self.shap_background_data.shape}")
                
                # Initialize SHAP explainer if model is available
                if self.ml_model is not None:
                    print("  Initializing SHAP explainer...")
                    self.shap_explainer = shap.Explainer(
                        self.ml_model.predict_proba,
                        self.shap_background_data
                    )
                    print("  ✓ SHAP explainer ready")
            else:
                print(f"⚠ Warning: SHAP background data not found at {self.shap_background_path}")
                print("  SHAP explanations will not be available.")
        except Exception as e:
            print(f"⚠ Error loading SHAP background: {e}")
            print("  SHAP explanations will not be available.")
    
    
    def _prepare_rule_based_features(self, patient_data: Dict) -> pd.Series:
        """
        Prepare features for rule-based (NEWS2) prediction
        
        Args:
            patient_data: Patient data (UI validated)
        
        Returns:
            pandas Series with required vital signs (unscaled)
        """
        features = {
            'resprate': patient_data['resprate'],
            'o2sat': patient_data['o2sat'],
            'temperature': patient_data['temperature'],
            'heartrate': patient_data['heartrate'],
            'sbp': patient_data['sbp']
        }
        return pd.Series(features)
    
    
    def _prepare_ml_features(self, patient_data: Dict) -> pd.DataFrame:
        """
        Prepare features for ML prediction (scaled vitals + TF-IDF text features)
        
        Args:
            patient_data: Patient data (UI validated)
        
        Returns:
            pandas DataFrame with all features ready for ML model
        """
        # Extract vitals
        vitals = pd.DataFrame([{
            'temperature': patient_data['temperature'],
            'heartrate': patient_data['heartrate'],
            'resprate': patient_data['resprate'],
            'o2sat': patient_data['o2sat'],
            'sbp': patient_data['sbp'],
            'dbp': patient_data['dbp'],
            'pain': patient_data['pain']
        }])
        
        # Scale vitals using the pre-fitted scaler from training
        if self.scaler is not None:
            vitals_scaled = pd.DataFrame(
                self.scaler.transform(vitals),
                columns=vitals.columns
            )
        else:
            # Fallback: use unscaled (will produce unreliable predictions)
            vitals_scaled = vitals
            print("⚠ Warning: Using unscaled vitals - predictions may be unreliable")
        
        # Process chief complaint text
        if self.tfidf_vectorizer is not None:
            chief_complaint = patient_data.get('chiefcomplaint', 'unknown')
            cleaned_text = clean_text(chief_complaint)
            
            # Transform using pre-fitted TF-IDF vectorizer
            text_features = self.tfidf_vectorizer.transform([cleaned_text])
            feature_names = [f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()]
            text_df = pd.DataFrame(text_features.toarray(), columns=feature_names)
            
            # Combine vitals and text features
            ml_features = pd.concat([vitals_scaled, text_df], axis=1)
        else:
            # Use only vitals if TF-IDF not available
            ml_features = vitals_scaled
        
        # Reorder features to match training order (critical for correct predictions)
        if self.feature_order is not None:
            # Ensure all required features are present
            missing_features = set(self.feature_order) - set(ml_features.columns)
            if missing_features:
                print(f"⚠ Warning: Missing features: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    ml_features[feat] = 0
            
            # Reorder to match training
            ml_features = ml_features[self.feature_order]
        
        return ml_features
    
    
    def _compute_shap_values(self, ml_features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for the given features
        
        Args:
            ml_features: Prepared ML features
        
        Returns:
            DataFrame with feature names and SHAP values
        """
        if self.shap_explainer is None:
            return None
        
        try:
            # Compute SHAP values
            shap_values = self.shap_explainer(ml_features)
            
            # Get the SHAP values for the predicted class
            # shap_values.values shape: (1, n_features, n_classes)
            # We'll use the values for all classes and get the max absolute impact
            feature_names = ml_features.columns.tolist()
            
            # Get SHAP values across all classes and find max absolute contribution
            shap_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_value': np.abs(shap_values.values[0]).max(axis=1)  # Max across classes
            })
            
            # Sort by absolute importance
            shap_importance = shap_importance.sort_values('shap_value', ascending=False)
            
            return shap_importance
        except Exception as e:
            print(f"⚠ Error computing SHAP values: {e}")
            return None
    
    
    def _generate_news2_explanation(self, patient_data: Dict, 
                                    total_score: int, 
                                    individual_scores: Dict) -> str:
        """
        Generate human-readable explanation for NEWS2 rule-based prediction
        
        Args:
            patient_data: Original patient data
            total_score: Total NEWS2 score
            individual_scores: Individual component scores
        
        Returns:
            Formatted explanation string
        """
        explanation = f"**Rule-Based (NEWS2) Assessment**\n\n"
        explanation += f"Total NEWS2 Score: **{total_score}**\n\n"
        explanation += "**Component Breakdown:**\n\n"
        
        # Respiration Rate
        resp_val = patient_data['resprate']
        resp_score = individual_scores['resp_rate']
        explanation += f"• **Respiration Rate**: {resp_val:.1f} breaths/min → Score: {resp_score}\n"
        if resp_score >= 3:
            explanation += "  ⚠️ Critically abnormal\n"
        elif resp_score > 0:
            explanation += "  ⚠️ Abnormal\n"
        
        # Oxygen Saturation
        o2_val = patient_data['o2sat']
        o2_score = individual_scores['o2sat']
        explanation += f"• **Oxygen Saturation**: {o2_val:.1f}% → Score: {o2_score}\n"
        if o2_score >= 3:
            explanation += "  ⚠️ Critically low\n"
        elif o2_score > 0:
            explanation += "  ⚠️ Below normal\n"
        
        # Temperature
        temp_val = patient_data['temperature']
        temp_score = individual_scores['temperature']
        temp_c = (temp_val - 32) * 5/9  # Convert to Celsius for display
        explanation += f"• **Temperature**: {temp_val:.1f}°F ({temp_c:.1f}°C) → Score: {temp_score}\n"
        if temp_score >= 2:
            explanation += "  ⚠️ High fever or hypothermia\n"
        elif temp_score > 0:
            explanation += "  ⚠️ Abnormal\n"
        
        # Blood Pressure
        sbp_val = patient_data['sbp']
        sbp_score = individual_scores['sbp']
        explanation += f"• **Systolic Blood Pressure**: {sbp_val:.1f} mmHg → Score: {sbp_score}\n"
        if sbp_score >= 3:
            explanation += "  ⚠️ Critically abnormal\n"
        elif sbp_score > 0:
            explanation += "  ⚠️ Abnormal\n"
        
        # Heart Rate
        hr_val = patient_data['heartrate']
        hr_score = individual_scores['heart_rate']
        explanation += f"• **Heart Rate**: {hr_val:.1f} bpm → Score: {hr_score}\n"
        if hr_score >= 3:
            explanation += "  ⚠️ Critically abnormal\n"
        elif hr_score > 0:
            explanation += "  ⚠️ Abnormal\n"
        
        explanation += f"\n**Clinical Interpretation:**\n"
        if total_score >= 7:
            explanation += "• NEWS2 ≥7 indicates **HIGH RISK** - Immediate clinical attention required\n"
            explanation += "• Patient classified as **CRITICAL**\n"
        elif total_score >= 5:
            explanation += "• NEWS2 5-6 indicates **MEDIUM RISK** - Urgent assessment needed\n"
        else:
            max_component = max(individual_scores.values())
            if max_component >= 3:
                explanation += "• Any component score ≥3 warrants urgent review\n"
        
        return explanation
    
    
    def _generate_ml_explanation(self, patient_data: Dict, 
                                 prediction_class: int,
                                 confidence: float,
                                 shap_importance: pd.DataFrame = None) -> str:
        """
        Generate explanation for ML model prediction
        
        Args:
            patient_data: Original patient data
            prediction_class: Predicted class
            confidence: Prediction confidence
        
        Returns:
            Formatted explanation string
        """
        explanation = f"**Machine Learning Assessment**\n\n"
        explanation += f"Model: Stacking Ensemble (Random Forest + XGBoost + Neural Network)\n"
        explanation += f"Prediction Confidence: {confidence:.1%}\n\n"
        
        # Chief complaint
        chief_complaint = patient_data.get('chiefcomplaint', 'unknown')
        if chief_complaint != 'unknown':
            explanation += f"**Chief Complaint:** {chief_complaint}\n\n"
        
        # Add SHAP feature importance if available
        if shap_importance is not None:
            explanation += "**Top Contributing Features (SHAP Analysis):**\n\n"
            # Map technical feature names to readable names
            feature_map = {
                'temperature': 'Temperature',
                'heartrate': 'Heart Rate',
                'resprate': 'Respiration Rate',
                'o2sat': 'Oxygen Saturation',
                'sbp': 'Systolic BP',
                'dbp': 'Diastolic BP',
                'pain': 'Pain Level'
            }
            
            # Show top 5 features (vitals only, not TF-IDF)
            top_vitals = shap_importance[
                shap_importance['feature'].isin(self.vital_features)
            ].head(5)
            
            for idx, row in top_vitals.iterrows():
                feat_name = feature_map.get(row['feature'], row['feature'])
                impact = row['shap_value']
                explanation += f"• {feat_name}: Impact score {impact:.3f}\n"
        
        explanation += f"\n**Model Decision:**\n"
        explanation += f"Based on the ensemble of multiple machine learning algorithms, "
        explanation += f"the patient's vital signs and clinical presentation suggest "
        explanation += f"**{self.class_labels[prediction_class]}** priority level.\n"
        
        return explanation
    
    
    def predict_with_explanation(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Main prediction method - makes triage prediction and generates explanation
        
        Args:
            patient_data: Dictionary containing patient information with keys:
                - temperature: body temperature (Fahrenheit)
                - heartrate: heart rate (bpm)
                - resprate: respiration rate (breaths/min)
                - o2sat: oxygen saturation (%)
                - sbp: systolic blood pressure (mmHg)
                - dbp: diastolic blood pressure (mmHg)
                - pain: pain level (0-10)
                - chiefcomplaint: text description of complaint (optional)
        
        Returns:
            Dictionary with:
                - prediction_class: int (1=Critical, 2=Urgent, 3=Non-Urgent)
                - prediction_label: str (class name)
                - confidence: float (0-1)
                - prediction_source: str ('Rule-Based' or 'ML')
                - explanation: str (detailed explanation)
                - news2_score: int (total NEWS2 score)
                - patient_summary: dict (patient data for display)
        """
        # Step 1: Apply rule-based (NEWS2) prediction
        rule_features = self._prepare_rule_based_features(patient_data)
        rule_prediction = rule_based_triage(rule_features)
        
        # Get NEWS2 scores for explanation
        total_score, max_score, individual_scores = calculate_news2_score(rule_features)
        
        # Step 2: Decide on prediction source
        if rule_prediction == 1:
            # Critical case identified by rules - use rule-based prediction
            final_prediction = 1
            prediction_source = 'Rule-Based'
            confidence = 1.0 if total_score >= 7 else 0.9
            explanation = self._generate_news2_explanation(
                patient_data, total_score, individual_scores
            )
        else:
            # Non-critical case - use ML model if available
            if self.ml_model is not None:
                ml_features = self._prepare_ml_features(patient_data)
                ml_prediction_0indexed = self.ml_model.predict(ml_features)[0]
                final_prediction = ml_prediction_0indexed + 1  # Convert to 1-indexed
                
                # Get prediction probabilities for confidence
                try:
                    probabilities = self.ml_model.predict_proba(ml_features)[0]
                    # ml_prediction_0indexed could be an int or numpy type
                    pred_idx = int(ml_prediction_0indexed)
                    if pred_idx < len(probabilities):
                        confidence = float(probabilities[pred_idx])
                    else:
                        # Find the probability for the predicted class from model's classes_
                        class_list = list(self.ml_model.classes_)
                        if pred_idx in class_list:
                            confidence = float(probabilities[class_list.index(pred_idx)])
                        else:
                            confidence = float(max(probabilities))
                    print(f"   Probabilities: {probabilities}")
                    print(f"   Predicted index: {pred_idx}, Confidence: {confidence:.3f}")
                except Exception as e:
                    print(f"⚠ predict_proba failed: {e}")
                    import traceback
                    traceback.print_exc()
                    confidence = 0.75  # Default confidence if probabilities not available
                
                # Compute SHAP values for explanation
                shap_importance = self._compute_shap_values(ml_features)
                
                prediction_source = 'ML'
                explanation = self._generate_ml_explanation(
                    patient_data, final_prediction, confidence, shap_importance
                )
            else:
                # Fallback to rule-based if ML model not available
                final_prediction = rule_prediction
                prediction_source = 'Rule-Based (Fallback)'
                confidence = 0.7
                explanation = self._generate_news2_explanation(
                    patient_data, total_score, individual_scores
                )
        
        # Step 3: Prepare result
        result = {
            'prediction_class': int(final_prediction),
            'prediction_label': self.class_labels[final_prediction],
            'confidence': float(confidence),
            'prediction_source': prediction_source,
            'explanation': explanation,
            'news2_score': int(total_score),
            'patient_summary': {
                'temperature': round(patient_data['temperature'], 1),
                'heartrate': round(patient_data['heartrate'], 1),
                'resprate': round(patient_data['resprate'], 1),
                'o2sat': round(patient_data['o2sat'], 1),
                'sbp': round(patient_data['sbp'], 1),
                'dbp': round(patient_data['dbp'], 1),
                'pain': int(patient_data['pain']),
                'chiefcomplaint': patient_data.get('chiefcomplaint', 'unknown')
            }
        }
        
        return result


# Convenience function for single predictions
def predict_patient_triage(patient_data: Dict) -> Dict[str, Any]:
    """
    Convenience function to make a single prediction
    
    Args:
        patient_data: Patient information dictionary with vitals and chief complaint
    
    Returns:
        Prediction result dictionary
    
    Example:
        >>> patient = {
        ...     'temperature': 100.4,
        ...     'heartrate': 110,
        ...     'resprate': 24,
        ...     'o2sat': 92,
        ...     'sbp': 95,
        ...     'dbp': 65,
        ...     'pain': 7,
        ...     'chiefcomplaint': 'chest pain shortness of breath'
        ... }
        >>> result = predict_patient_triage(patient)
        >>> print(f"Predicted class: {result['prediction_label']}")
        >>> print(result['explanation'])
    """
    service = TriagePredictionService()
    return service.predict_with_explanation(patient_data)


def get_patient_input():
    """
    Get patient data from user input
    
    Returns:
        dict: Patient data dictionary with vitals and chief complaint
    """
    print("=" * 70)
    print("PATIENT TRIAGE PREDICTION SERVICE")
    print("=" * 70)
    print("\nPlease enter patient information:")
    print("-" * 70)
    
    patient_data = {}
    
    # Get vital signs with validation
    while True:
        try:
            patient_data['temperature'] = float(input("Temperature (°F, e.g., 98.6): "))
            if 91.4 <= patient_data['temperature'] <= 107.6:
                break
            print("  ⚠ Temperature should be between 91.4-107.6°F. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    while True:
        try:
            patient_data['heartrate'] = float(input("Heart Rate (bpm, e.g., 75): "))
            if 10 <= patient_data['heartrate'] <= 300:
                break
            print("  ⚠ Heart rate should be between 10-300 bpm. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    while True:
        try:
            patient_data['resprate'] = float(input("Respiratory Rate (breaths/min, e.g., 16): "))
            if 3 <= patient_data['resprate'] <= 60:
                break
            print("  ⚠ Respiratory rate should be between 3-60 breaths/min. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    while True:
        try:
            patient_data['o2sat'] = float(input("Oxygen Saturation (%, e.g., 98): "))
            if 60 <= patient_data['o2sat'] <= 100:
                break
            print("  ⚠ O2 saturation should be between 60-100%. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    while True:
        try:
            patient_data['sbp'] = float(input("Systolic Blood Pressure (mmHg, e.g., 120): "))
            if 30 <= patient_data['sbp'] <= 300:
                break
            print("  ⚠ Systolic BP should be between 30-300 mmHg. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    while True:
        try:
            patient_data['dbp'] = float(input("Diastolic Blood Pressure (mmHg, e.g., 80): "))
            if 30 <= patient_data['dbp'] <= 300:
                break
            print("  ⚠ Diastolic BP should be between 30-300 mmHg. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    while True:
        try:
            patient_data['pain'] = float(input("Pain Level (0-10, e.g., 5): "))
            if 0 <= patient_data['pain'] <= 10:
                break
            print("  ⚠ Pain level should be between 0-10. Please try again.")
        except ValueError:
            print("  ⚠ Invalid input. Please enter a number.")
    
    # Get chief complaint (text input)
    patient_data['chiefcomplaint'] = input("Chief Complaint (e.g., 'chest pain, shortness of breath'): ").strip()
    if not patient_data['chiefcomplaint']:
        patient_data['chiefcomplaint'] = "none"
    
    return patient_data


if __name__ == "__main__":
    while True:
        # Get patient input from user
        patient_data = get_patient_input()
        
        # Make prediction
        print("\n" + "=" * 70)
        print("PROCESSING PREDICTION...")
        print("=" * 70)
        
        result = predict_patient_triage(patient_data)
        
        # Display results
        print("\n" + "=" * 70)
        print("TRIAGE PREDICTION RESULT")
        print("=" * 70)
        print(f"\nPrediction: {result['prediction_label']} (Class {result['prediction_class']})")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Source: {result['prediction_source']}")
        print(f"NEWS2 Score: {result['news2_score']}")
        print(f"\n{result['explanation']}")
        print("\n" + "=" * 70)
        
        # Ask if user wants to continue
        another = input("\nPredict another patient? (y/n): ").strip().lower()
        if another != 'y':
            print("\nThank you for using the Patient Triage Prediction Service!")
            break
