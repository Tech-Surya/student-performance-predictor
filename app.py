from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

app = Flask(__name__)

class StudentPerformancePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['study_hours', 'attendance', 'previous_score', 'internet_access', 'extracurricular']
        self.feature_importance = None
        
    def create_synthetic_dataset(self, n_samples=1000):
        """Create a synthetic dataset for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic data
        study_hours = np.random.normal(15, 5, n_samples)
        study_hours = np.clip(study_hours, 0, 40)
        
        attendance = np.random.normal(80, 15, n_samples)
        attendance = np.clip(attendance, 0, 100)
        
        previous_score = np.random.normal(75, 15, n_samples)
        previous_score = np.clip(previous_score, 0, 100)
        
        internet_access = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        extracurricular = np.random.poisson(2, n_samples)
        extracurricular = np.clip(extracurricular, 0, 10)
        
        # Create target variable with logical relationships
        pass_probability = (
            0.02 * study_hours +
            0.01 * attendance +
            0.008 * previous_score +
            0.1 * internet_access +
            0.05 * extracurricular +
            np.random.normal(0, 0.1, n_samples) - 1.5
        )
        
        target = (pass_probability > 0).astype(int)
        
        df = pd.DataFrame({
            'study_hours': study_hours,
            'attendance': attendance,
            'previous_score': previous_score,
            'internet_access': internet_access,
            'extracurricular': extracurricular,
            'pass': target
        })
        
        return df
    
    def train_model(self):
        """Train the prediction model"""
        # Create synthetic dataset
        df = self.create_synthetic_dataset()
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['pass']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        
        # Save model and scaler
        joblib.dump(self.model, 'model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return accuracy
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load('model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            
            # Recreate feature importance if model exists
            if self.model and hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, features):
        """Make prediction for given features"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        feature_array = np.array(features).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_scaled)[0]
        confidence = self.model.predict_proba(feature_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'confidence': float(max(confidence)) * 100,
            'pass_probability': float(confidence[1]) * 100,
            'fail_probability': float(confidence[0]) * 100
        }
    
    def get_feature_importance(self):
        """Get feature importance for visualization"""
        if self.feature_importance is None:
            return {}
        return self.feature_importance
    
    def get_recommendations(self, features, prediction_result):
        """Generate personalized recommendations"""
        study_hours, attendance, previous_score, internet_access, extracurricular = features
        
        recommendations = []
        
        if study_hours < 10:
            recommendations.append("📚 Increase study hours to at least 10-15 hours per week for better performance.")
        
        if attendance < 75:
            recommendations.append("🎯 Improve attendance to at least 75% to stay on track with coursework.")
        
        if previous_score < 60:
            recommendations.append("📖 Focus on understanding fundamental concepts and seek help from teachers.")
        
        if internet_access == 0:
            recommendations.append("🌐 Consider accessing internet resources for additional learning materials.")
        
        if extracurricular < 1:
            recommendations.append("🏃‍♂️ Participate in at least 1-2 extracurricular activities for balanced development.")
        elif extracurricular > 5:
            recommendations.append("⚖️ Consider reducing extracurricular activities to focus more on studies.")
        
        if prediction_result['prediction'] == 0:  # Fail prediction
            recommendations.append("⚠️ High risk of failure - consider seeking academic counseling and tutoring support.")
        
        if not recommendations:
            recommendations.append("✅ Great job! Keep maintaining your current study habits and performance.")
        
        return recommendations

# Initialize predictor
predictor = StudentPerformancePredictor()

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("[DEBUG] Received JSON:", data)

        if not data:
            raise ValueError("No input data received.")

        features = [
            float(data['study_hours']),
            float(data['attendance']),
            float(data['previous_score']),
            int(data['internet_access']),
            int(data['extracurricular'])
        ]
        print("[DEBUG] Parsed features:", features)

        result = predictor.predict(features)
        recommendations = predictor.get_recommendations(features, result)
        feature_importance = predictor.get_feature_importance()

        response = {
            'success': True,
            'prediction': 'PASS' if result['prediction'] == 1 else 'FAIL',
            'confidence': round(result['confidence'], 2),
            'pass_probability': round(result['pass_probability'], 2),
            'fail_probability': round(result['fail_probability'], 2),
            'recommendations': recommendations,
            'feature_importance': feature_importance,
            'input_data': {
                'study_hours': features[0],
                'attendance': features[1],
                'previous_score': features[2],
                'internet_access': 'Yes' if features[3] == 1 else 'No',
                'extracurricular': features[4]
            }
        }

        return jsonify(response)

    except Exception as e:
        print("[ERROR] Exception in /predict:", str(e))  # Debug info in terminal
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/api/analytics_data')
def analytics_data():
    """Get analytics data for dashboard"""
    try:
        # Generate sample analytics data
        df = predictor.create_synthetic_dataset(1000)
        
        # Distribution data
        pass_count = df['pass'].sum()
        fail_count = len(df) - pass_count
        
        # Feature averages by outcome
        pass_students = df[df['pass'] == 1]
        fail_students = df[df['pass'] == 0]
        
        analytics = {
            'distribution': {
                'pass': int(pass_count),
                'fail': int(fail_count)
            },
            'feature_averages': {
                'pass_students': {
                    'study_hours': float(pass_students['study_hours'].mean()),
                    'attendance': float(pass_students['attendance'].mean()),
                    'previous_score': float(pass_students['previous_score'].mean()),
                    'extracurricular': float(pass_students['extracurricular'].mean())
                },
                'fail_students': {
                    'study_hours': float(fail_students['study_hours'].mean()),
                    'attendance': float(fail_students['attendance'].mean()),
                    'previous_score': float(fail_students['previous_score'].mean()),
                    'extracurricular': float(fail_students['extracurricular'].mean())
                }
            },
            'feature_importance': predictor.get_feature_importance()
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Train or load model
    if not predictor.load_model():
        print("Training new model...")
        predictor.train_model()
        print("Model trained successfully!")
    else:
        print("Model loaded successfully!")

    port = int(os.environ.get("PORT", 5000))  # Use cloud-assigned port, default to 5000
    app.run(host='0.0.0.0', port=port)