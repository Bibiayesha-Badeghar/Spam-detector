"""
SPAM DETECTOR - FLASK WEB APPLICATION
======================================
Interactive web interface for spam detection with self-learning capability.
Features:
  - Email spam detection using trained ML model
  - Form validation and error handling
  - Self-training on user feedback for uncertain predictions
  - Confidence scoring and visual feedback
"""

import os
import json
import pickle
import webbrowser
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuration
UNCERTAINTY_THRESHOLD = 0.60  # Flag predictions below this as uncertain
FEEDBACK_FILE = 'user_feedback.json'

def load_model_and_vectorizer():
    """Load the trained model and vectorizer from disk."""
    try:
        with open('spam_detector_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, None
    except FileNotFoundError as e:
        return None, None, f"Error: Model files not found. Please run train_model.py first. ({str(e)})"
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Load model at startup
model, vectorizer, load_error = load_model_and_vectorizer()

def save_feedback(email_text, actual_label, predicted_label, confidence):
    """Save user feedback for model retraining."""
    feedback_data = []
    
    # Load existing feedback
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_data = json.load(f)
        except Exception:
            feedback_data = []
    
    # Add new feedback
    feedback_data.append({
        'timestamp': datetime.now().isoformat(),
        'text': email_text,
        'label': 'spam' if actual_label == 1 else 'ham',
        'predicted': 'spam' if predicted_label == 1 else 'ham',
        'confidence': float(confidence)
    })
    
    # Save feedback
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        return True, len(feedback_data)
    except Exception as e:
        return False, str(e)

def validate_email_input(email_text):
    """Validate email text input."""
    if not email_text:
        return False, "Email text cannot be empty"
    
    if len(email_text.strip()) < 5:
        return False, "Email text must be at least 5 characters long"
    
    if len(email_text) > 10000:
        return False, "Email text exceeds maximum length (10000 characters)"
    
    return True, "Valid"

# Landing page
@app.route("/")
def landing():
    """Home page with project information."""
    return render_template("landing.html")

# Check page (form to submit email)
@app.route("/checkpage")
def checkpage():
    """Form page for email submission."""
    if load_error:
        return f"<h1>Error: Model not loaded</h1><p>{load_error}</p>", 500
    return render_template("index.html")

# Main spam detection endpoint
@app.route("/check", methods=["POST"])
def check():
    """
    Handle email spam detection request.
    
    Returns:
        HTML page with prediction result, confidence, and feedback option
    """
    if load_error:
        return render_template(
            "result.html",
            email="",
            label="ERROR",
            confidence="0%",
            confidence_value=0,
            predicted_class=-1,
            error=f"Model not loaded: {load_error}"
        ), 500
    
    email_text = request.form.get("email_text", "").strip()
    
    # Validate input
    is_valid, validation_message = validate_email_input(email_text)
    if not is_valid:
        return render_template(
            "result.html",
            email="",
            label="ERROR",
            confidence="0%",
            confidence_value=0,
            predicted_class=-1,
            error=validation_message
        ), 400
    
    try:
        # Make prediction
        X = vectorizer.transform([email_text])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        # Get confidence for the predicted class
        confidence_value = float(max(proba))
        
        # Determine if prediction is uncertain
        is_uncertain = confidence_value < UNCERTAINTY_THRESHOLD
        
        # Format output
        label = "SPAM" if pred == 1 else "REAL"
        confidence = f"{confidence_value * 100:.1f}%"
        
        return render_template(
            "result.html",
            email=email_text,
            label=label,
            confidence=confidence,
            confidence_value=confidence_value,
            is_uncertain=is_uncertain,
            predicted_class=pred,
            threshold=UNCERTAINTY_THRESHOLD
        )
        
    except Exception as e:
        return render_template(
            "result.html",
            email="",
            label="ERROR",
            confidence="0%",
            confidence_value=0,
            predicted_class=-1,
            error=f"Prediction error: {str(e)}"
        ), 500

# Feedback endpoint (AJAX for self-training)
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Handle user feedback on predictions.
    Stores feedback for model retraining.
    
    Expected JSON:
        - email_text: The email that was checked
        - actual_label: What the user says it really is (0=ham, 1=spam)
        - predicted_label: What the model predicted
        - confidence: Model's confidence score
    """
    try:
        data = request.get_json()
        
        email_text = data.get('email_text', '').strip()
        actual_label = int(data.get('actual_label', -1))
        predicted_label = int(data.get('predicted_label', -1))
        confidence = float(data.get('confidence', 0))
        
        # Validate
        if not email_text:
            return jsonify({'success': False, 'error': 'Email text is required'}), 400
        
        if actual_label not in [0, 1]:
            return jsonify({'success': False, 'error': 'Invalid actual label'}), 400
        
        # Save feedback
        success, result = save_feedback(email_text, actual_label, predicted_label, confidence)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Thank you! Feedback saved. Total feedback samples: {result}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Could not save feedback: {result}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing feedback: {str(e)}'
        }), 500

# Retrain endpoint
@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Retrain the model using accumulated user feedback.
    This endpoint is typically called manually or on a schedule.
    """
    global model, vectorizer, load_error
    
    try:
        # Check if feedback exists
        if not os.path.exists(FEEDBACK_FILE):
            return jsonify({
                'success': False,
                'error': 'No feedback data available for retraining'
            }), 400
        
        # Retrain model using train_model.py
        from train_model import train_model
        
        model, vectorizer, accuracy, metrics = train_model(include_feedback=True)
        
        return jsonify({
            'success': True,
            'message': f'Model retrained successfully!',
            'accuracy': accuracy,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Retraining error: {str(e)}'
        }), 500

# Status endpoint
@app.route("/status", methods=["GET"])
def status():
    """Get current model status and feedback statistics."""
    try:
        feedback_count = 0
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_count = len(json.load(f))
        
        return jsonify({
            'model_loaded': model is not None,
            'feedback_count': feedback_count,
            'uncertainty_threshold': UNCERTAINTY_THRESHOLD
        })
    except Exception as e:
        return jsonify({
            'model_loaded': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000/")
    
    print("\n" + "="*60)
    print("SPAM DETECTOR WEB APP STARTING")
    print("="*60)
    print(f"Model loaded: {model is not None}")
    print(f"Uncertainty threshold: {UNCERTAINTY_THRESHOLD*100:.0f}%")
    print(f"Feedback file: {FEEDBACK_FILE}")
    print("\nOpen: http://127.0.0.1:5000/")
    print("="*60 + "\n")
    
    app.run(debug=True, use_reloader=True)
