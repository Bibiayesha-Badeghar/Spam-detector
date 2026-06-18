"""
SPAM DETECTOR - MODEL TRAINING SCRIPT
======================================
Trains a spam detection model with proper train/test split and evaluation metrics.
Includes feedback-based retraining capability for continuous learning.
"""

import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def train_model(include_feedback=True):
    """
    Train or retrain the spam detection model.
    
    Args:
        include_feedback (bool): If True, includes user feedback data if available
    
    Returns:
        tuple: (model, vectorizer, accuracy, metrics_dict)
    """
    
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    # Load main dataset
    with open('dataset.json', 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    labels = [1 if item['label'] == 'spam' else 0 for item in data]
    
    # Load user feedback if retraining
    feedback_texts = []
    feedback_labels = []
    if include_feedback and os.path.exists('user_feedback.json'):
        try:
            with open('user_feedback.json', 'r') as f:
                feedback_data = json.load(f)
                feedback_texts = [item['text'] for item in feedback_data]
                feedback_labels = [1 if item['label'] == 'spam' else 0 for item in feedback_data]
                print(f"Loaded {len(feedback_data)} user feedback samples")
        except Exception as e:
            print(f"Could not load feedback data: {e}")
    
    # Combine datasets
    all_texts = texts + feedback_texts
    all_labels = labels + feedback_labels
    
    print(f"Total training samples: {len(all_texts)}")
    print(f"   - Original dataset: {len(texts)}")
    print(f"   - User feedback: {len(feedback_texts)}")
    
    # Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"\nDATA SPLIT")
    print(f"   - Training set: {len(X_train)} samples")
    print(f"   - Test set: {len(X_test)} samples")
    print(f"   - Spam ratio: {sum(all_labels)/len(all_labels)*100:.1f}%")
    
    # Vectorize text using TF-IDF
    print("\n" + "="*60)
    print("VECTORIZING TEXT")
    print("="*60)
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Vectorized {len(vectorizer.get_feature_names_out())} features")
    
    # Train classifier
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        class_weight='balanced'  # Handle imbalanced datasets
    )
    model.fit(X_train_vec, y_train)
    print("Model training complete")
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nMETRICS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save model and vectorizer
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    with open('spam_detector_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model saved: spam_detector_model.pkl")
    print("Vectorizer saved: vectorizer.pkl")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(all_texts),
        'test_samples': len(X_test),
        'feedback_samples': len(feedback_texts)
    }
    
    return model, vectorizer, accuracy, metrics

if __name__ == "__main__":
    model, vectorizer, accuracy, metrics = train_model(include_feedback=True)
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model ready for predictions with {metrics['accuracy']*100:.2f}% accuracy!")
    print(f"Total training samples used: {metrics['total_samples']}")
