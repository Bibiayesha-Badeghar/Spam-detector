"""
SPAM DETECTOR - MODEL TRAINING SCRIPT
======================================
Trains a spam detection model with proper train/test split and evaluation metrics.
Includes feedback-based retraining capability for continuous learning.
"""

import json
import pickle
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def train_model(include_feedback=True):
    """
    Train or retrain the spam detection model.
    
    Args:
        include_feedback (bool): If True, includes user feedback data if available
    
    Returns:
        tuple: (model, vectorizer, accuracy, metrics_dict)
    """
    
    logger.info("="*60)
    logger.info("LOADING DATASET")
    logger.info("="*60)
    
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
                logger.info(f"Loaded {len(feedback_data)} user feedback samples")
        except json.JSONDecodeError:
            logger.warning("Feedback file contains invalid JSON. Ignoring.")
        except IOError as e:
            logger.error(f"Could not load feedback data: {e}")
    
    # Combine datasets
    all_texts = texts + feedback_texts
    all_labels = labels + feedback_labels
    
    logger.info(f"Total training samples: {len(all_texts)}")
    logger.info(f"   - Original dataset: {len(texts)}")
    logger.info(f"   - User feedback: {len(feedback_texts)}")
    
    # Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    logger.info(f"DATA SPLIT")
    logger.info(f"   - Training set: {len(X_train)} samples")
    logger.info(f"   - Test set: {len(X_test)} samples")
    logger.info(f"   - Spam ratio: {sum(all_labels)/len(all_labels)*100:.1f}%")
    
    # Vectorize text using TF-IDF
    logger.info("="*60)
    logger.info("VECTORIZING TEXT")
    logger.info("="*60)
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info(f"Vectorized {len(vectorizer.get_feature_names_out())} features")
    
    # Train classifier
    logger.info("="*60)
    logger.info("TRAINING MODEL")
    logger.info("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        class_weight='balanced'  # Handle imbalanced datasets
    )
    model.fit(X_train_vec, y_train)
    logger.info("Model training complete")
    
    # Evaluate model
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info("METRICS:")
    logger.info(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    logger.info(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    logger.info(f"   F1-Score:  {f1:.4f}")
    
    logger.info("CLASSIFICATION REPORT:\n%s", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save model and vectorizer
    logger.info("="*60)
    logger.info("SAVING MODEL")
    logger.info("="*60)
    
    with open('spam_detector_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    logger.info("Model saved: spam_detector_model.pkl")
    logger.info("Vectorizer saved: vectorizer.pkl")
    
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
    logger.info("="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Model ready for predictions with {metrics['accuracy']*100:.2f}% accuracy!")
    logger.info(f"Total training samples used: {metrics['total_samples']}")
