"""
Tests for model training, loading, and prediction functionality
"""

import pytest
import pickle
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Note: These tests assume model files exist or can be mocked


@pytest.mark.model
class TestModelLoading:
    """Test model and vectorizer loading functionality"""

    def test_model_file_exists(self):
        """Test that model pickle file exists"""
        assert os.path.exists('spam_detector_model.pkl'), "Model file not found"

    def test_vectorizer_file_exists(self):
        """Test that vectorizer pickle file exists"""
        assert os.path.exists('vectorizer.pkl'), "Vectorizer file not found"

    def test_model_loads_successfully(self):
        """Test that model can be loaded from pickle file"""
        try:
            with open('spam_detector_model.pkl', 'rb') as f:
                model = pickle.load(f)
            assert model is not None, "Model loaded as None"
            assert hasattr(model, 'predict'), "Model missing predict method"
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")

    def test_vectorizer_loads_successfully(self):
        """Test that vectorizer can be loaded from pickle file"""
        try:
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            assert vectorizer is not None, "Vectorizer loaded as None"
            assert hasattr(vectorizer, 'transform'), "Vectorizer missing transform method"
        except Exception as e:
            pytest.fail(f"Failed to load vectorizer: {e}")


@pytest.mark.model
class TestPrediction:
    """Test model prediction functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load model and vectorizer for testing"""
        with open('spam_detector_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

    def test_prediction_returns_binary_class(self, sample_spam_emails, sample_ham_emails):
        """Test that predictions are binary (0 or 1)"""
        for email in sample_spam_emails + sample_ham_emails:
            X = self.vectorizer.transform([email])
            pred = self.model.predict(X)[0]
            assert pred in [0, 1], f"Prediction {pred} not binary"

    def test_confidence_scores_valid_range(self, sample_spam_emails):
        """Test that confidence scores are between 0 and 1"""
        for email in sample_spam_emails:
            X = self.vectorizer.transform([email])
            proba = self.model.predict_proba(X)[0]
            confidence = max(proba)
            assert 0 <= confidence <= 1, f"Confidence {confidence} out of range"

    def test_probability_distribution_sums_to_one(self, sample_spam_emails):
        """Test that probability distributions sum to 1"""
        for email in sample_spam_emails:
            X = self.vectorizer.transform([email])
            proba = self.model.predict_proba(X)[0]
            total_prob = sum(proba)
            assert abs(total_prob - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total_prob}"

    def test_vectorizer_transform_shape(self, sample_spam_emails):
        """Test that vectorizer produces correct feature shape"""
        X = self.vectorizer.transform(sample_spam_emails)
        assert X.shape[0] == len(sample_spam_emails), "Vectorizer output row count mismatch"
        assert X.shape[1] > 0, "Vectorizer produced no features"

    def test_multiple_emails_batch_prediction(self, sample_spam_emails, sample_ham_emails):
        """Test batch prediction on multiple emails"""
        all_emails = sample_spam_emails + sample_ham_emails
        X = self.vectorizer.transform(all_emails)
        predictions = self.model.predict(X)
        
        assert len(predictions) == len(all_emails), "Batch prediction count mismatch"
        assert all(p in [0, 1] for p in predictions), "Batch predictions not binary"


@pytest.mark.model
class TestDatasetIntegrity:
    """Test dataset and training data integrity"""

    def test_dataset_file_exists(self):
        """Test that dataset JSON file exists"""
        assert os.path.exists('dataset.json'), "Dataset file not found"

    def test_dataset_is_valid_json(self):
        """Test that dataset is valid JSON"""
        try:
            with open('dataset.json', 'r') as f:
                data = json.load(f)
            assert isinstance(data, list), "Dataset not a list"
            assert len(data) > 0, "Dataset is empty"
        except Exception as e:
            pytest.fail(f"Dataset JSON parsing failed: {e}")

    def test_dataset_has_required_fields(self):
        """Test that dataset samples have required fields"""
        with open('dataset.json', 'r') as f:
            data = json.load(f)
        
        required_fields = ['id', 'label', 'text']
        for sample in data[:10]:  # Check first 10
            for field in required_fields:
                assert field in sample, f"Sample missing required field: {field}"
                assert sample[field], f"Sample has empty {field}"

    def test_dataset_labels_valid(self):
        """Test that all dataset labels are 'spam' or 'ham'"""
        with open('dataset.json', 'r') as f:
            data = json.load(f)
        
        for sample in data:
            assert sample['label'] in ['spam', 'ham'], f"Invalid label: {sample['label']}"


@pytest.mark.model
class TestModelMetrics:
    """Test model metrics and evaluation"""

    def test_model_has_feature_importance(self):
        """Test that Random Forest model has feature importance"""
        with open('spam_detector_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        assert hasattr(model, 'feature_importances_'), "Model missing feature importances"
        assert len(model.feature_importances_) > 0, "No feature importances calculated"

    def test_model_has_n_estimators(self):
        """Test that model has estimators"""
        with open('spam_detector_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        assert hasattr(model, 'estimators_'), "Model missing estimators"
        assert len(model.estimators_) > 0, "Model has no estimators"
