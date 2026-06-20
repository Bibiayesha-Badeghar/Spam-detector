"""
Pytest fixtures and configuration for Spam Detector tests
"""

import pytest
import json
import sys
from pathlib import Path

# Add parent directory to path to import app and other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app as flask_app  # noqa: E402


@pytest.fixture
def app():
    """Flask application fixture."""
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client fixture."""
    return app.test_client()


@pytest.fixture
def sample_spam_emails():
    """Sample spam emails for testing."""
    return [
        "CONGRATULATIONS! You've won a $500 gift card! Claim now: http://bit.ly/rewards2024",
        "Subject: Urgent: Your bank account has been compromised\n\nClick here to verify your account immediately",
        "FREE MONEY! You qualify for $10,000 cash. Visit us now",
        "Subject: Confirm Your PayPal Account\n\nDear User, please verify your account by clicking the link below",
    ]


@pytest.fixture
def sample_ham_emails():
    """Sample legitimate emails for testing."""
    return [
        "Hi John, Let's schedule a meeting for next Tuesday at 2 PM. How does that work for you?",
        "The quarterly report is ready for review. Please check the attached document.",
        "Your order #12345 has been shipped and will arrive by Friday.",
        "Thanks for the update. I'll review the proposal and get back to you soon.",
    ]


@pytest.fixture
def temp_feedback_file(tmp_path):
    """Create a temporary feedback file for testing."""
    feedback_data = [
        {
            "timestamp": "2026-06-17T10:00:00",
            "text": "Test email",
            "label": "spam",
            "predicted": "spam",
            "confidence": 0.95,
        }
    ]
    feedback_file = tmp_path / "user_feedback_test.json"
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f)
    return feedback_file


@pytest.fixture
def validation_test_cases():
    """Test cases for input validation."""
    return {
        "valid": [
            "This is a legitimate email with proper content",
            "Subject: Meeting Request\n\nPlease join us for a meeting tomorrow at 10 AM",
            "Your order has been confirmed. Thank you for shopping with us!",
        ],
        "invalid": {
            "empty": ("", "Email text cannot be empty"),
            "too_short": ("abc", "Email text must be at least 5 characters long"),
            "too_long": ("a" * 10001, "Email text exceeds maximum length"),
        },
    }
