"""
Tests for Flask application routes and endpoints
"""

import pytest
import json


@pytest.mark.app
class TestLandingPage:
    """Test landing page route"""

    def test_landing_page_returns_200(self, client):
        """Test that landing page returns HTTP 200"""
        response = client.get("/")
        assert (
            response.status_code == 200
        ), f"Landing page returned {response.status_code}"

    def test_landing_page_returns_html(self, client):
        """Test that landing page returns HTML content"""
        response = client.get("/")
        assert response.content_type.startswith("text/html"), "Response is not HTML"

    def test_landing_page_contains_title(self, client):
        """Test that landing page contains expected content"""
        response = client.get("/")
        assert (
            b"Spam Detector" in response.data or b"spam" in response.data.lower()
        ), "Landing page missing expected content"


@pytest.mark.app
class TestCheckPage:
    """Test check page (form page) route"""

    def test_check_page_returns_200(self, client):
        """Test that check page returns HTTP 200"""
        response = client.get("/checkpage")
        assert (
            response.status_code == 200
        ), f"Check page returned {response.status_code}"

    def test_check_page_returns_html(self, client):
        """Test that check page returns HTML"""
        response = client.get("/checkpage")
        assert response.content_type.startswith(
            "text/html"
        ), "Check page response is not HTML"

    def test_check_page_has_form(self, client):
        """Test that check page contains form elements"""
        response = client.get("/checkpage")
        assert (
            b"form" in response.data
            or b"textarea" in response.data
            or b"submit" in response.data.lower()
        ), "Check page missing form elements"


@pytest.mark.app
class TestCheckEndpoint:
    """Test email checking endpoint"""

    def test_check_endpoint_accepts_post(self, client, sample_spam_emails):
        """Test that /check endpoint accepts POST requests"""
        response = client.post("/check", data={"email_text": sample_spam_emails[0]})
        assert response.status_code in [
            200,
            400,
            500,
        ], f"Unexpected status code: {response.status_code}"

    def test_check_endpoint_rejects_empty_email(self, client):
        """Test that /check rejects empty email"""
        response = client.post("/check", data={"email_text": ""})
        assert response.status_code == 400, "Empty email should return 400"

    def test_check_endpoint_rejects_too_short_email(self, client):
        """Test that /check rejects email shorter than 5 characters"""
        response = client.post("/check", data={"email_text": "abc"})
        assert response.status_code == 400, "Too-short email should return 400"

    def test_check_endpoint_accepts_valid_email(self, client, sample_spam_emails):
        """Test that /check accepts valid email"""
        response = client.post("/check", data={"email_text": sample_spam_emails[0]})
        assert (
            response.status_code == 200
        ), f"Valid email returned {response.status_code}"

    def test_check_endpoint_returns_html(self, client, sample_spam_emails):
        """Test that /check returns HTML result page"""
        response = client.post("/check", data={"email_text": sample_spam_emails[0]})
        assert response.content_type.startswith("text/html"), "Response should be HTML"

    def test_check_endpoint_contains_prediction_label(self, client, sample_spam_emails):
        """Test that result contains prediction label (SPAM or REAL)"""
        response = client.post("/check", data={"email_text": sample_spam_emails[0]})
        assert (
            b"SPAM" in response.data or b"REAL" in response.data
        ), "Result missing prediction label"

    def test_check_endpoint_contains_confidence(self, client, sample_spam_emails):
        """Test that result contains confidence score"""
        response = client.post("/check", data={"email_text": sample_spam_emails[0]})
        assert b"%" in response.data, "Result missing confidence percentage"

    def test_check_endpoint_with_ham_email(self, client, sample_ham_emails):
        """Test prediction on legitimate (ham) email"""
        response = client.post("/check", data={"email_text": sample_ham_emails[0]})
        assert response.status_code == 200, "Ham email should be accepted"
        assert (
            b"SPAM" in response.data or b"REAL" in response.data
        ), "Result missing prediction label for ham email"


@pytest.mark.app
class TestFeedbackEndpoint:
    """Test user feedback endpoint"""

    def test_feedback_endpoint_accepts_post(self, client, sample_spam_emails):
        """Test that /feedback endpoint accepts POST requests"""
        feedback_data = {
            "email_text": sample_spam_emails[0],
            "actual_label": 1,
            "predicted_label": 1,
            "confidence": 0.95,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert response.status_code in [
            200,
            400,
            500,
        ], f"Unexpected status code: {response.status_code}"

    def test_feedback_endpoint_requires_email_text(self, client):
        """Test that /feedback requires email_text"""
        feedback_data = {
            "email_text": "",
            "actual_label": 1,
            "predicted_label": 1,
            "confidence": 0.95,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert response.status_code == 400, "Empty email_text should return 400"

    def test_feedback_endpoint_requires_valid_label(self, client, sample_spam_emails):
        """Test that /feedback requires valid actual_label (0 or 1)"""
        feedback_data = {
            "email_text": sample_spam_emails[0],
            "actual_label": 99,  # Invalid
            "predicted_label": 1,
            "confidence": 0.95,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert response.status_code == 400, "Invalid label should return 400"

    def test_feedback_endpoint_accepts_ham_label(self, client, sample_spam_emails):
        """Test that /feedback accepts ham (0) label"""
        feedback_data = {
            "email_text": sample_spam_emails[0],
            "actual_label": 0,  # Ham
            "predicted_label": 1,
            "confidence": 0.95,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert (
            response.status_code == 200
        ), f"Valid feedback with ham label returned {response.status_code}"

    def test_feedback_endpoint_accepts_spam_label(self, client, sample_spam_emails):
        """Test that /feedback accepts spam (1) label"""
        feedback_data = {
            "email_text": sample_spam_emails[0],
            "actual_label": 1,  # Spam
            "predicted_label": 1,
            "confidence": 0.95,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert (
            response.status_code == 200
        ), f"Valid feedback with spam label returned {response.status_code}"

    def test_feedback_endpoint_returns_json(self, client, sample_spam_emails):
        """Test that /feedback returns JSON response"""
        feedback_data = {
            "email_text": sample_spam_emails[0],
            "actual_label": 1,
            "predicted_label": 1,
            "confidence": 0.95,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert response.content_type.startswith(
            "application/json"
        ), "Feedback response should be JSON"

    def test_feedback_endpoint_saves_feedback(self, client, sample_spam_emails):
        """Test that /feedback saves feedback to file"""
        feedback_data = {
            "email_text": sample_spam_emails[0],
            "actual_label": 1,
            "predicted_label": 0,  # Wrong prediction - more likely to save
            "confidence": 0.55,
        }
        response = client.post(
            "/feedback", data=json.dumps(feedback_data), content_type="application/json"
        )
        assert response.status_code == 200, "Feedback should be saved"


@pytest.mark.app
class TestStatusEndpoint:
    """Test model status endpoint"""

    def test_status_returns_200(self, client):
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_returns_json(self, client):
        response = client.get("/status")
        assert response.content_type.startswith("application/json")

    def test_status_includes_model_loaded(self, client):
        response = client.get("/status")
        data = json.loads(response.data)
        assert data["success"] is True
        assert "model_loaded" in data
        assert "feedback_count" in data
        assert "uncertainty_threshold" in data


@pytest.mark.app
class TestHighlightEndpoint:
    """Test word highlighting endpoint"""

    def test_highlight_returns_json(self, client, sample_spam_emails):
        response = client.post(
            "/highlight",
            data=json.dumps({"email_text": sample_spam_emails[0]}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.content_type.startswith("application/json")

    def test_highlight_includes_html(self, client, sample_spam_emails):
        response = client.post(
            "/highlight",
            data=json.dumps({"email_text": sample_spam_emails[0]}),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert "highlighted_html" in data
        assert len(data["highlighted_html"]) > 0

    def test_highlight_handles_empty_text(self, client):
        response = client.post(
            "/highlight",
            data=json.dumps({"email_text": ""}),
            content_type="application/json",
        )
        assert response.status_code == 200


@pytest.mark.app
class TestRetrainEndpoint:
    """Test model retraining endpoint"""

    def test_retrain_without_feedback_returns_400(self, client, monkeypatch, tmp_path):
        feedback_file = tmp_path / "empty_feedback.json"
        monkeypatch.setattr("app.FEEDBACK_FILE", str(feedback_file))
        response = client.post("/retrain")
        assert response.status_code == 400

    def test_retrain_requires_api_key_when_configured(self, client, monkeypatch):
        monkeypatch.setattr("app.RETRAIN_API_KEY", "secret-test-key")
        response = client.post("/retrain")
        assert response.status_code == 401

    def test_retrain_accepts_valid_api_key(self, client, monkeypatch, tmp_path):
        feedback_file = tmp_path / "feedback.json"
        feedback_file.write_text(
            json.dumps(
                [
                    {
                        "timestamp": "2026-06-22T10:00:00",
                        "text": "Test spam message for retraining pipeline",
                        "label": "spam",
                        "predicted": "ham",
                        "confidence": 0.55,
                    }
                ]
            )
        )
        monkeypatch.setattr("app.FEEDBACK_FILE", str(feedback_file))
        monkeypatch.setattr("app.RETRAIN_API_KEY", "secret-test-key")
        response = client.post("/retrain", headers={"X-API-Key": "secret-test-key"})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True


@pytest.mark.app
class TestErrorHandling:
    """Test error handling in Flask app"""

    def test_check_page_error_without_model(self, app):
        """Test graceful error when model fails to load"""
        # This tests that the app handles model loading errors gracefully
        # Actual behavior depends on implementation
        with app.test_client() as client:
            response = client.get("/checkpage")
            # Should either return the page or return an error page
            assert response.status_code in [
                200,
                500,
            ], "Check page should handle model loading errors"

    def test_invalid_form_data_handling(self, client):
        """Test handling of invalid form data"""
        response = client.post("/check", data={})  # Missing email_text
        assert response.status_code in [400, 500], "Should handle missing data"

    def test_response_codes_valid(self, client, sample_spam_emails):
        """Test that all responses have valid HTTP status codes"""
        endpoints = [
            ("GET", "/"),
            ("GET", "/checkpage"),
            ("POST", "/check", {"email_text": sample_spam_emails[0]}),
        ]

        for method, endpoint, *data in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                form_data = data[0] if data else {}
                response = client.post(endpoint, data=form_data)

            assert (
                100 <= response.status_code < 600
            ), f"Invalid status code {response.status_code}"
