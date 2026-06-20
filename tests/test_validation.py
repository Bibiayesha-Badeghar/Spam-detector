"""
Tests for input validation functionality
"""

import pytest
from app import validate_email_input


@pytest.mark.validation
class TestEmailValidation:
    """Test email input validation"""

    def test_valid_email_accepted(self, validation_test_cases):
        """Test that valid emails pass validation"""
        for email in validation_test_cases["valid"]:
            is_valid, message = validate_email_input(email)
            assert is_valid is True, f"Valid email rejected: {email}"
            assert (
                message == "Valid"
            ), f"Valid email returned unexpected message: {message}"

    def test_empty_email_rejected(self, validation_test_cases):
        """Test that empty email is rejected"""
        email, expected_msg = validation_test_cases["invalid"]["empty"]
        is_valid, message = validate_email_input(email)
        assert is_valid is False, "Empty email should be invalid"
        assert (
            "empty" in message.lower()
        ), f"Expected 'empty' in message, got: {message}"

    def test_too_short_email_rejected(self, validation_test_cases):
        """Test that email shorter than 5 chars is rejected"""
        email, expected_msg = validation_test_cases["invalid"]["too_short"]
        is_valid, message = validate_email_input(email)
        assert is_valid is False, "Too-short email should be invalid"
        assert (
            "5 character" in message or "characters" in message
        ), f"Expected character count message, got: {message}"

    def test_too_long_email_rejected(self, validation_test_cases):
        """Test that email longer than 10000 chars is rejected"""
        email, expected_msg = validation_test_cases["invalid"]["too_long"]
        is_valid, message = validate_email_input(email)
        assert is_valid is False, "Too-long email should be invalid"
        assert (
            "exceeds" in message.lower() or "maximum" in message.lower()
        ), f"Expected maximum length message, got: {message}"

    def test_minimum_length_boundary(self):
        """Test boundary case: exactly 5 characters (should pass)"""
        email = "abcde"  # Exactly 5 characters
        is_valid, message = validate_email_input(email)
        assert is_valid is True, "Email with exactly 5 characters should be valid"

    def test_minimum_length_boundary_minus_one(self):
        """Test boundary case: exactly 4 characters (should fail)"""
        email = "abcd"  # Exactly 4 characters
        is_valid, message = validate_email_input(email)
        assert is_valid is False, "Email with 4 characters should be invalid"

    def test_maximum_length_boundary(self):
        """Test boundary case: exactly 10000 characters (should pass)"""
        email = "a" * 10000  # Exactly 10000 characters
        is_valid, message = validate_email_input(email)
        assert is_valid is True, "Email with exactly 10000 characters should be valid"

    def test_maximum_length_boundary_plus_one(self):
        """Test boundary case: exactly 10001 characters (should fail)"""
        email = "a" * 10001  # Exactly 10001 characters
        is_valid, message = validate_email_input(email)
        assert is_valid is False, "Email with 10001 characters should be invalid"

    def test_email_with_whitespace_only(self):
        """Test that email with only whitespace is rejected"""
        email = "     "  # Only whitespace
        is_valid, message = validate_email_input(email)
        assert is_valid is False, "Email with only whitespace should be invalid"

    def test_email_with_newlines(self):
        """Test that email with newlines is handled correctly"""
        email = "Hello\nWorld\nThis is an email"  # Contains newlines
        is_valid, message = validate_email_input(email)
        # Should be valid as long as length requirements are met
        if len(email.strip()) >= 5:
            assert (
                is_valid is True
            ), "Email with newlines should be valid if long enough"

    def test_email_with_special_characters(self):
        """Test that email with special characters is handled"""
        email = "Test@#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        is_valid, message = validate_email_input(email)
        # Should be valid as long as length requirements are met
        if len(email) >= 5 and len(email) <= 10000:
            assert is_valid is True, "Email with special characters should be valid"

    def test_email_with_unicode_characters(self):
        """Test that email with unicode characters is handled"""
        email = "Hello 世界 你好 Привет"  # Unicode characters
        is_valid, message = validate_email_input(email)
        if len(email) >= 5 and len(email) <= 10000:
            assert is_valid is True, "Email with unicode should be valid"

    def test_validation_returns_tuple(self):
        """Test that validation function returns tuple"""
        result = validate_email_input("test email")
        assert isinstance(result, tuple), "Validation should return tuple"
        assert len(result) == 2, "Validation should return tuple of length 2"

    def test_validation_return_types(self):
        """Test that validation returns correct types"""
        is_valid, message = validate_email_input("test email")
        assert isinstance(is_valid, bool), "First return value should be bool"
        assert isinstance(message, str), "Second return value should be string"


@pytest.mark.validation
class TestValidationEdgeCases:
    """Test edge cases in validation"""

    def test_none_input_handling(self):
        """Test handling of None input (if applicable)"""
        try:
            result = validate_email_input(None)
            # If it doesn't raise an error, it should return invalid
            assert result[0] is False, "None should be invalid"
        except (TypeError, AttributeError):
            # It's okay if function raises error on None
            pass

    def test_validation_is_case_insensitive(self):
        """Test that validation doesn't depend on case"""
        email = "This IS a TEST email"
        is_valid, _ = validate_email_input(email)
        assert is_valid is True, "Case shouldn't affect validation"

    def test_html_in_email(self):
        """Test handling of HTML in email text"""
        email = "<html><body><h1>Test Email</h1></body></html>"
        is_valid, message = validate_email_input(email)
        if len(email) >= 5 and len(email) <= 10000:
            assert is_valid is True, "HTML should be treated as regular text"

    def test_sql_like_content_in_email(self):
        """Test handling of SQL-like content"""
        email = "SELECT * FROM users; DROP TABLE users;"
        is_valid, message = validate_email_input(email)
        if len(email) >= 5 and len(email) <= 10000:
            assert is_valid is True, "SQL should be treated as regular text"

    def test_very_long_word_no_spaces(self):
        """Test handling of very long word without spaces"""
        email = "a" * 100  # 100 character word
        is_valid, message = validate_email_input(email)
        assert is_valid is True, "Long word should be valid"
