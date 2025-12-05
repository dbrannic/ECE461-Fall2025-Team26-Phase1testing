"""
Tests specifically designed to hit edge cases and low-coverage branches
in main.py, particularly environment setup, logging, and utility functions.
"""
import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from io import StringIO
import logging

# Adjust path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after path adjustment
import main
from main import (
    validate_environment,
    validate_github_token,
    setup_logging,
    extract_model_name,
    run_batch_evaluation,
    parse_input
)


# --- Test validate_github_token ---

class TestValidateGithubToken:
    """Tests for environment variable validation."""

    @pytest.mark.parametrize("token", ["ghp_a", "github_pat_a", "gho_a", "ghu_a", "ghs_a", "ghr_a"])
    def test_validate_token_valid_prefixes(self, monkeypatch, capsys, token):
        """Test with valid token prefixes (should return token, no warning)."""
        monkeypatch.setenv("GITHUB_TOKEN", token)
        result = validate_github_token()
        assert result == token
        assert "Warning" not in capsys.readouterr().err

    def test_validate_token_invalid_prefix(self, monkeypatch, capsys):
        """Test with an invalid token prefix (should return None, print warning to stderr)."""
        monkeypatch.setenv("GITHUB_TOKEN", "invalid_prefix_token")
        result = validate_github_token()
        assert result is None
        assert "Warning: GitHub token format appears invalid" in capsys.readouterr().err

    def test_validate_token_missing(self, monkeypatch):
        """Test with no GITHUB_TOKEN set."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = validate_github_token()
        assert result is None


# --- Test setup_logging ---

class TestSetupLogging:
    """Tests for logging setup, covering all branches."""

    def test_setup_logging_level_debug(self, monkeypatch):
        """Test log level 2 (DEBUG)."""
        monkeypatch.setenv("LOG_LEVEL", "2")
        logger = setup_logging()
        assert logger.level == logging.DEBUG

    def test_setup_logging_level_info(self, monkeypatch):
        """Test log level 1 (INFO)."""
        monkeypatch.setenv("LOG_LEVEL", "1")
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_setup_logging_level_silent(self, monkeypatch):
        """Test log level 0 (CRITICAL/SILENT)."""
        monkeypatch.setenv("LOG_LEVEL", "0")
        logger = setup_logging()
        assert logger.level == logging.CRITICAL

    def test_setup_logging_invalid_level_number(self, monkeypatch):
        """Test invalid log level number (>2) falls back to CRITICAL."""
        monkeypatch.setenv("LOG_LEVEL", "99")
        logger = setup_logging()
        assert logger.level == logging.CRITICAL

    def test_setup_logging_invalid_level_string(self, monkeypatch):
        """Test invalid log level string falls back to CRITICAL."""
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        logger = setup_logging()
        assert logger.level == logging.CRITICAL

    @patch('main.Path.mkdir', side_effect=PermissionError)
    def test_setup_logging_log_file_path_failure(self, mock_mkdir, monkeypatch, capsys, tmp_path):
        """Test log file creation failure (hitting the try/except block)."""
        log_file_path = tmp_path / "logs" / "test.log"
        monkeypatch.setenv("LOG_FILE", str(log_file_path))
        monkeypatch.setenv("LOG_LEVEL", "2")
        
        logger = setup_logging()
        
        # Should log a warning to stderr but still return a logger
        warning = capsys.readouterr().err
        assert "Warning: Could not setup log file" in warning
        assert logger.level == logging.DEBUG # Level should still be set correctly
        
    @patch('builtins.open', side_effect=IOError)
    def test_setup_logging_log_file_open_failure(self, mock_open, monkeypatch, capsys, tmp_path):
        """Test log file open failure (hitting the try/except block)."""
        log_file_path = tmp_path / "logs" / "test.log"
        monkeypatch.setenv("LOG_FILE", str(log_file_path))
        monkeypatch.setenv("LOG_LEVEL", "2")
        
        logger = setup_logging()
        
        # Should log a warning to stderr but still return a logger
        warning = capsys.readouterr().err
        assert "Warning: Could not setup log file" in warning
        assert logger.level == logging.DEBUG


# --- Test extract_model_name ---

class TestExtractModelName:
    """Tests for model name extraction edge cases."""

    def test_extract_model_name_with_tree(self):
        """Test extracting model name from a link pointing to a branch/tree."""
        link = "https://huggingface.co/microsoft/DialoGPT-medium/tree/main/config.json"
        result = extract_model_name(link)
        assert result == "DialoGPT-medium"

    def test_extract_model_name_with_params_and_slash(self):
        """Test extracting model name from link with params and trailing slash."""
        link = "https://huggingface.co/bert-base-uncased/?tab=readme"
        result = extract_model_name(link)
        # Note: Depending on main's regex, this might return 'bert-base-uncased' or 'readme'
        # Based on inspection of main.py, it should strip params first, then trailing slash.
        assert result == "bert-base-uncased"
    
    @pytest.mark.parametrize("invalid_input", [None, 123, []])
    def test_extract_model_name_invalid_input(self, invalid_input):
        """Test function robustness with invalid input types."""
        result = extract_model_name(invalid_input)
        assert result == "unknown_model"

    def test_extract_model_name_non_hf_url(self):
        """Test function on non-HuggingFace URL."""
        link = "https://www.google.com/model/test"
        result = extract_model_name(link)
        assert result == "unknown_model"


# --- Test parse_input and run_batch_evaluation failure paths ---

class TestEndToEndFailure:
    """Tests to cover final exception handling blocks."""

    def test_parse_input_generic_exception(self, monkeypatch, tmp_path):
        """Test parse_input hitting the generic catch-all Exception block."""
        
        # Create a file that will fail the inner logic (e.g., mock open failure)
        temp_path = tmp_path / "failing_input.txt"
        temp_path.write_text("https://github.com/a,https://hf.co/b,https://hf.co/model\n")
        
        # Mock the open function to raise a generic exception after the initial open succeeds
        def mock_open_failing(*args, **kwargs):
            raise Exception("Parsing failed unexpectedly")

        monkeypatch.setattr('builtins.open', mock_open_failing)
        
        # Because parse_input calls sys.exit(1) on failure, we need to mock it
        with pytest.raises(SystemExit):
            parse_input(str(temp_path))


    def test_run_batch_evaluation_no_jobs(self, monkeypatch, tmp_path, caplog):
        """Test run_batch_evaluation exits when no valid jobs are found."""
        
        # Create an empty file
        temp_file = tmp_path / "empty.txt"
        temp_file.write_text("   \n,,,\n") # Lines that result in no model_link
        
        # Mock sys.exit to catch the termination
        with pytest.raises(SystemExit):
            run_batch_evaluation(str(temp_file))
        
        # Check if the correct error was logged
        assert "No valid URLs found in input file" in caplog.text

    @patch('main.Controller')
    @patch('main.calculate_net_score', side_effect=Exception("Net score calculation crash"))
    def test_run_batch_evaluation_processing_exception(self, mock_net_score, mock_controller_class, monkeypatch, tmp_path, capsys, caplog):
        """Test the main processing loop's generic exception handler, ensuring it prints failure JSON."""
        
        # Mock file input to return one valid job
        input_file = tmp_path / "input.txt"
        input_file.write_text("https://huggingface.co/test/model-crash\n")
        
        # Mock controller fetch to succeed (so we get into the main loop)
        mock_controller_class.return_value.fetch.return_value = Mock()
        
        # Mock parallel runner to return valid, simple results
        mock_result = Mock(value=0.5)
        mock_results = {
            "License": (mock_result, 0.1),
            "Size": (mock_result, 0.1)
        }
        monkeypatch.setattr('main.run_evaluations_parallel', lambda *a, **k: mock_results)
        
        # The exception is injected by mock_net_score side_effect
        run_batch_evaluation(str(input_file))
        
        # Check if the fallback JSON was printed to stdout
        output = capsys.readouterr().out.strip()
        assert "model-crash" in output
        assert '"net_score": 0.0' in output
        
        # Check if the exception was logged to stderr/log
        assert "Net score calculation crash" in caplog.text