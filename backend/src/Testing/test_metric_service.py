"""
Unit tests for the Metric_Model_Service module.
Tests the evaluation logic for all 8 metrics under various conditions.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch

# Adjust path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after path adjustment
from Services.Metric_Model_Service import ModelMetricService  # noqa: E402
from lib.Metric_Result import MetricResult, MetricType  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


# --- Mock Data and Fixtures ---

# A comprehensive mock ModelManager object to pass to the service
@pytest.fixture
def mock_model_data():
    """Provides a mocked ModelManager instance with rich, complete data."""
    model_data = Mock()
    # Mock Model/HF Info
    model_data.id = "test/model"
    model_data.info = Mock(lastModified="2023-01-01T10:00:00Z")
    model_data.card = "Model README with Performance Claims, and a GPL-3.0 License."
    
    # Mock Dataset Info
    dataset_info = Mock()
    dataset_info.id = "test/dataset"
    dataset_info.card = "Dataset README with a high-quality description."
    model_data.dataset_ids = ["test/dataset"]
    model_data.dataset_infos = {"test/dataset": dataset_info}
    
    # Mock GitHub Repo Info
    model_data.repo_metadata = {
        'default_branch': 'main',
        'created_at': '2022-01-01T10:00:00Z',
        'updated_at': '2023-11-01T10:00:00Z',
        'stargazers_count': 500,
        'forks_count': 50,
        'size': 100000,  # 100MB
    }
    model_data.repo_contents = [
        {'name': 'src', 'type': 'dir'},
        {'name': 'setup.py', 'type': 'file'}
    ]
    model_data.repo_contributors = [
        {'login': 'contributor1', 'contributions': 100},
        {'login': 'contributor2', 'contributions': 50}
    ]
    model_data.code_link = "https://github.com/test/repo"
    return model_data

# --- Mock Dependencies (Internal Helpers) ---

@pytest.fixture
def mock_helpers():
    """Mocks internal helper functions used by the service."""
    with patch('Services.Metric_Model_Service._parse_iso8601') as mock_parse_iso, \
         patch('Services.Metric_Model_Service._months_between') as mock_months_between:
        
        # Default mock for date parsing to return a consistent mock datetime object
        mock_parse_iso.return_value = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        # Default mock for time calculation to return a large enough value
        mock_months_between.return_value = 12.0 
        
        yield mock_parse_iso, mock_months_between


# --- Main Test Class ---

@patch('Services.Metric_Model_Service.GithubAPIManager')
@patch('Services.Metric_Model_Service.HuggingFaceAPIManager')
@patch('Services.Metric_Model_Service.LLMManager')
class TestModelMetricService:
    """Tests for the ModelMetricService and its 8 metric evaluation methods."""

    def setup_method(self):
        """Setup before each test."""
        self.service = ModelMetricService()

    # --- Setup Mocks for LLM-based Metrics ---

    def _setup_llm_success(self, mock_llm_class, score):
        """Sets up the LLM mock to return a successful score string."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_response = Mock(content=str(score))
        mock_llm.call_genai_api.return_value = mock_response
        return mock_llm

    def _setup_llm_failure(self, mock_llm_class):
        """Sets up the LLM mock to raise a Runtime error."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.call_genai_api.side_effect = RuntimeError("LLM API failed")
        return mock_llm

    # --- Metric 1: Ramp-Up Time ---
    
    def test_evaluate_ramp_up_time_success(self, mock_llm, mock_hf, mock_gh, mock_model_data, mock_helpers):
        """Test successful Ramp-Up Time calculation (non-LLM metric)."""
        mock_parse_iso, mock_months_between = mock_helpers
        
        # Simulate a 12 month old repo. Ramp-Up score should be high.
        mock_months_between.return_value = 12.0
        
        result = self.service.EvaluateRampUpTime(mock_model_data)
        
        assert result.metric_type == MetricType.RAMP_UP_TIME
        assert result.value > 0.5  # Expect a decent score
        assert result.error is None
        mock_months_between.assert_called()

    def test_evaluate_ramp_up_time_no_repo(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Ramp-Up Time when GitHub repository is missing."""
        mock_model_data.repo_metadata = {}
        result = self.service.EvaluateRampUpTime(mock_model_data)
        
        assert result.value == 0.0
        assert "no code repository" in result.error

    # --- Metric 2: Bus Factor ---

    def test_evaluate_bus_factor_success(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test successful Bus Factor calculation (non-LLM metric)."""
        # Ensure contributors are mocked
        mock_model_data.repo_contributors = [
            {'login': 'c1', 'contributions': 800},
            {'login': 'c2', 'contributions': 100},
            {'login': 'c3', 'contributions': 10}
        ]
        
        result = self.service.EvaluateBusFactor(mock_model_data)
        
        assert result.metric_type == MetricType.BUS_FACTOR
        assert result.value > 0.0  # Should be based on contribution spread
        assert result.error is None

    def test_evaluate_bus_factor_no_contributors(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Bus Factor when contributors list is empty."""
        mock_model_data.repo_contributors = []
        result = self.service.EvaluateBusFactor(mock_model_data)
        
        assert result.value == 0.0
        assert "no contributors" in result.details.get("reason", "")
    
    def test_evaluate_bus_factor_no_repo(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Bus Factor when GitHub repository is missing."""
        mock_model_data.repo_metadata = {}
        result = self.service.EvaluateBusFactor(mock_model_data)
        
        assert result.value == 0.0
        assert "no code repository" in result.error

    # --- Metric 3: Performance Claims ---

    def test_evaluate_performance_claims_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test successful Performance Claims evaluation (LLM metric)."""
        score = 0.9
        self._setup_llm_success(mock_llm_class, score)
        
        result = self.service.EvaluatePerformanceClaims(mock_model_data)
        
        assert result.metric_type == MetricType.PERFORMANCE_CLAIMS
        assert result.value == score
        assert result.error is None
        mock_llm_class.return_value.call_genai_api.assert_called_once()

    def test_evaluate_performance_claims_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test Performance Claims handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        
        result = self.service.EvaluatePerformanceClaims(mock_model_data)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error

    # --- Metric 4: License ---

    def test_evaluate_license_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test successful License evaluation (LLM metric)."""
        score = 1.0
        self._setup_llm_success(mock_llm_class, score)
        
        result = self.service.EvaluateLicense(mock_model_data)
        
        assert result.metric_type == MetricType.LICENSE
        assert result.value == score
        assert result.error is None

    def test_evaluate_license_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test License handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        
        result = self.service.EvaluateLicense(mock_model_data)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error

    # --- Metric 5: Size ---

    def test_evaluate_size_success(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test successful Size evaluation (non-LLM metric)."""
        # Repo size is 100000 KB (100MB) from fixture
        result = self.service.EvaluateSize(mock_model_data)
        
        assert result.metric_type == MetricType.SIZE_SCORE
        # Expected value should be based on size tiers, 100MB should be a decent score
        assert result.value > 0.0
        assert result.error is None

    def test_evaluate_size_no_repo(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Size evaluation when GitHub repository is missing."""
        mock_model_data.repo_metadata = {}
        result = self.service.EvaluateSize(mock_model_data)
        
        assert result.value == 0.0
        assert "no code repository" in result.error

    # --- Metric 6: Dataset And Code Availability Score (Availability) ---

    def test_evaluate_availability_success(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test successful Availability evaluation (non-LLM metric)."""
        # Data has 1 dataset and a code link
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(mock_model_data)
        
        assert result.metric_type == MetricType.DATASET_AND_CODE_SCORE
        # Should be a high score (1.0 or near 1.0) since all links are present
        assert result.value >= 0.8
        assert result.error is None
        assert result.details["dataset_count"] == 1
        assert result.details["has_code_link"] is True

    def test_evaluate_availability_no_links(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Availability evaluation when both dataset and code links are missing."""
        mock_model_data.dataset_ids = []
        mock_model_data.code_link = None
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(mock_model_data)
        
        assert result.value == 0.0
        assert result.error is None
        assert result.details["has_code_link"] is False

    # --- Metric 7: Dataset Quality ---

    def test_evaluate_datasets_quality_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test successful Dataset Quality evaluation (LLM metric)."""
        score = 0.8
        self._setup_llm_success(mock_llm_class, score)
        
        result = self.service.EvaluateDatasetsQuality(mock_model_data)
        
        assert result.metric_type == MetricType.DATASET_QUALITY
        assert result.value == score
        assert result.error is None

    def test_evaluate_datasets_quality_no_datasets(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Dataset Quality when no datasets are linked."""
        mock_model_data.dataset_ids = []
        result = self.service.EvaluateDatasetsQuality(mock_model_data)
        
        assert result.value == 0.0
        assert "no datasets" in result.details.get("reason", "")

    def test_evaluate_datasets_quality_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test Dataset Quality handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        
        result = self.service.EvaluateDatasetsQuality(mock_model_data)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error

    # --- Metric 8: Code Quality ---

    def test_evaluate_code_quality_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test successful Code Quality evaluation (LLM metric)."""
        score = 0.75
        self._setup_llm_success(mock_llm_class, score)
        
        result = self.service.EvaluateCodeQuality(mock_model_data)
        
        assert result.metric_type == MetricType.CODE_QUALITY
        assert result.value == score
        assert result.error is None

    def test_evaluate_code_quality_no_repo(self, mock_llm, mock_hf, mock_gh, mock_model_data):
        """Test Code Quality when no code repository is linked."""
        mock_model_data.code_link = None
        mock_model_data.repo_metadata = {}
        result = self.service.EvaluateCodeQuality(mock_model_data)
        
        assert result.value == 0.0
        assert "no code repository" in result.details.get("reason", "")

    def test_evaluate_code_quality_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test Code Quality handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        
        result = self.service.EvaluateCodeQuality(mock_model_data)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error

    # --- Utility: LLM Score Parsing (for robustness) ---

    def test_llm_score_parsing_non_numeric_fallback(self, mock_llm_class, mock_hf, mock_gh, mock_model_data):
        """Test LLM-based metric handles non-numeric LLM output gracefully."""
        # Setup LLM to return text instead of a number
        mock_llm = self._setup_llm_success(mock_llm_class, "The score is maybe 0.7.")
        mock_llm.call_genai_api.return_value = Mock(content="The score is maybe 0.7.")
        
        # Test a representative LLM metric (e.g., License)
        result = self.service.EvaluateLicense(mock_model_data)
        
        assert result.value == 0.0 # Should fallback to 0.0 or handle exception internally
        assert "Could not parse" in result.error