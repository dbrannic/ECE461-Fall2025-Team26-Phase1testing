"""
Unit tests for the Metric_Model_Service module.
Tests the evaluation logic for all 8 metrics under various conditions.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta

# Adjust path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after path adjustment
from Services.Metric_Model_Service import ModelMetricService  # noqa: E402
from lib.Metric_Result import MetricResult, MetricType  # noqa: E402
from Helpers.ISO_Parser import _parse_iso8601  # noqa: E402


# --- Mock Data and Fixtures ---

@pytest.fixture
def mock_model_data_full():
    """Provides a mocked ModelManager instance with rich, complete data."""
    model_data = Mock()
    # Mock Model/HF Info
    model_data.id = "test/model"
    # Set dates far apart for max Ramp-Up score
    model_data.info = Mock(lastModified=(datetime.now(timezone.utc) - timedelta(days=365*3)).isoformat())
    model_data.card = "Model README with Performance Claims, and a GPL-3.0 License."
    
    # Mock Dataset Info
    dataset_info = Mock()
    dataset_info.id = "test/dataset"
    dataset_info.card = "Dataset README with a high-quality description."
    model_data.dataset_ids = ["test/dataset"]
    model_data.dataset_infos = {"test/dataset": dataset_info}
    
    # Mock GitHub Repo Info (High activity, old repo)
    model_data.repo_metadata = {
        'default_branch': 'main',
        'created_at': (datetime.now(timezone.utc) - timedelta(days=365*3)).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'stargazers_count': 5000,
        'forks_count': 500,
        'size': 500000,  # 500MB
    }
    model_data.repo_contents = [
        {'name': 'src', 'type': 'dir'},
        {'name': 'setup.py', 'type': 'file'}
    ]
    # Diverse contributors for a good Bus Factor
    model_data.repo_contributors = [
        {'login': 'c1', 'contributions': 500},
        {'login': 'c2', 'contributions': 250},
        {'login': 'c3', 'contributions': 100},
        {'login': 'c4', 'contributions': 10}
    ]
    model_data.code_link = "https://github.com/test/repo"
    return model_data

@pytest.fixture
def mock_model_data_minimal():
    """Provides a mocked ModelManager instance with minimal/empty data."""
    model_data = Mock()
    model_data.id = "test/model"
    model_data.info = Mock(lastModified="2023-01-01T10:00:00Z")
    model_data.card = "Minimal model card."
    model_data.dataset_ids = []
    model_data.dataset_infos = {}
    model_data.repo_metadata = {}
    model_data.repo_contents = []
    model_data.repo_contributors = []
    model_data.code_link = None
    return model_data

# --- Mock Dependencies (Internal Helpers) ---

@pytest.fixture
def mock_time_helpers():
    """Mocks internal helper functions to control time differences."""
    with patch('Services.Metric_Model_Service._parse_iso8601') as mock_parse_iso, \
         patch('Services.Metric_Model_Service._months_between') as mock_months_between:
        
        # Default mock for time calculation to return a large enough value (1 year)
        mock_months_between.return_value = 12.0 
        
        # Mock date parsing to return a mock datetime object
        mock_parse_iso.return_value = datetime.now(timezone.utc)
        
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
    
    def _setup_llm_non_numeric(self, mock_llm_class):
        """Sets up the LLM mock to return non-numeric content."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.call_genai_api.return_value = Mock(content="This is not a number")
        return mock_llm

    # --- Metric 1: Ramp-Up Time (Success & Edge Cases) ---
    
    def test_evaluate_ramp_up_time_max_score(self, mock_llm, mock_hf, mock_gh, mock_model_data_full, mock_time_helpers):
        """Test Ramp-Up Time when repo is very old (should be max score)."""
        _, mock_months_between = mock_time_helpers
        mock_months_between.return_value = 48.0 # 4 years, max score range
        
        result = self.service.EvaluateRampUpTime(mock_model_data_full)
        
        assert result.metric_type == MetricType.RAMP_UP_TIME
        assert result.value > 0.95
        assert result.error is None

    def test_evaluate_ramp_up_time_min_score(self, mock_llm, mock_hf, mock_gh, mock_model_data_full, mock_time_helpers):
        """Test Ramp-Up Time when repo is very new (should be low score)."""
        _, mock_months_between = mock_time_helpers
        mock_months_between.return_value = 0.5 # Half a month
        
        result = self.service.EvaluateRampUpTime(mock_model_data_full)
        
        assert result.metric_type == MetricType.RAMP_UP_TIME
        assert result.value < 0.1
        assert result.error is None
        
    def test_evaluate_ramp_up_time_no_repo(self, mock_llm, mock_hf, mock_gh, mock_model_data_minimal):
        """Test Ramp-Up Time when GitHub repository is missing."""
        result = self.service.EvaluateRampUpTime(mock_model_data_minimal)
        
        assert result.value == 0.0
        assert "no code repository" in result.error

    # --- Metric 2: Bus Factor (Success & Edge Cases) ---

    def test_evaluate_bus_factor_multiple_contributors(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Bus Factor calculation with diverse contributors."""
        result = self.service.EvaluateBusFactor(mock_model_data_full)
        
        assert result.metric_type == MetricType.BUS_FACTOR
        assert result.value > 0.5 
        assert result.error is None

    def test_evaluate_bus_factor_single_contributor(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Bus Factor when one person does almost all the work (low score)."""
        mock_model_data_full.repo_contributors = [
            {'login': 'c1', 'contributions': 990},
            {'login': 'c2', 'contributions': 10},
        ]
        result = self.service.EvaluateBusFactor(mock_model_data_full)
        
        assert result.value < 0.2
        assert result.error is None
    
    def test_evaluate_bus_factor_no_repo_data(self, mock_llm, mock_hf, mock_gh, mock_model_data_minimal):
        """Test Bus Factor when GitHub repository is missing."""
        result = self.service.EvaluateBusFactor(mock_model_data_minimal)
        
        assert result.value == 0.0
        assert "no code repository" in result.error

    def test_evaluate_bus_factor_no_contributors(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Bus Factor when contributors list is empty."""
        mock_model_data_full.repo_contributors = []
        result = self.service.EvaluateBusFactor(mock_model_data_full)
        
        assert result.value == 0.0
        assert "no contributors" in result.details.get("reason", "")

    # --- Metric 3: Performance Claims (Success & Failure) ---

    @pytest.mark.parametrize("score", [1.0, 0.5, 0.0])
    def test_evaluate_performance_claims_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full, score):
        """Test successful Performance Claims evaluation (LLM metric)."""
        self._setup_llm_success(mock_llm_class, score)
        result = self.service.EvaluatePerformanceClaims(mock_model_data_full)
        
        assert result.metric_type == MetricType.PERFORMANCE_CLAIMS
        assert result.value == score
        assert result.error is None

    def test_evaluate_performance_claims_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Test Performance Claims handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        result = self.service.EvaluatePerformanceClaims(mock_model_data_full)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error
        
    def test_evaluate_performance_claims_non_numeric(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Test Performance Claims handles non-numeric LLM output."""
        self._setup_llm_non_numeric(mock_llm_class)
        result = self.service.EvaluatePerformanceClaims(mock_model_data_full)
        
        assert result.value == 0.0
        assert "Could not parse" in result.error

    # --- Metric 4: License (Success & Failure) ---

    @pytest.mark.parametrize("score", [1.0, 0.75, 0.25])
    def test_evaluate_license_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full, score):
        """Test successful License evaluation (LLM metric)."""
        self._setup_llm_success(mock_llm_class, score)
        result = self.service.EvaluateLicense(mock_model_data_full)
        
        assert result.metric_type == MetricType.LICENSE
        assert result.value == score
        assert result.error is None

    def test_evaluate_license_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Test License handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        result = self.service.EvaluateLicense(mock_model_data_full)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error

    # --- Metric 5: Size (Success & Edge Cases) ---

    def test_evaluate_size_high_score(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Size evaluation for a large model (500MB)."""
        # Repo size is 500000 KB (500MB) from fixture
        result = self.service.EvaluateSize(mock_model_data_full)
        
        assert result.metric_type == MetricType.SIZE_SCORE
        assert result.value > 0.0
        assert result.error is None

    def test_evaluate_size_zero_size(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Size evaluation when repo size is 0."""
        mock_model_data_full.repo_metadata = {'size': 0}
        result = self.service.EvaluateSize(mock_model_data_full)
        
        assert result.value == 0.0
        assert result.error is None

    def test_evaluate_size_no_repo_data(self, mock_llm, mock_hf, mock_gh, mock_model_data_minimal):
        """Test Size evaluation when GitHub repository is missing."""
        result = self.service.EvaluateSize(mock_model_data_minimal)
        
        assert result.value == 0.0
        assert "no code repository" in result.error

    # --- Metric 6: Availability (Success & Edge Cases) ---

    def test_evaluate_availability_full_links(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Availability when both dataset and code links are present."""
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(mock_model_data_full)
        
        assert result.metric_type == MetricType.DATASET_AND_CODE_SCORE
        # Should be maximum score
        assert result.value == 1.0
        assert result.error is None
        assert result.details["dataset_count"] == 1
        assert result.details["has_code_link"] is True

    def test_evaluate_availability_code_only(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Availability with code link only."""
        mock_model_data_full.dataset_ids = []
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(mock_model_data_full)
        
        assert result.value < 1.0 # Should be less than 1.0
        assert result.value > 0.0
        assert result.details["dataset_count"] == 0
        assert result.details["has_code_link"] is True

    def test_evaluate_availability_dataset_only(self, mock_llm, mock_hf, mock_gh, mock_model_data_full):
        """Test Availability with dataset link only."""
        mock_model_data_full.code_link = None
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(mock_model_data_full)
        
        assert result.value < 1.0 # Should be less than 1.0
        assert result.value > 0.0
        assert result.details["dataset_count"] == 1
        assert result.details["has_code_link"] is False

    def test_evaluate_availability_no_links(self, mock_llm, mock_hf, mock_gh, mock_model_data_minimal):
        """Test Availability evaluation when both dataset and code links are missing."""
        result = self.service.EvaluateDatasetAndCodeAvailabilityScore(mock_model_data_minimal)
        
        assert result.value == 0.0
        assert result.error is None
        assert result.details["has_code_link"] is False

    # --- Metric 7: Dataset Quality (Success & Failure) ---

    @pytest.mark.parametrize("score", [0.9, 0.4, 0.1])
    def test_evaluate_datasets_quality_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full, score):
        """Test successful Dataset Quality evaluation (LLM metric)."""
        self._setup_llm_success(mock_llm_class, score)
        
        result = self.service.EvaluateDatasetsQuality(mock_model_data_full)
        
        assert result.metric_type == MetricType.DATASET_QUALITY
        assert result.value == score
        assert result.error is None

    def test_evaluate_datasets_quality_no_datasets(self, mock_llm, mock_hf, mock_gh, mock_model_data_minimal):
        """Test Dataset Quality when no datasets are linked."""
        result = self.service.EvaluateDatasetsQuality(mock_model_data_minimal)
        
        assert result.value == 0.0
        assert "no datasets" in result.details.get("reason", "")
    
    def test_evaluate_datasets_quality_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Test Dataset Quality handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        result = self.service.EvaluateDatasetsQuality(mock_model_data_full)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error
        
    def test_evaluate_datasets_quality_non_numeric(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Test Dataset Quality handles non-numeric LLM output."""
        self._setup_llm_non_numeric(mock_llm_class)
        result = self.service.EvaluateDatasetsQuality(mock_model_data_full)
        
        assert result.value == 0.0
        assert "Could not parse" in result.error

    # --- Metric 8: Code Quality (Success & Failure) ---

    @pytest.mark.parametrize("score", [0.75, 0.5, 0.2])
    def test_evaluate_code_quality_success(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full, score):
        """Test successful Code Quality evaluation (LLM metric)."""
        self._setup_llm_success(mock_llm_class, score)
        
        result = self.service.EvaluateCodeQuality(mock_model_data_full)
        
        assert result.metric_type == MetricType.CODE_QUALITY
        assert result.value == score
        assert result.error is None

    def test_evaluate_code_quality_no_repo(self, mock_llm, mock_hf, mock_gh, mock_model_data_minimal):
        """Test Code Quality when no code repository is linked."""
        result = self.service.EvaluateCodeQuality(mock_model_data_minimal)
        
        assert result.value == 0.0
        assert "no code repository" in result.details.get("reason", "")

    def test_evaluate_code_quality_llm_failure(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Test Code Quality handling LLM API failure."""
        self._setup_llm_failure(mock_llm_class)
        result = self.service.EvaluateCodeQuality(mock_model_data_full)
        
        assert result.value == 0.0
        assert "LLM API failed" in result.error

    # --- Utility: Robustness Check ---
    
    def test_all_metrics_handle_llm_non_numeric_gracefully(self, mock_llm_class, mock_hf, mock_gh, mock_model_data_full):
        """Ensure all LLM-based metrics return 0.0 when LLM returns garbage string."""
        self._setup_llm_non_numeric(mock_llm_class)
        
        llm_metrics = [
            self.service.EvaluatePerformanceClaims,
            self.service.EvaluateLicense,
            self.service.EvaluateDatasetsQuality,
            self.service.EvaluateCodeQuality,
        ]
        
        for metric_func in llm_metrics:
            result = metric_func(mock_model_data_full)
            assert result.value == 0.0
            assert "Could not parse" in result.error