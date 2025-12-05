"""
Unit tests for Controller module.
Tests data fetching and controller functionality.
"""
import os
import sys
from unittest.mock import Mock, patch
from huggingface_hub.utils._errors import RepositoryNotFoundError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after adding to path
from Controllers.Controller import Controller  # noqa: E402
from Models.Manager_Models_Model import ModelManager  # noqa: E402


# Mock necessary API dependencies for isolation
@patch('Models.Model.HuggingFaceAPIManager')
@patch('Models.Model.GitHubAPIManager')
class TestController:
    """Test cases for Controller class."""

    def setup_method(self):
        """Setup mock instances for each test."""
        self.controller = Controller()
        self.mock_hf_manager_class = TestController.mock_hf_api_class
        self.mock_github_manager_class = TestController.mock_github_api_class

        self.mock_hf_instance = self.mock_hf_manager_class.return_value
        self.mock_github_instance = self.mock_github_manager_class.return_value

        # Default successful setup for HF API dependencies
        self.mock_hf_instance.model_link_to_id.return_value = "gpt2"
        self.mock_hf_instance.get_model_info.return_value = Mock(id="gpt2", cardData="Mock card")
        self.mock_hf_instance.get_model_card.return_value = "Mock card"

        # Default successful setup for GitHub API dependencies
        self.mock_github_instance.code_link_to_repo.return_value = ("openai", "gpt-2")
        self.mock_github_instance.get_repo_info.return_value = {"name": "gpt-2", "owner": "openai"}
        self.mock_github_instance.get_repo_contents.return_value = ["file1"]
        self.mock_github_instance.get_repo_contributors.return_value = ["user1"]
        self.mock_github_instance.get_repo_readme.return_value = {"content": "Base64Readme"}


    def test_fetch_basic(self, *args):
        """Test basic fetch functionality with mocked model."""
        
        # Test 1: Short name without org prefix (mocked to succeed)
        model_link_short = "https://huggingface.co/gpt2"
        result_short = self.controller.fetch(model_link_short)
        
        assert isinstance(result_short, ModelManager)
        assert result_short.id == "gpt2"
        assert result_short.info is not None
        assert result_short.card == "Mock card"
        
        # Test 2: Full name with org prefix (mocked to succeed)
        self.mock_hf_instance.model_link_to_id.return_value = "google-bert/bert-base-uncased"
        model_link_full = "https://huggingface.co/google-bert/bert-base-uncased"
        result_full = self.controller.fetch(model_link_full)
        
        assert isinstance(result_full, ModelManager)
        assert result_full.id == "google-bert/bert-base-uncased"
        assert result_full.info is not None

    def test_fetch_model_only(self, *args):
        """Test fetch with model link only (no datasets or code)."""
        
        model_link = "https://huggingface.co/gpt2"
        result = self.controller.fetch(model_link)
        
        assert isinstance(result, ModelManager)
        assert result.id == "gpt2"
        assert result.dataset_ids == []
        assert result.repo_metadata == {}

    def test_fetch_with_invalid_model_link(self, *args):
        """Test fetch with invalid model link format."""
        
        # Mock HF manager to raise ValueError on link parsing
        self.mock_hf_instance.model_link_to_id.side_effect = ValueError("Invalid model link format")
        model_link = "https://huggingface.co/"
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid model link format"):
            self.controller.fetch(model_link)

    def test_fetch_with_model_api_failure(self, *args):
        """Test fetch when HuggingFace API returns a 404/RepositoryNotFoundError."""
        
        # Mock HF manager model fetch to fail
        self.mock_hf_instance.get_model_info.side_effect = RepositoryNotFoundError("Model not found")
        model_link = "https://huggingface.co/nonexistent/model"
        
        # Should propagate the API exception
        with pytest.raises(RepositoryNotFoundError):
            self.controller.fetch(model_link)


    def test_fetch_with_multiple_datasets(self, *args):
        """Test fetch with multiple dataset links."""
        
        # Setup mocks for datasets
        self.mock_hf_instance.dataset_link_to_id.side_effect = ["squad", "glue"]
        mock_ds_info_squad = Mock(id="squad")
        mock_ds_info_glue = Mock(id="glue")
        self.mock_hf_instance.get_dataset_info.side_effect = [mock_ds_info_squad, mock_ds_info_glue]

        model_link = "https://huggingface.co/gpt2"
        dataset_links = [
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/datasets/glue"
        ]
        
        result = self.controller.fetch(model_link, dataset_links)
        
        assert isinstance(result, ModelManager)
        assert len(result.dataset_ids) == 2
        assert "squad" in result.dataset_ids
        assert "glue" in result.dataset_ids
        assert len(result.dataset_infos) == 2
        
    def test_fetch_dataset_info_failure_is_ignored(self, *args):
        """Test fetch where dataset info retrieval fails but the overall fetch continues."""
        
        # Setup mocks for datasets
        self.mock_hf_instance.dataset_link_to_id.side_effect = ["squad", "failed_ds"]
        mock_ds_info_squad = Mock(id="squad")
        # Second dataset info fetch raises an exception
        self.mock_hf_instance.get_dataset_info.side_effect = [mock_ds_info_squad, Exception("DS fetch error")]

        model_link = "https://huggingface.co/gpt2"
        dataset_links = [
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/datasets/failed_ds"
        ]
        
        result = self.controller.fetch(model_link, dataset_links)
        
        assert isinstance(result, ModelManager)
        # Should have tried to fetch 2 IDs, but only successfully fetched info for 1
        assert len(result.dataset_ids) == 2
        assert len(result.dataset_infos) == 1
        assert "squad" in result.dataset_infos


    def test_fetch_with_github_repo(self, *args):
        """Test fetch with code link, ensuring all GitHub API calls were attempted."""
        
        model_link = "https://huggingface.co/gpt2"
        code_link = "https://github.com/openai/gpt-2"
        
        result = self.controller.fetch(model_link, code_link=code_link)
        
        assert isinstance(result, ModelManager)
        assert result.code_link == code_link
        
        # Verify GitHub API methods were called
        self.mock_github_instance.get_repo_info.assert_called_once()
        self.mock_github_instance.get_repo_contents.assert_called_once()
        self.mock_github_instance.get_repo_contributors.assert_called_once()
        self.mock_github_instance.get_repo_readme.assert_called_once()

        # Verify data fields were populated (using mock returns)
        assert result.repo_metadata == {"name": "gpt-2", "owner": "openai"}
        assert result.repo_contents == ["file1"]
        assert result.repo_contributors == ["user1"]


    def test_fetch_github_api_failure_is_ignored(self, *args):
        """Test fetch when GitHub API fails, ensuring graceful degradation."""
        
        # Mock GitHub API calls to fail
        self.mock_github_instance.get_repo_info.side_effect = Exception("GH info error")
        self.mock_github_instance.get_repo_contents.side_effect = Exception("GH contents error")
        self.mock_github_instance.get_repo_contributors.side_effect = Exception("GH contributors error")
        
        model_link = "https://huggingface.co/gpt2"
        code_link = "https://github.com/test/repo"
        
        result = self.controller.fetch(model_link, code_link=code_link)
        
        # Should still return a ModelManager instance with HF data
        assert isinstance(result, ModelManager)
        assert result.id == "gpt2"
        
        # Verify GitHub fields are empty due to failure
        assert result.repo_metadata == {}
        assert result.repo_contents == []
        assert result.repo_contributors == []
        
        # Ensure all methods were still *called* despite the side effect
        self.mock_github_instance.get_repo_info.assert_called_once()
        self.mock_github_instance.get_repo_contents.assert_called_once()
        self.mock_github_instance.get_repo_contributors.assert_called_once()

    def test_controller_initialization(self, *args):
        """Test controller initialization."""
        controller = Controller()
        assert controller is not None
        assert isinstance(controller.model_manager, ModelManager)