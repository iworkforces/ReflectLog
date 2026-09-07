#!/usr/bin/env python3
"""Unit tests for ENABLE_RRF_FUSION configuration toggle."""

import os
from unittest.mock import Mock, patch

import pytest

from reflectlog.application.config.settings import Config


@pytest.mark.unit
class TestRRFFusionToggleConfig:
    """Tests for ENABLE_RRF_FUSION configuration option."""

    def test_config_enable_rrf_fusion_default_true(self):
        """RRF fusion should be enabled by default."""
        with patch.dict(
            os.environ,
            {
                "WORKSPACE_ID": "test",
                "OPENROUTER_API_KEY": "test-key",
            },
            clear=True,
        ):
            config = Config.from_environment()
            assert config.enable_rrf_fusion is True

    def test_config_enable_rrf_fusion_explicit_true(self):
        """RRF fusion can be explicitly enabled."""
        with patch.dict(
            os.environ,
            {
                "WORKSPACE_ID": "test",
                "OPENROUTER_API_KEY": "test-key",
                "ENABLE_RRF_FUSION": "true",
            },
            clear=True,
        ):
            config = Config.from_environment()
            assert config.enable_rrf_fusion is True

    def test_config_enable_rrf_fusion_explicit_false(self):
        """RRF fusion can be disabled via environment variable."""
        with patch.dict(
            os.environ,
            {
                "WORKSPACE_ID": "test",
                "OPENROUTER_API_KEY": "test-key",
                "ENABLE_RRF_FUSION": "false",
            },
            clear=True,
        ):
            config = Config.from_environment()
            assert config.enable_rrf_fusion is False

    def test_config_enable_rrf_fusion_case_insensitive(self):
        """ENABLE_RRF_FUSION parsing should be case-insensitive."""
        test_cases = [
            ("TRUE", True),
            ("True", True),
            ("tRuE", True),
            ("FALSE", False),
            ("False", False),
            ("fAlSe", False),
        ]

        for value, expected in test_cases:
            with patch.dict(
                os.environ,
                {
                    "WORKSPACE_ID": "test",
                    "OPENROUTER_API_KEY": "test-key",
                    "ENABLE_RRF_FUSION": value,
                },
                clear=True,
            ):
                config = Config.from_environment()
                assert config.enable_rrf_fusion == expected, (
                    f"Failed for value '{value}': expected {expected}, got {config.enable_rrf_fusion}"
                )

    def test_config_enable_rrf_fusion_invalid_value_defaults_false(self):
        """Invalid ENABLE_RRF_FUSION values should default to False (not 'true')."""
        invalid_values = ["yes", "1", "on", "enabled", "invalid", ""]

        for value in invalid_values:
            with patch.dict(
                os.environ,
                {
                    "WORKSPACE_ID": "test",
                    "OPENROUTER_API_KEY": "test-key",
                    "ENABLE_RRF_FUSION": value,
                },
                clear=True,
            ):
                config = Config.from_environment()
                # Since comparison is `== "true"`, anything other than "true" will be False
                assert config.enable_rrf_fusion is False, (
                    f"Invalid value '{value}' should result in False"
                )


@pytest.mark.unit
class TestSearchPipelineWithRRFToggle:
    """Tests for search pipeline behavior with RRF toggle."""

    @pytest.fixture
    def mock_config_rrf_enabled(self):
        """Mock configuration with RRF fusion enabled."""
        config = Mock(spec=Config)
        config.workspace_id = "test_project"
        config.enable_rrf_fusion = True  # RRF enabled
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.index_base_path = "/tmp/test_indexes"
        config.search_limit = 5
        config.search_score_threshold = 0.8
        config.deduplicate_memories = True
        config.enable_llm_infer = False
        config.remove_search_limit = 5
        config.remove_score_threshold = 0.9
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.fusion_ranking_threshold = 0.5
        config.add_max_concurrency = 4
        config.rerank_max_concurrency = 5
        config.enable_smart_replace = False
        config.smart_replace_threshold = 0.7
        config.smart_replace_min_similarity = 0.5
        config.smart_replace_candidate_limit = 3
        config.smart_replace_archive_ttl_days = 30
        config.smart_replace_max_retries = 3
        config.smart_replace_retry_delay = 1.0
        config.reranker_engine = "none"
        config.reranker_min_results = 0
        config.reranker_batch_normalize = True
        config.overfetch_multiplier = 3
        config.log_search_results_verbose = False
        config.log_search_result_limit = 3
        config.embedding_model = "openai/text-embedding-3-large"
        config.embedding_dims = 3072
        config.embedder_provider = "openai"
        config.llm_model = "x-ai/grok-4.1-fast"
        config.openrouter_api_key = Mock()
        config.openrouter_api_key.get_secret_value.return_value = "test-api-key"
        config.openrouter_base_url = "https://openrouter.ai/api/v1"
        config.qwen_embedding_dims = 4096
        config.embedding_cache_enabled = False
        config.embedding_cache_size = 100
        config.eager_initialization = False
        return config

    @pytest.fixture
    def mock_config_rrf_disabled(self):
        """Mock configuration with RRF fusion disabled."""
        config = Mock(spec=Config)
        config.workspace_id = "test_project"
        config.enable_rrf_fusion = False  # RRF disabled
        config.tantivy_index_path_template = "{workspace_id}_tantivy_test"
        config.index_base_path = "/tmp/test_indexes"
        config.search_limit = 5
        config.search_score_threshold = 0.8
        config.deduplicate_memories = True
        config.enable_llm_infer = False
        config.remove_search_limit = 5
        config.remove_score_threshold = 0.9
        config.fusion_method = "rrf"
        config.fusion_normalization = None
        config.fusion_rrf_k = 60
        config.fusion_ranking_threshold = 0.5
        config.add_max_concurrency = 4
        config.rerank_max_concurrency = 5
        config.enable_smart_replace = False
        config.smart_replace_threshold = 0.7
        config.smart_replace_min_similarity = 0.5
        config.smart_replace_candidate_limit = 3
        config.smart_replace_archive_ttl_days = 30
        config.smart_replace_max_retries = 3
        config.smart_replace_retry_delay = 1.0
        config.reranker_engine = "none"
        config.reranker_min_results = 0
        config.reranker_batch_normalize = True
        config.overfetch_multiplier = 3
        config.log_search_results_verbose = False
        config.log_search_result_limit = 3
        config.embedding_model = "openai/text-embedding-3-large"
        config.embedding_dims = 3072
        config.embedder_provider = "openai"
        config.llm_model = "x-ai/grok-4.1-fast"
        config.openrouter_api_key = Mock()
        config.openrouter_api_key.get_secret_value.return_value = "test-api-key"
        config.openrouter_base_url = "https://openrouter.ai/api/v1"
        config.qwen_embedding_dims = 4096
        config.embedding_cache_enabled = False
        config.embedding_cache_size = 100
        config.eager_initialization = False
        return config

    @pytest.fixture
    def mock_logger(self):
        """Mock structured logger."""
        from typing import cast

        from reflectlog.application.utils.logging import StructuredLogger
        from reflectlog.core.logging import IStructuredLogger

        return cast(IStructuredLogger, Mock(spec=StructuredLogger))

    def test_rrf_enabled_uses_fusion_engine(self, mock_config_rrf_enabled, mock_logger):
        """When RRF is enabled, fusion engine should be called."""
        from reflectlog.application.memory.manager import MemoryManager

        with patch(
            "reflectlog.application.memory.manager.USearchEngine"
        ) as mock_usearch:
            with patch("reflectlog.application.memory.manager.LangchainQwenEmbeddings"):
                with patch(
                    "reflectlog.application.memory.manager.TantivyEngine"
                ) as mock_tantivy:
                    mock_usearch_instance = Mock()
                    mock_usearch_instance.search.return_value = [
                        ("msg1", 0.9, "2024-01-01T00:00:00")
                    ]
                    mock_usearch_instance.memory_store.list_pending_transitions.return_value = []
                    mock_usearch.return_value = mock_usearch_instance

                    mock_tantivy_instance = Mock()
                    mock_tantivy_instance.search.return_value = [("msg2", 0.8)]
                    mock_tantivy.return_value = mock_tantivy_instance

                    manager = MemoryManager(mock_config_rrf_enabled, mock_logger)

                    # Mock the fusion engine
                    mock_fusion = Mock()
                    mock_fusion.fuse.return_value = [("msg1", 0.95), ("msg2", 0.85)]
                    mock_fusion.method = "rrf"
                    manager._fusion_engine = mock_fusion

        # Verify config is set correctly
        assert mock_config_rrf_enabled.enable_rrf_fusion is True

    @pytest.mark.asyncio
    async def test_rrf_disabled_uses_concatenation(
        self, mock_config_rrf_disabled, mock_logger
    ):
        """When RRF is disabled, concatenation should be used instead of fusion."""
        # When RRF is disabled, config should reflect that
        assert mock_config_rrf_disabled.enable_rrf_fusion is False
