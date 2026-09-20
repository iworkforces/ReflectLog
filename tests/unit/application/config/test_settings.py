"""Tests for reflectlog.application.config.settings module."""

import threading
from typing import Any
from unittest.mock import patch

import pytest

from reflectlog.application.config.settings import (
    Config,
    _LazyConfig,
    _parse_optional_bool,
    get_config,
)
from reflectlog.application.utils.security import SecretString
from reflectlog.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Minimal env vars required for Config.from_environment()
# ---------------------------------------------------------------------------

REQUIRED_ENV = {
    "WORKSPACE_ID": "test-project",
    "OPENROUTER_API_KEY": "sk-test-key-12345",
}


# ---------------------------------------------------------------------------
# _parse_optional_bool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseOptionalBool:
    """Tests for the _parse_optional_bool helper."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("  true  ", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("  false  ", False),
            (None, None),
            ("", None),
            ("  ", None),
            ("maybe", None),
            ("invalid", None),
        ],
    )
    def test_parse_values(self, value: str | None, expected: bool | None):
        """Test all valid boolean strings, empty, and None inputs."""
        assert _parse_optional_bool(value) is expected


# ---------------------------------------------------------------------------
# Config defaults (frozen dataclass)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigDefaults:
    """Verify default field values on a directly constructed Config."""

    def _make_config(self, **overrides: Any) -> Config:
        defaults: dict[str, Any] = {
            "workspace_id": "proj",
            "openrouter_api_key": SecretString("key"),
        }
        defaults.update(overrides)
        return Config(**defaults)

    def test_transport_defaults(self):
        cfg = self._make_config()
        assert cfg.transport == "stdio"
        assert cfg.port == 9103
        assert cfg.host == "127.0.0.1"
        assert cfg.path == "/mcp"

    def test_embedding_defaults(self):
        cfg = self._make_config()
        assert cfg.embedder_provider == "openai"
        assert cfg.embedding_model == "openai/text-embedding-3-large"
        assert cfg.embedding_dims == 3072
        assert cfg.qwen_embedding_dims == 4096
        assert cfg.wemm_embedding_dims == 2048
        assert cfg.wemm_device == "auto"
        assert cfg.embedding_batch_size == 512
        assert cfg.embedding_max_concurrent_batches == 4
        assert cfg.embedding_cache_enabled is True
        assert cfg.embedding_cache_size == 100

    def test_search_defaults(self):
        cfg = self._make_config()
        assert cfg.search_limit == 5
        assert cfg.remove_search_limit == 5
        assert cfg.overfetch_multiplier == 3
        assert cfg.overfetch_adaptive is True
        assert cfg.overfetch_min_multiplier == 1.5
        assert cfg.overfetch_max_multiplier == 3.0

    def test_fusion_defaults(self):
        cfg = self._make_config()
        assert cfg.fusion_method == "rrf"
        assert cfg.fusion_normalization is None
        assert cfg.fusion_rrf_k == 60
        assert cfg.fusion_ranking_threshold == 0.0
        assert cfg.enable_rrf_fusion is True
        assert cfg.fusion_weights is None

    def test_reranker_defaults(self):
        cfg = self._make_config()
        assert cfg.reranker_engine == "cross_encoder"
        assert cfg.llm_model == "x-ai/grok-4.1-fast"
        assert cfg.search_score_threshold == 0.5
        assert cfg.rerank_max_concurrency == 10
        assert cfg.cross_encoder_model == "BAAI/bge-reranker-v2-m3"
        assert cfg.cross_encoder_top_k == 20
        assert cfg.cross_encoder_device == "cpu"

    def test_smart_replace_defaults(self):
        cfg = self._make_config()
        assert cfg.enable_smart_replace is True
        assert cfg.smart_replace_threshold == 0.7
        assert cfg.smart_replace_min_similarity == 0.9
        assert cfg.smart_replace_candidate_limit == 3
        assert cfg.smart_replace_archive_ttl_days == 30
        assert cfg.smart_replace_max_retries == 3
        assert cfg.smart_replace_retry_delay == 1.0

    def test_tantivy_defaults(self):
        cfg = self._make_config()
        assert cfg.tantivy_soft_delete_enabled is True
        assert cfg.tantivy_compaction_threshold_ratio == 0.2
        assert cfg.tantivy_compaction_max_tombstones == 10000
        assert cfg.tantivy_tombstone_ttl_days == 7
        assert cfg.tantivy_normalize_scores is True

    def test_usearch_defaults(self):
        cfg = self._make_config()
        assert cfg.usearch_exact_search is False
        assert cfg.usearch_exact_search_threshold == 256

    def test_memory_validation_defaults(self):
        cfg = self._make_config()
        assert cfg.max_memory_length == 30720
        assert cfg.min_memory_length == 1
        assert cfg.deduplicate_memories is True

    def test_init_defaults(self):
        cfg = self._make_config()
        assert cfg.add_max_concurrency == 4
        assert cfg.eager_initialization is True
        assert cfg.eager_initialize_search_engines is None
        assert cfg.eager_initialize_reranker is None
        assert cfg.eager_initialize_smart_replacer is None

    def test_logging_defaults(self):
        cfg = self._make_config()
        assert cfg.log_level == "INFO"
        assert cfg.log_search_results_verbose is False
        assert cfg.log_search_result_limit == 3

    def test_recency_defaults(self):
        cfg = self._make_config()
        assert cfg.enable_recency_boost is True
        assert cfg.recency_decay_rate == 0.001

    def test_allowed_tools_default(self):
        cfg = self._make_config()
        assert cfg.allowed_tools is None

    def test_frozen_raises_on_mutation(self):
        cfg = self._make_config()
        with pytest.raises(AttributeError):
            type(cfg).__setattr__(cfg, "port", 1234)


# ---------------------------------------------------------------------------
# Config.from_environment() – required vars & validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromEnvironmentRequired:
    """Config.from_environment() raises on missing/invalid required vars."""

    def test_missing_workspace_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("WORKSPACE_ID", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError, match="WORKSPACE_ID"):
            Config.from_environment()

    def test_project_id_env_is_ignored(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("WORKSPACE_ID", raising=False)
        monkeypatch.setenv("PROJECT_ID", "legacy-ws")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError, match="WORKSPACE_ID"):
            Config.from_environment()

    def test_empty_workspace_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError, match="WORKSPACE_ID"):
            Config.from_environment()

    def test_invalid_workspace_id_characters(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "bad/project!")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError, match="Invalid WORKSPACE_ID"):
            Config.from_environment()

    def test_workspace_id_too_long(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "a" * 65)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError, match="Invalid WORKSPACE_ID"):
            Config.from_environment()

    def test_workspace_id_path_traversal(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "a..b")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError, match="path traversal"):
            Config.from_environment()

    def test_missing_openrouter_api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "valid-project")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
            Config.from_environment()

    def test_empty_openrouter_api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "valid-project")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
            Config.from_environment()


# ---------------------------------------------------------------------------
# Config.from_environment() – successful creation with defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromEnvironmentDefaults:
    """Config.from_environment() returns correct defaults."""

    def test_creates_config_with_defaults(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        cfg = Config.from_environment()
        assert cfg.workspace_id == "test-project"
        assert isinstance(cfg.openrouter_api_key, SecretString)
        assert cfg.openrouter_api_key.get_secret_value() == "sk-test-key-12345"
        assert cfg.transport == "stdio"
        assert cfg.port == 9103
        assert cfg.embedding_dims == 3072

    def test_workspace_id_with_dots_and_underscores(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("WORKSPACE_ID", "my_project.v2")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        cfg = Config.from_environment()
        assert cfg.workspace_id == "my_project.v2"

    def test_workspace_id_keeps_env_case(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WORKSPACE_ID", "MyProject")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        cfg = Config.from_environment()
        assert cfg.workspace_id == "MyProject"


# ---------------------------------------------------------------------------
# Config.from_environment() – env var overrides
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromEnvironmentOverrides:
    """Config.from_environment() correctly reads env var overrides."""

    def _with_env(self, monkeypatch: pytest.MonkeyPatch, **extras: str) -> Config:
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        for k, v in extras.items():
            monkeypatch.setenv(k, v)
        return Config.from_environment()

    # --- Transport overrides ---

    def test_transport_http(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MCP_TRANSPORT="http")
        assert cfg.transport == "http"

    def test_transport_sse(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MCP_TRANSPORT="sse")
        assert cfg.transport == "sse"

    def test_transport_streamable_http(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MCP_TRANSPORT="streamable-http")
        assert cfg.transport == "streamable-http"

    def test_transport_unknown_raises(self, monkeypatch: pytest.MonkeyPatch):
        from reflectlog.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Invalid MCP_TRANSPORT"):
            _ = self._with_env(monkeypatch, MCP_TRANSPORT="unknown")

    def test_port_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, PORT="8080")
        assert cfg.port == 8080

    def test_mcp_port_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MCP_PORT="7777")
        assert cfg.port == 7777

    def test_port_takes_precedence_over_mcp_port(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, PORT="8080", MCP_PORT="7777")
        assert cfg.port == 8080

    def test_host_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MCP_HOST="0.0.0.0")
        assert cfg.host == "0.0.0.0"

    def test_path_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MCP_PATH="/custom")
        assert cfg.path == "/custom"

    def test_openrouter_base_url_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, OPENROUTER_BASE_URL="https://custom.api/v1")
        assert cfg.openrouter_base_url == "https://custom.api/v1"

    # --- Embedding overrides ---

    def test_embedder_provider_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, EMBEDDER_PROVIDER="langchain")
        assert cfg.embedder_provider == "langchain"

    def test_embedding_model_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, EMBEDDING_MODEL="custom/model")
        assert cfg.embedding_model == "custom/model"

    def test_embedding_dims_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, EMBEDDING_DIMS="1536")
        assert cfg.embedding_dims == 1536

    def test_qwen_embedding_dims_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, QWEN_EMBEDDING_DIMS="2048")
        assert cfg.qwen_embedding_dims == 2048

    def test_embedding_batch_size_clamped_to_min_1(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(ConfigurationError, match="EMBEDDING_BATCH_SIZE"):
            _ = self._with_env(monkeypatch, EMBEDDING_BATCH_SIZE="0")

    def test_embedding_max_concurrent_batches_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(
            ConfigurationError, match="EMBEDDING_MAX_CONCURRENT_BATCHES"
        ):
            _ = self._with_env(monkeypatch, EMBEDDING_MAX_CONCURRENT_BATCHES="0")

    def test_embedding_cache_disabled(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, EMBEDDING_CACHE_ENABLED="false")
        assert cfg.embedding_cache_enabled is False

    def test_embedding_cache_size_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="EMBEDDING_CACHE_SIZE"):
            _ = self._with_env(monkeypatch, EMBEDDING_CACHE_SIZE="0")

    # --- Search overrides ---

    def test_search_limit_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, SEARCH_LIMIT="10")
        assert cfg.search_limit == 10

    def test_remove_search_limit_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, REMOVE_SEARCH_LIMIT="20")
        assert cfg.remove_search_limit == 20

    def test_overfetch_multiplier_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="OVERFETCH_MULTIPLIER"):
            _ = self._with_env(monkeypatch, OVERFETCH_MULTIPLIER="0")

    def test_overfetch_adaptive_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, OVERFETCH_ADAPTIVE="false")
        assert cfg.overfetch_adaptive is False

    def test_overfetch_min_multiplier_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="OVERFETCH_MIN_MULTIPLIER"):
            _ = self._with_env(monkeypatch, OVERFETCH_MIN_MULTIPLIER="0.5")

    def test_overfetch_max_multiplier_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="OVERFETCH_MAX_MULTIPLIER"):
            _ = self._with_env(monkeypatch, OVERFETCH_MAX_MULTIPLIER="0.3")

    def test_usearch_exact_search_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, USEARCH_EXACT_SEARCH="false")
        assert cfg.usearch_exact_search is False

    def test_usearch_exact_search_threshold_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(ConfigurationError, match="USEARCH_EXACT_SEARCH_THRESHOLD"):
            _ = self._with_env(monkeypatch, USEARCH_EXACT_SEARCH_THRESHOLD="-1")

    # --- Fusion overrides ---

    def test_fusion_method_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_METHOD="SUM")
        assert cfg.fusion_method == "sum"

    def test_fusion_normalization_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_NORMALIZATION="min-max")
        assert cfg.fusion_normalization == "min-max"

    def test_fusion_normalization_empty_is_none(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_NORMALIZATION="")
        assert cfg.fusion_normalization is None

    def test_fusion_rrf_k_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_RRF_K="30")
        assert cfg.fusion_rrf_k == 30

    def test_fusion_ranking_threshold_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_RANKING_THRESHOLD="0.5")
        assert cfg.fusion_ranking_threshold == 0.5

    def test_enable_rrf_fusion_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, ENABLE_RRF_FUSION="false")
        assert cfg.enable_rrf_fusion is False

    def test_fusion_weights_parsed(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_WEIGHTS="0.6, 0.4")
        assert cfg.fusion_weights == [0.6, 0.4]

    def test_fusion_weights_empty_is_none(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, FUSION_WEIGHTS="")
        assert cfg.fusion_weights is None

    # --- Tantivy overrides ---

    def test_tantivy_soft_delete_disabled(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, TANTIVY_SOFT_DELETE_ENABLED="false")
        assert cfg.tantivy_soft_delete_enabled is False

    def test_tantivy_compaction_threshold_ratio_clamped_low(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(
            ConfigurationError, match="TANTIVY_COMPACTION_THRESHOLD_RATIO"
        ):
            _ = self._with_env(monkeypatch, TANTIVY_COMPACTION_THRESHOLD_RATIO="0.001")

    def test_tantivy_compaction_threshold_ratio_clamped_high(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(
            ConfigurationError, match="TANTIVY_COMPACTION_THRESHOLD_RATIO"
        ):
            _ = self._with_env(monkeypatch, TANTIVY_COMPACTION_THRESHOLD_RATIO="2.0")

    def test_tantivy_compaction_max_tombstones_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(
            ConfigurationError, match="TANTIVY_COMPACTION_MAX_TOMBSTONES"
        ):
            _ = self._with_env(monkeypatch, TANTIVY_COMPACTION_MAX_TOMBSTONES="50")

    def test_tantivy_tombstone_ttl_days_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="TANTIVY_TOMBSTONE_TTL_DAYS"):
            _ = self._with_env(monkeypatch, TANTIVY_TOMBSTONE_TTL_DAYS="-1")

    def test_tantivy_normalize_scores_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, TANTIVY_NORMALIZE_SCORES="false")
        assert cfg.tantivy_normalize_scores is False

    # --- Reranker overrides ---

    def test_reranker_engine_cross_encoder(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, RERANKER_ENGINE="cross_encoder")
        assert cfg.reranker_engine == "cross_encoder"

    def test_reranker_engine_none(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, RERANKER_ENGINE="none")
        assert cfg.reranker_engine == "none"

    def test_reranker_engine_invalid(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("RERANKER_ENGINE", "invalid_engine")
        with pytest.raises(ConfigurationError, match="Invalid RERANKER_ENGINE"):
            Config.from_environment()

    def test_reranker_engine_llm_rejected(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("RERANKER_ENGINE", "llm")
        with pytest.raises(ConfigurationError, match="Invalid RERANKER_ENGINE"):
            Config.from_environment()

    def test_llm_provider_openai(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, LLM_PROVIDER="openai")
        assert cfg.llm_provider == "openai"

    def test_llm_provider_invalid(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("LLM_PROVIDER", "google")
        with pytest.raises(ConfigurationError, match="Invalid LLM_PROVIDER"):
            Config.from_environment()

    def test_llm_model_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, LLM_MODEL="openai/gpt-4o")
        assert cfg.llm_model == "openai/gpt-4o"

    def test_search_score_threshold_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, SEARCH_SCORE_THRESHOLD="0.8")
        assert cfg.search_score_threshold == 0.8

    def test_rerank_max_concurrency_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="RERANK_MAX_CONCURRENCY"):
            _ = self._with_env(monkeypatch, RERANK_MAX_CONCURRENCY="0")

    def test_cross_encoder_settings(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(
            monkeypatch,
            CROSS_ENCODER_MODEL="custom/model",
            CROSS_ENCODER_TOP_K="10",
            CROSS_ENCODER_DEVICE="mps",
            CROSS_ENCODER_BATCH_SIZE="16",
            CROSS_ENCODER_SCORE_THRESHOLD="0.7",
            CROSS_ENCODER_USE_FP16="false",
            CROSS_ENCODER_NORMALIZE="false",
            CROSS_ENCODER_MAX_LENGTH="256",
        )
        assert cfg.cross_encoder_model == "custom/model"
        assert cfg.cross_encoder_top_k == 10
        assert cfg.cross_encoder_device == "mps"
        assert cfg.cross_encoder_batch_size == 16
        assert cfg.cross_encoder_score_threshold == 0.7
        assert cfg.cross_encoder_use_fp16 is False
        assert cfg.cross_encoder_normalize is False
        assert cfg.cross_encoder_max_length == 256

    def test_reranker_min_results_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="RERANKER_MIN_RESULTS"):
            _ = self._with_env(monkeypatch, RERANKER_MIN_RESULTS="-1")

    def test_reranker_batch_normalize_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, RERANKER_BATCH_NORMALIZE="false")
        assert cfg.reranker_batch_normalize is False

    def test_enable_recency_boost_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, ENABLE_RECENCY_BOOST="false")
        assert cfg.enable_recency_boost is False

    def test_recency_decay_rate_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="RECENCY_DECAY_RATE"):
            _ = self._with_env(monkeypatch, RECENCY_DECAY_RATE="-0.5")

    # --- Memory overrides ---

    def test_max_memory_length_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MAX_MEMORY_LENGTH="50000")
        assert cfg.max_memory_length == 50000

    def test_min_memory_length_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, MIN_MEMORY_LENGTH="10")
        assert cfg.min_memory_length == 10

    def test_deduplicate_memories_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, DEDUPLICATE_MEMORIES="false")
        assert cfg.deduplicate_memories is False

    # --- Smart replace overrides ---

    def test_enable_smart_replace_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, ENABLE_SMART_REPLACE="false")
        assert cfg.enable_smart_replace is False

    def test_smart_replace_threshold_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, SMART_REPLACE_THRESHOLD="0.5")
        assert cfg.smart_replace_threshold == 0.5

    def test_smart_replace_min_similarity_override(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = self._with_env(monkeypatch, SMART_REPLACE_MIN_SIMILARITY="0.8")
        assert cfg.smart_replace_min_similarity == 0.8

    def test_smart_replace_candidate_limit_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(ConfigurationError, match="SMART_REPLACE_CANDIDATE_LIMIT"):
            _ = self._with_env(monkeypatch, SMART_REPLACE_CANDIDATE_LIMIT="0")

    def test_smart_replace_archive_ttl_days_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(ConfigurationError, match="SMART_REPLACE_ARCHIVE_TTL_DAYS"):
            _ = self._with_env(monkeypatch, SMART_REPLACE_ARCHIVE_TTL_DAYS="-5")

    def test_smart_replace_max_retries_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="SMART_REPLACE_MAX_RETRIES"):
            _ = self._with_env(monkeypatch, SMART_REPLACE_MAX_RETRIES="0")

    def test_smart_replace_retry_delay_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="SMART_REPLACE_RETRY_DELAY"):
            _ = self._with_env(monkeypatch, SMART_REPLACE_RETRY_DELAY="0.01")

    # --- Init overrides ---

    def test_add_max_concurrency_clamped(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ConfigurationError, match="ADD_MAX_CONCURRENCY"):
            _ = self._with_env(monkeypatch, ADD_MAX_CONCURRENCY="0")

    def test_eager_initialization_false(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, EAGER_INITIALIZATION="false")
        assert cfg.eager_initialization is False

    def test_eager_initialize_search_engines_true(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = self._with_env(monkeypatch, EAGER_INITIALIZE_SEARCH_ENGINES="true")
        assert cfg.eager_initialize_search_engines is True

    def test_eager_initialize_search_engines_false(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = self._with_env(monkeypatch, EAGER_INITIALIZE_SEARCH_ENGINES="false")
        assert cfg.eager_initialize_search_engines is False

    def test_eager_initialize_reranker_true(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, EAGER_INITIALIZE_RERANKER="1")
        assert cfg.eager_initialize_reranker is True

    def test_eager_initialize_smart_replacer_true(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = self._with_env(monkeypatch, EAGER_INITIALIZE_SMART_REPLACER="yes")
        assert cfg.eager_initialize_smart_replacer is True

    # --- Logging overrides ---

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, LOG_LEVEL="DEBUG")
        assert cfg.log_level == "DEBUG"

    def test_log_search_results_verbose(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, LOG_SEARCH_RESULTS_VERBOSE="true")
        assert cfg.log_search_results_verbose is True

    def test_log_search_result_limit_override(self, monkeypatch: pytest.MonkeyPatch):
        cfg = self._with_env(monkeypatch, LOG_SEARCH_RESULT_LIMIT="10")
        assert cfg.log_search_result_limit == 10


# ---------------------------------------------------------------------------
# _parse_allowed_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseAllowedTools:
    """Tests for Config._parse_allowed_tools()."""

    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
        assert Config._parse_allowed_tools() is None

    def test_empty_string_returns_empty_tuple(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "")
        assert Config._parse_allowed_tools() == ()

    def test_blank_string_returns_empty_tuple(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "   ")
        assert Config._parse_allowed_tools() == ()

    def test_all_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "all")
        assert Config._parse_allowed_tools() is None

    def test_star_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "*")
        assert Config._parse_allowed_tools() is None

    def test_none_keyword_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "none")
        assert Config._parse_allowed_tools() == ()

    def test_disabled_keyword_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "disabled")
        assert Config._parse_allowed_tools() == ()

    def test_csv_tools_parsed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "add,search,remove")
        result = Config._parse_allowed_tools()
        assert result == ("add", "search", "remove")

    def test_normalizes_hyphens_to_underscores(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "health-check")
        result = Config._parse_allowed_tools()
        assert result == ("health_check",)

    def test_normalizes_camel_case(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "healthCheck")
        result = Config._parse_allowed_tools()
        assert result == ("health_check",)

    def test_normalizes_uppercase(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "ADD,SEARCH")
        result = Config._parse_allowed_tools()
        assert result == ("add", "search")

    def test_deduplicates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "add,search,add")
        result = Config._parse_allowed_tools()
        assert result == ("add", "search")

    def test_normalizes_whitespace(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "  add , search  ")
        result = Config._parse_allowed_tools()
        assert result == ("add", "search")

    def test_all_overrides_other_entries(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "add,all,search")
        assert Config._parse_allowed_tools() is None

    def test_spaces_in_tool_names_normalized(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ALLOWED_TOOLS", "get all")
        result = Config._parse_allowed_tools()
        assert result == ("get_all",)


# ---------------------------------------------------------------------------
# Singleton: get_config() and _LazyConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleton:
    """Tests for get_config() singleton and _LazyConfig proxy."""

    def test_get_config_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_get_config_raises_without_workspace_id(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("WORKSPACE_ID", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        with pytest.raises(ConfigurationError):
            get_config()

    def test_get_config_thread_safety(self, monkeypatch: pytest.MonkeyPatch):
        """Concurrent calls return the same instance."""
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)

        results: list[Config] = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(get_config())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4
        assert all(r is results[0] for r in results)

    def test_lazy_config_repr_not_initialized(self):
        proxy = _LazyConfig()
        assert "initialized=False" in repr(proxy)

    def test_lazy_config_repr_after_init(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        # Trigger initialization
        _ = get_config()
        proxy = _LazyConfig()
        assert "initialized=True" in repr(proxy)

    def test_lazy_config_forwards_attribute(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        proxy = _LazyConfig()
        assert proxy.workspace_id == "test-project"


# ---------------------------------------------------------------------------
# Preset integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPresetIntegration:
    """Config.from_environment() applies presets when REFLECTLOG_PROFILE is set."""

    def test_preset_applied(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("REFLECTLOG_PROFILE", "simple")
        cfg = Config.from_environment()
        assert cfg.reranker_engine == "none"

    def test_no_preset_uses_defaults(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("REFLECTLOG_PROFILE", raising=False)
        # Ensure no leftover preset-applied env vars from prior test
        for var in (
            "SEARCH_LIMIT",
            "RERANKER_ENGINE",
            "ENABLE_RECENCY_BOOST",
            "ENABLE_SMART_REPLACE",
            "EMBEDDING_BATCH_SIZE",
            "EMBEDDING_MAX_CONCURRENT_BATCHES",
            "USEARCH_EXACT_SEARCH",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = Config.from_environment()
        assert cfg.reranker_engine == "cross_encoder"


# ---------------------------------------------------------------------------
# Static parse methods (unit-level)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaticParseMethods:
    """Direct tests for static _parse_*_config methods with env overrides."""

    def test_parse_transport_config_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)
        monkeypatch.delenv("MCP_HOST", raising=False)
        monkeypatch.delenv("MCP_PATH", raising=False)
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        result = Config._parse_transport_config()
        assert result["transport"] == "stdio"
        assert result["port"] == 9103
        assert result["host"] == "127.0.0.1"
        assert result["path"] == "/mcp"

    def test_parse_embedding_config_custom(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EMBEDDER_PROVIDER", "langchain")
        monkeypatch.setenv("EMBEDDING_MODEL", "my-model")
        monkeypatch.setenv("EMBEDDING_DIMS", "768")
        result = Config._parse_embedding_config()
        assert result["embedder_provider"] == "langchain"
        assert result["embedding_model"] == "my-model"
        assert result["embedding_dims"] == 768

    def test_parse_search_config_fusion_weights(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FUSION_WEIGHTS", "0.7,0.3")
        result = Config._parse_search_config()
        assert result["fusion_weights"] == [0.7, 0.3]

    def test_parse_search_config_fusion_weights_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FUSION_WEIGHTS", "")
        result = Config._parse_search_config()
        assert result["fusion_weights"] is None

    def test_parse_tantivy_config_ratio_in_bounds(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("TANTIVY_COMPACTION_THRESHOLD_RATIO", "0.5")
        result = Config._parse_tantivy_config()
        assert result["tantivy_compaction_threshold_ratio"] == 0.5

    def test_parse_reranker_config_valid(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RERANKER_ENGINE", "cross_encoder")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        result = Config._parse_reranker_config()
        assert result["reranker_engine"] == "cross_encoder"
        assert result["llm_provider"] == "anthropic"

    def test_parse_reranker_config_invalid_engine_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("RERANKER_ENGINE", "bad")
        with pytest.raises(ConfigurationError, match="Invalid RERANKER_ENGINE"):
            Config._parse_reranker_config()

    def test_parse_reranker_config_invalid_llm_provider_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("RERANKER_ENGINE", "cross_encoder")
        monkeypatch.setenv("LLM_PROVIDER", "google")
        with pytest.raises(ConfigurationError, match="Invalid LLM_PROVIDER"):
            Config._parse_reranker_config()

    def test_parse_memory_config_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MAX_MEMORY_LENGTH", raising=False)
        monkeypatch.delenv("MIN_MEMORY_LENGTH", raising=False)
        monkeypatch.delenv("DEDUPLICATE_MEMORIES", raising=False)
        result = Config._parse_memory_config()
        assert result["max_memory_length"] == 30720
        assert result["min_memory_length"] == 1
        assert result["deduplicate_memories"] is True

    def test_parse_smart_replace_config_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ENABLE_SMART_REPLACE", raising=False)
        monkeypatch.delenv("SMART_REPLACE_THRESHOLD", raising=False)
        result = Config._parse_smart_replace_config()
        assert result["enable_smart_replace"] is True
        assert result["smart_replace_threshold"] == 0.7

    def test_parse_init_config_with_optional_bools(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("EAGER_INITIALIZE_SEARCH_ENGINES", "true")
        monkeypatch.setenv("EAGER_INITIALIZE_RERANKER", "false")
        monkeypatch.delenv("EAGER_INITIALIZE_SMART_REPLACER", raising=False)
        result = Config._parse_init_config()
        assert result["eager_initialize_search_engines"] is True
        assert result["eager_initialize_reranker"] is False
        assert result["eager_initialize_smart_replacer"] is None

    def test_parse_logging_config_custom(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_SEARCH_RESULTS_VERBOSE", "true")
        monkeypatch.setenv("LOG_SEARCH_RESULT_LIMIT", "20")
        result = Config._parse_logging_config()
        assert result["log_level"] == "DEBUG"
        assert result["log_search_results_verbose"] is True
        assert result["log_search_result_limit"] == 20


# ---------------------------------------------------------------------------
# setup_config_reload
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetupConfigReload:
    """Tests for setup_config_reload function."""

    def test_setup_config_reload_returns_config(self, monkeypatch: pytest.MonkeyPatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)

        with patch(
            "reflectlog.application.utils.config_reload.setup_signal_handler",
            return_value=None,
        ):
            from reflectlog.application.config.settings import setup_config_reload

            cfg = setup_config_reload()
            assert isinstance(cfg, Config)
            assert cfg.workspace_id == "test-project"


# ---------------------------------------------------------------------------
# TypedDict return types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypedDictReturns:
    """Ensure static methods return dicts with expected keys."""

    def test_transport_config_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        result = Config._parse_transport_config()
        expected_keys = {"transport", "port", "host", "path", "openrouter_base_url"}
        assert set(result.keys()) == expected_keys

    def test_embedding_config_keys(self):
        result = Config._parse_embedding_config()
        expected_keys = {
            "embedder_provider",
            "embedding_model",
            "embedding_dims",
            "qwen_embedding_dims",
            "wemm_embedding_dims",
            "wemm_device",
            "embedding_batch_size",
            "embedding_max_concurrent_batches",
            "embedding_cache_enabled",
            "embedding_cache_size",
        }
        assert set(result.keys()) == expected_keys

    def test_search_config_keys(self):
        result = Config._parse_search_config()
        expected_keys = {
            "search_limit",
            "remove_search_limit",
            "tantivy_index_path_template",
            "overfetch_multiplier",
            "overfetch_adaptive",
            "overfetch_min_multiplier",
            "overfetch_max_multiplier",
            "usearch_exact_search",
            "usearch_exact_search_threshold",
            "fusion_method",
            "fusion_normalization",
            "fusion_rrf_k",
            "fusion_ranking_threshold",
            "enable_rrf_fusion",
            "fusion_weights",
        }
        assert set(result.keys()) == expected_keys

    def test_reranker_config_keys(self):
        result = Config._parse_reranker_config()
        assert "reranker_engine" in result
        assert "llm_provider" in result
        assert "cross_encoder_model" in result

    def test_storage_config_keys(self):
        result = Config._parse_tantivy_config()
        expected_keys = {
            "tantivy_soft_delete_enabled",
            "tantivy_compaction_threshold_ratio",
            "tantivy_compaction_max_tombstones",
            "tantivy_tombstone_ttl_days",
            "tantivy_normalize_scores",
        }
        assert set(result.keys()) == expected_keys


@pytest.mark.unit
class TestMalformedNumericSettings:
    """Malformed numeric env values raise field-specific ConfigurationError."""

    def _load(self, monkeypatch: pytest.MonkeyPatch, **extras: str) -> Config:
        for key, value in REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)
        for key, value in extras.items():
            monkeypatch.setenv(key, value)
        return Config.from_environment()

    @pytest.mark.parametrize(
        ("env_name", "raw"),
        [
            ("MCP_PORT", "nan"),
            ("MCP_PORT", ""),
            ("MCP_PORT", "not-a-port"),
            ("EMBEDDING_BATCH_SIZE", ""),
            ("EMBEDDING_BATCH_SIZE", "abc"),
            ("ADD_MAX_CONCURRENCY", "-1"),
            ("SEARCH_LIMIT", "0"),
            ("OVERFETCH_MIN_MULTIPLIER", "inf"),
            ("RECENCY_DECAY_RATE", "nan"),
            ("SMART_REPLACE_RETRY_DELAY", "-0.5"),
        ],
    )
    def test_malformed_values_raise_named_error(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str, raw: str
    ) -> None:
        with pytest.raises(ConfigurationError, match=env_name) as exc_info:
            _ = self._load(monkeypatch, **{env_name: raw})
        message = str(exc_info.value)
        assert raw not in message or raw == ""
        assert "sk-test-key-12345" not in message

    def test_valid_boundary_values_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._load(
            monkeypatch,
            MCP_PORT="1",
            SEARCH_LIMIT="1",
            RECENCY_DECAY_RATE="0.0",
            TANTIVY_TOMBSTONE_TTL_DAYS="0",
        )
        assert cfg.port == 1
        assert cfg.search_limit == 1
        assert cfg.recency_decay_rate == 0.0
        assert cfg.tantivy_tombstone_ttl_days == 0

    def test_port_still_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._load(monkeypatch, PORT="8080", MCP_PORT="7777")
        assert cfg.port == 8080
