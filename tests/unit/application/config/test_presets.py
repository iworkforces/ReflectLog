"""Tests for reflectlog.application.config.presets module."""

import os

import pytest
from pytest import MonkeyPatch

from reflectlog.application.config.presets import (
    BALANCED_PRESET,
    PERFORMANCE_PRESET,
    PRESETS,
    QUALITY_PRESET,
    SIMPLE_PRESET,
    ConfigPreset,
    apply_preset_to_env,
    get_active_preset,
    get_preset_summary,
)
from reflectlog.core.enums import RerankerEngine

# ---------------------------------------------------------------------------
# ConfigPreset dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfigPreset:
    """Tests for the ConfigPreset dataclass."""

    def test_all_fields_default_to_none(self):
        """All optional fields should default to None."""
        preset = ConfigPreset(name="test")
        assert preset.search_limit is None
        assert preset.search_score_threshold is None
        assert preset.fusion_rrf_k is None
        assert preset.overfetch_multiplier is None
        assert preset.reranker_engine is None
        assert preset.enable_recency_boost is None
        assert preset.recency_decay_rate is None
        assert preset.enable_smart_replace is None
        assert preset.smart_replace_threshold is None
        assert preset.embedding_batch_size is None
        assert preset.embedding_max_concurrent_batches is None
        assert preset.enable_embedding_cache is None
        assert preset.embedding_cache_size is None
        assert preset.usearch_exact_search is None

    def test_fields_can_be_set(self):
        """All fields should accept explicit values."""
        preset = ConfigPreset(
            name="custom",
            search_limit=10,
            search_score_threshold=0.5,
            fusion_rrf_k=60,
            overfetch_multiplier=3,
            reranker_engine=RerankerEngine.CROSS_ENCODER,
            enable_recency_boost=True,
            recency_decay_rate=0.02,
            enable_smart_replace=True,
            smart_replace_threshold=0.8,
            embedding_batch_size=512,
            embedding_max_concurrent_batches=4,
            enable_embedding_cache=True,
            embedding_cache_size=100,
            usearch_exact_search=False,
        )
        assert preset.name == "custom"
        assert preset.search_limit == 10
        assert preset.search_score_threshold == 0.5
        assert preset.fusion_rrf_k == 60
        assert preset.overfetch_multiplier == 3
        assert preset.reranker_engine == "cross_encoder"
        assert preset.enable_recency_boost is True
        assert preset.recency_decay_rate == 0.02
        assert preset.enable_smart_replace is True
        assert preset.smart_replace_threshold == 0.8
        assert preset.embedding_batch_size == 512
        assert preset.embedding_max_concurrent_batches == 4
        assert preset.enable_embedding_cache is True
        assert preset.embedding_cache_size == 100
        assert preset.usearch_exact_search is False


# ---------------------------------------------------------------------------
# Preset constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPresetConstants:
    """Tests for pre-defined preset instances."""

    def test_simple_preset_values(self):
        """SIMPLE_PRESET should have low-resource settings."""
        assert SIMPLE_PRESET.name == "simple"
        assert SIMPLE_PRESET.search_limit == 3
        assert SIMPLE_PRESET.reranker_engine == "none"
        assert SIMPLE_PRESET.enable_recency_boost is False
        assert SIMPLE_PRESET.enable_smart_replace is False
        assert SIMPLE_PRESET.embedding_batch_size == 128
        assert SIMPLE_PRESET.embedding_max_concurrent_batches == 2
        assert SIMPLE_PRESET.enable_embedding_cache is True
        assert SIMPLE_PRESET.embedding_cache_size == 50
        assert SIMPLE_PRESET.usearch_exact_search is True

    def test_balanced_preset_all_none(self):
        """BALANCED_PRESET should leave all optional fields as None."""
        assert BALANCED_PRESET.name == "balanced"
        assert BALANCED_PRESET.search_limit is None
        assert BALANCED_PRESET.search_score_threshold is None
        assert BALANCED_PRESET.fusion_rrf_k is None
        assert BALANCED_PRESET.overfetch_multiplier is None
        assert BALANCED_PRESET.reranker_engine is None
        assert BALANCED_PRESET.enable_recency_boost is None
        assert BALANCED_PRESET.recency_decay_rate is None
        assert BALANCED_PRESET.enable_smart_replace is None
        assert BALANCED_PRESET.smart_replace_threshold is None
        assert BALANCED_PRESET.embedding_batch_size is None
        assert BALANCED_PRESET.embedding_max_concurrent_batches is None
        assert BALANCED_PRESET.enable_embedding_cache is None
        assert BALANCED_PRESET.embedding_cache_size is None
        assert BALANCED_PRESET.usearch_exact_search is None

    def test_performance_preset_values(self):
        """PERFORMANCE_PRESET should favor speed."""
        assert PERFORMANCE_PRESET.name == "performance"
        assert PERFORMANCE_PRESET.search_limit == 10
        assert PERFORMANCE_PRESET.search_score_threshold == 0.3
        assert PERFORMANCE_PRESET.fusion_rrf_k == 40
        assert PERFORMANCE_PRESET.overfetch_multiplier == 2
        assert PERFORMANCE_PRESET.reranker_engine == "none"
        assert PERFORMANCE_PRESET.enable_recency_boost is False
        assert PERFORMANCE_PRESET.enable_smart_replace is False
        assert PERFORMANCE_PRESET.embedding_batch_size == 1024
        assert PERFORMANCE_PRESET.embedding_max_concurrent_batches == 8
        assert PERFORMANCE_PRESET.enable_embedding_cache is True
        assert PERFORMANCE_PRESET.embedding_cache_size == 200
        assert PERFORMANCE_PRESET.usearch_exact_search is False

    def test_quality_preset_values(self):
        """QUALITY_PRESET should favor accuracy."""
        assert QUALITY_PRESET.name == "quality"
        assert QUALITY_PRESET.search_limit == 5
        assert QUALITY_PRESET.search_score_threshold == 0.7
        assert QUALITY_PRESET.fusion_rrf_k == 80
        assert QUALITY_PRESET.overfetch_multiplier == 5
        assert QUALITY_PRESET.reranker_engine == "cross_encoder"
        assert QUALITY_PRESET.enable_recency_boost is True
        assert QUALITY_PRESET.enable_smart_replace is True
        assert QUALITY_PRESET.smart_replace_threshold == 0.9
        assert QUALITY_PRESET.embedding_batch_size == 256
        assert QUALITY_PRESET.embedding_max_concurrent_batches == 2
        assert QUALITY_PRESET.enable_embedding_cache is True
        assert QUALITY_PRESET.embedding_cache_size == 200
        assert QUALITY_PRESET.usearch_exact_search is True

    def test_presets_dict_has_all_four(self):
        """PRESETS dict should contain all four presets keyed by name."""
        assert len(PRESETS) == 4
        assert set(PRESETS.keys()) == {"simple", "balanced", "performance", "quality"}
        assert PRESETS["simple"] is SIMPLE_PRESET
        assert PRESETS["balanced"] is BALANCED_PRESET
        assert PRESETS["performance"] is PERFORMANCE_PRESET
        assert PRESETS["quality"] is QUALITY_PRESET


# ---------------------------------------------------------------------------
# get_active_preset()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetActivePreset:
    """Tests for the get_active_preset function."""

    def test_returns_none_when_env_not_set(self, monkeypatch: MonkeyPatch) -> None:
        """Should return None when REFLECTLOG_PROFILE is not set."""
        monkeypatch.delenv("REFLECTLOG_PROFILE", raising=False)
        assert get_active_preset() is None

    def test_returns_none_for_empty_string(self, monkeypatch: MonkeyPatch) -> None:
        """Should return None when REFLECTLOG_PROFILE is empty."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", "")
        assert get_active_preset() is None

    def test_returns_none_for_custom(self, monkeypatch: MonkeyPatch) -> None:
        """Should return None when REFLECTLOG_PROFILE is "custom"."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", "custom")
        assert get_active_preset() is None

    def test_returns_none_for_custom_uppercase(self, monkeypatch: MonkeyPatch) -> None:
        """Should return None when REFLECTLOG_PROFILE is "CUSTOM" (case-insensitive)."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", "CUSTOM")
        assert get_active_preset() is None

    @pytest.mark.parametrize(
        "profile_name", ["simple", "balanced", "performance", "quality"]
    )
    def test_returns_correct_preset(
        self, monkeypatch: MonkeyPatch, profile_name: str
    ) -> None:
        """Should return matching preset for valid profile names."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", profile_name)
        result = get_active_preset()
        assert result is not None
        assert result.name == profile_name

    @pytest.mark.parametrize(
        "profile_name", ["SIMPLE", "Balanced", "PERFORMANCE", "Quality"]
    )
    def test_returns_preset_case_insensitive(
        self, monkeypatch: MonkeyPatch, profile_name: str
    ) -> None:
        """Should handle case-insensitive profile names."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", profile_name)
        result = get_active_preset()
        assert result is not None
        assert result.name == profile_name.lower()

    def test_returns_none_for_unknown_profile(self, monkeypatch: MonkeyPatch) -> None:
        """Should return None for unrecognized profile names."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", "turbo")
        assert get_active_preset() is None


# ---------------------------------------------------------------------------
# apply_preset_to_env()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyPresetToEnv:
    """Tests for the apply_preset_to_env function."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: MonkeyPatch) -> None:
        """Remove preset-related env vars before each test."""
        env_keys = [
            "SEARCH_LIMIT",
            "SEARCH_SCORE_THRESHOLD",
            "FUSION_RRF_K",
            "OVERFETCH_MULTIPLIER",
            "RERANKER_ENGINE",
            "ENABLE_RECENCY_BOOST",
            "RECENCY_DECAY_RATE",
            "ENABLE_SMART_REPLACE",
            "SMART_REPLACE_THRESHOLD",
            "EMBEDDING_BATCH_SIZE",
            "EMBEDDING_MAX_CONCURRENT_BATCHES",
            "EMBEDDING_CACHE_ENABLED",
            "EMBEDDING_CACHE_SIZE",
            "USEARCH_EXACT_SEARCH",
        ]
        for key in env_keys:
            monkeypatch.delenv(key, raising=False)

    def test_simple_preset_sets_expected_env_vars(self):
        """apply_preset_to_env with SIMPLE_PRESET should set all non-None fields."""
        apply_preset_to_env(SIMPLE_PRESET)

        assert os.environ["SEARCH_LIMIT"] == "3"
        assert os.environ["RERANKER_ENGINE"] == "none"
        assert os.environ["ENABLE_RECENCY_BOOST"] == "false"
        assert os.environ["ENABLE_SMART_REPLACE"] == "false"
        assert os.environ["EMBEDDING_BATCH_SIZE"] == "128"
        assert os.environ["EMBEDDING_MAX_CONCURRENT_BATCHES"] == "2"
        assert os.environ["EMBEDDING_CACHE_ENABLED"] == "true"
        assert os.environ["EMBEDDING_CACHE_SIZE"] == "50"
        assert os.environ["USEARCH_EXACT_SEARCH"] == "true"

    def test_simple_preset_does_not_set_none_fields(self):
        """Fields that are None on the preset should not be set."""
        apply_preset_to_env(SIMPLE_PRESET)

        # These are None on SIMPLE_PRESET
        assert "SEARCH_SCORE_THRESHOLD" not in os.environ
        assert "FUSION_RRF_K" not in os.environ
        assert "OVERFETCH_MULTIPLIER" not in os.environ
        assert "RECENCY_DECAY_RATE" not in os.environ
        assert "SMART_REPLACE_THRESHOLD" not in os.environ

    def test_balanced_preset_sets_nothing(self):
        apply_preset_to_env(BALANCED_PRESET)

        assert "SEARCH_LIMIT" not in os.environ
        assert "SEARCH_SCORE_THRESHOLD" not in os.environ
        assert "FUSION_RRF_K" not in os.environ
        assert "OVERFETCH_MULTIPLIER" not in os.environ
        assert "RERANKER_ENGINE" not in os.environ
        assert "ENABLE_RECENCY_BOOST" not in os.environ
        assert "RECENCY_DECAY_RATE" not in os.environ
        assert "ENABLE_SMART_REPLACE" not in os.environ
        assert "SMART_REPLACE_THRESHOLD" not in os.environ
        assert "EMBEDDING_BATCH_SIZE" not in os.environ
        assert "EMBEDDING_MAX_CONCURRENT_BATCHES" not in os.environ
        assert "EMBEDDING_CACHE_ENABLED" not in os.environ
        assert "EMBEDDING_CACHE_SIZE" not in os.environ
        assert "USEARCH_EXACT_SEARCH" not in os.environ

    def test_performance_preset_sets_all_fields(self):
        """PERFORMANCE_PRESET has all fields set."""
        apply_preset_to_env(PERFORMANCE_PRESET)

        assert os.environ["SEARCH_LIMIT"] == "10"
        assert os.environ["SEARCH_SCORE_THRESHOLD"] == "0.3"
        assert os.environ["FUSION_RRF_K"] == "40"
        assert os.environ["OVERFETCH_MULTIPLIER"] == "2"
        assert os.environ["RERANKER_ENGINE"] == "none"
        assert os.environ["ENABLE_RECENCY_BOOST"] == "false"
        assert os.environ["ENABLE_SMART_REPLACE"] == "false"
        assert os.environ["EMBEDDING_BATCH_SIZE"] == "1024"
        assert os.environ["EMBEDDING_MAX_CONCURRENT_BATCHES"] == "8"
        assert os.environ["EMBEDDING_CACHE_ENABLED"] == "true"
        assert os.environ["EMBEDDING_CACHE_SIZE"] == "200"
        assert os.environ["USEARCH_EXACT_SEARCH"] == "false"

    def test_quality_preset_sets_all_fields(self):
        """QUALITY_PRESET has all fields set including smart replace threshold."""
        apply_preset_to_env(QUALITY_PRESET)

        assert os.environ["SEARCH_LIMIT"] == "5"
        assert os.environ["SEARCH_SCORE_THRESHOLD"] == "0.7"
        assert os.environ["FUSION_RRF_K"] == "80"
        assert os.environ["OVERFETCH_MULTIPLIER"] == "5"
        assert os.environ["RERANKER_ENGINE"] == "cross_encoder"
        assert os.environ["ENABLE_RECENCY_BOOST"] == "true"
        # recency_decay_rate is None on QUALITY_PRESET, so not set
        assert "RECENCY_DECAY_RATE" not in os.environ
        assert os.environ["ENABLE_SMART_REPLACE"] == "true"
        assert os.environ["SMART_REPLACE_THRESHOLD"] == "0.9"
        assert os.environ["EMBEDDING_BATCH_SIZE"] == "256"
        assert os.environ["EMBEDDING_MAX_CONCURRENT_BATCHES"] == "2"
        assert os.environ["EMBEDDING_CACHE_ENABLED"] == "true"
        assert os.environ["EMBEDDING_CACHE_SIZE"] == "200"
        assert os.environ["USEARCH_EXACT_SEARCH"] == "true"

    def test_custom_preset_with_recency_decay_rate(self):
        """Verify recency_decay_rate branch is covered."""
        preset = ConfigPreset(
            name="custom_decay",
            recency_decay_rate=0.05,
        )
        apply_preset_to_env(preset)
        assert os.environ["RECENCY_DECAY_RATE"] == "0.05"

    def test_preset_with_all_none_sets_nothing(self):
        """A preset with all None fields should not set any env vars."""
        preset = ConfigPreset(name="empty")
        apply_preset_to_env(preset)

        assert "SEARCH_LIMIT" not in os.environ
        assert "SEARCH_SCORE_THRESHOLD" not in os.environ
        assert "FUSION_RRF_K" not in os.environ
        assert "OVERFETCH_MULTIPLIER" not in os.environ
        assert "RERANKER_ENGINE" not in os.environ
        assert "ENABLE_RECENCY_BOOST" not in os.environ
        assert "RECENCY_DECAY_RATE" not in os.environ
        assert "ENABLE_SMART_REPLACE" not in os.environ
        assert "SMART_REPLACE_THRESHOLD" not in os.environ
        assert "EMBEDDING_BATCH_SIZE" not in os.environ
        assert "EMBEDDING_MAX_CONCURRENT_BATCHES" not in os.environ
        assert "EMBEDDING_CACHE_ENABLED" not in os.environ
        assert "EMBEDDING_CACHE_SIZE" not in os.environ
        assert "USEARCH_EXACT_SEARCH" not in os.environ


# ---------------------------------------------------------------------------
# get_preset_summary()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetPresetSummary:
    """Tests for the get_preset_summary function."""

    def test_returns_no_preset_when_unset(self, monkeypatch: MonkeyPatch) -> None:
        """Should return custom configuration message when no profile set."""
        monkeypatch.delenv("REFLECTLOG_PROFILE", raising=False)
        assert get_preset_summary() == "No preset (custom configuration)"

    def test_returns_no_preset_for_custom(self, monkeypatch: MonkeyPatch) -> None:
        """Should return custom configuration message for "custom" profile."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", "custom")
        assert get_preset_summary() == "No preset (custom configuration)"

    @pytest.mark.parametrize(
        "profile_name,expected_upper",
        [
            ("simple", "SIMPLE"),
            ("balanced", "BALANCED"),
            ("performance", "PERFORMANCE"),
            ("quality", "QUALITY"),
        ],
    )
    def test_returns_active_preset_name_uppercased(
        self, monkeypatch: MonkeyPatch, profile_name: str, expected_upper: str
    ) -> None:
        """Should return formatted string with uppercased preset name."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", profile_name)
        result = get_preset_summary()
        assert result == f"Active preset: {expected_upper}"

    def test_returns_no_preset_for_unknown_profile(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Unknown profile should return custom configuration message."""
        monkeypatch.setenv("REFLECTLOG_PROFILE", "nonexistent")
        assert get_preset_summary() == "No preset (custom configuration)"
