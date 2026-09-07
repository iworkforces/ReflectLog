"""Configuration presets for different use cases.

Provides pre-configured settings profiles to simplify setup:
- simple: Minimal configuration, low resource usage
- balanced: Default production settings (default)
- performance: Maximum speed, higher resource usage
- quality: Maximum accuracy, slower operations

Select preset via environment variable:
    REFLECTLOG_PROFILE=simple|balanced|performance|quality

Individual settings can override preset values if specified.
"""

from dataclasses import dataclass
import os

from reflectlog.core.enums import ConfigProfile, RerankerEngine


@dataclass
class ConfigPreset:
    """Configuration preset with pre-defined settings.

    Fields override Config defaults when preset is active.
    None values mean keep default from environment variable or Config class.
    """

    name: str
    search_limit: int | None = None
    search_score_threshold: float | None = None
    fusion_rrf_k: int | None = None
    overfetch_multiplier: int | None = None
    reranker_engine: RerankerEngine | None = None
    enable_recency_boost: bool | None = None
    recency_decay_rate: float | None = None
    enable_smart_replace: bool | None = None
    smart_replace_threshold: float | None = None
    embedding_batch_size: int | None = None
    embedding_max_concurrent_batches: int | None = None
    enable_embedding_cache: bool | None = None
    embedding_cache_size: int | None = None
    usearch_exact_search: bool | None = None


# Preset definitions
SIMPLE_PRESET = ConfigPreset(
    name="simple",
    search_limit=3,
    reranker_engine=RerankerEngine.NONE,
    enable_recency_boost=False,
    enable_smart_replace=False,
    embedding_batch_size=128,
    embedding_max_concurrent_batches=2,
    enable_embedding_cache=True,
    embedding_cache_size=50,
    usearch_exact_search=True,
)

BALANCED_PRESET = ConfigPreset(
    name="balanced",
    search_limit=None,
    search_score_threshold=None,
    fusion_rrf_k=None,
    overfetch_multiplier=None,
    reranker_engine=None,
    enable_recency_boost=None,
    recency_decay_rate=None,
    enable_smart_replace=None,
    smart_replace_threshold=None,
    embedding_batch_size=None,
    embedding_max_concurrent_batches=None,
    enable_embedding_cache=None,
    embedding_cache_size=None,
    usearch_exact_search=None,
)

PERFORMANCE_PRESET = ConfigPreset(
    name="performance",
    search_limit=10,
    search_score_threshold=0.3,
    fusion_rrf_k=40,
    overfetch_multiplier=2,
    reranker_engine=RerankerEngine.NONE,
    enable_recency_boost=False,
    enable_smart_replace=False,
    embedding_batch_size=1024,
    embedding_max_concurrent_batches=8,
    enable_embedding_cache=True,
    embedding_cache_size=200,
    usearch_exact_search=False,
)

QUALITY_PRESET = ConfigPreset(
    name="quality",
    search_limit=5,
    search_score_threshold=0.7,
    fusion_rrf_k=80,
    overfetch_multiplier=5,
    reranker_engine=RerankerEngine.CROSS_ENCODER,
    enable_recency_boost=True,
    enable_smart_replace=True,
    smart_replace_threshold=0.9,
    embedding_batch_size=256,
    embedding_max_concurrent_batches=2,
    enable_embedding_cache=True,
    embedding_cache_size=200,
    usearch_exact_search=True,
)


PRESETS: dict[str, ConfigPreset] = {
    ConfigProfile.SIMPLE: SIMPLE_PRESET,
    ConfigProfile.BALANCED: BALANCED_PRESET,
    ConfigProfile.PERFORMANCE: PERFORMANCE_PRESET,
    ConfigProfile.QUALITY: QUALITY_PRESET,
}


def get_active_preset() -> ConfigPreset | None:
    """Get active configuration preset from environment.

    Returns:
        ConfigPreset if REFLECTLOG_PROFILE is set to valid value, None otherwise.
    """
    profile_name = os.getenv("REFLECTLOG_PROFILE", "").lower()

    if not profile_name or profile_name == ConfigProfile.CUSTOM:
        return None

    return PRESETS.get(profile_name.lower())


def apply_preset_to_env(preset: ConfigPreset) -> None:
    """Apply preset settings to environment variables.

    Sets environment variables that will be read by Config.from_environment().
    Explicitly set environment variables take precedence over preset.

    Args:
        preset: Configuration preset to apply.
    """
    if preset.search_limit is not None:
        os.environ["SEARCH_LIMIT"] = str(preset.search_limit)

    if preset.search_score_threshold is not None:
        os.environ["SEARCH_SCORE_THRESHOLD"] = str(preset.search_score_threshold)

    if preset.fusion_rrf_k is not None:
        os.environ["FUSION_RRF_K"] = str(preset.fusion_rrf_k)

    if preset.overfetch_multiplier is not None:
        os.environ["OVERFETCH_MULTIPLIER"] = str(preset.overfetch_multiplier)

    if preset.reranker_engine is not None:
        os.environ["RERANKER_ENGINE"] = preset.reranker_engine

    if preset.enable_recency_boost is not None:
        os.environ["ENABLE_RECENCY_BOOST"] = str(preset.enable_recency_boost).lower()

    if preset.recency_decay_rate is not None:
        os.environ["RECENCY_DECAY_RATE"] = str(preset.recency_decay_rate)

    if preset.enable_smart_replace is not None:
        os.environ["ENABLE_SMART_REPLACE"] = str(preset.enable_smart_replace).lower()

    if preset.smart_replace_threshold is not None:
        os.environ["SMART_REPLACE_THRESHOLD"] = str(preset.smart_replace_threshold)

    if preset.embedding_batch_size is not None:
        os.environ["EMBEDDING_BATCH_SIZE"] = str(preset.embedding_batch_size)

    if preset.embedding_max_concurrent_batches is not None:
        os.environ["EMBEDDING_MAX_CONCURRENT_BATCHES"] = str(
            preset.embedding_max_concurrent_batches
        )

    if preset.enable_embedding_cache is not None:
        os.environ["EMBEDDING_CACHE_ENABLED"] = str(
            preset.enable_embedding_cache
        ).lower()

    if preset.embedding_cache_size is not None:
        os.environ["EMBEDDING_CACHE_SIZE"] = str(preset.embedding_cache_size)

    if preset.usearch_exact_search is not None:
        os.environ["USEARCH_EXACT_SEARCH"] = str(preset.usearch_exact_search).lower()


def get_preset_summary() -> str:
    """Get summary of active preset.

    Returns:
        String describing active preset or "No preset (custom configuration)".
    """
    preset = get_active_preset()
    if preset is None:
        return "No preset (custom configuration)"

    return f"Active preset: {preset.name.upper()}"
