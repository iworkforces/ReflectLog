"""Engine factory for search engine initialization.

This module provides the EngineFactory class that encapsulates the creation
and configuration of search engine instances based on application configuration.
It enables testing with mock engines and supports new engine types without
modifying the factory interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reflectlog.application.memory.fusion import create_fusion_engine
from reflectlog.core.config_adapters import ConfigAdapter
from reflectlog.core.enums import EmbedderProvider, RerankerEngine, WeMMModel
from reflectlog.infrastructure.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
)
from reflectlog.infrastructure.embeddings.cached_embeddings import CachedEmbeddings
from reflectlog.infrastructure.embeddings.qwen3_embedding import LangchainQwenEmbeddings
from reflectlog.infrastructure.embeddings.wemm_embedding import (
    WeMMEmbeddingConfig,
    WeMMEmbeddings,
)
from reflectlog.infrastructure.smart_replacer import SmartReplacer, SmartReplacerConfig
from reflectlog.infrastructure.tantivy_engine import TantivyConfig, TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

if TYPE_CHECKING:
    from reflectlog.application.config.settings import Config
    from reflectlog.application.memory.fusion.base import FusionEngine
    from reflectlog.core.logging import IStructuredLogger
    from reflectlog.core.storage_coordination import IStorageCoordinator
    from reflectlog.core.types import Embeddings


@dataclass
class EngineFactoryResult:
    """Result of engine factory initialization."""

    semantic_engine: USearchEngine
    tantivy_engine: TantivyEngine
    fusion_engine: FusionEngine
    reranker_engine: RerankerEngine


class EngineFactory:
    """Factory for creating and configuring search engine instances.

    This factory encapsulates all engine initialization logic, making it
    easy to test with mock engines and add support for new engine types.

    Example:
        factory = EngineFactory()
        result = factory.create_engines(config, logger)
        semantic = result.semantic_engine
        tantivy = result.tantivy_engine
    """

    def __init__(self) -> None:
        """Initialize the engine factory."""
        pass

    def create_engines(
        self,
        config: Config,
        logger: IStructuredLogger | None,
        coordinator: IStorageCoordinator | None = None,
    ) -> EngineFactoryResult:
        """Create and configure all search engines based on configuration.

        Args:
            config: Application configuration.
            logger: Structured logger instance.
            coordinator: Optional workspace storage coordinator.

        Returns:
            EngineFactoryResult with all initialized engines.
        """
        # Create USearch semantic engine
        semantic_engine = self._create_semantic_engine(
            config, logger, coordinator=coordinator
        )

        # Create Tantivy full-text engine
        tantivy_engine = self._create_tantivy_engine(
            config, logger, coordinator=coordinator
        )

        # Create fusion engine for hybrid ranking
        fusion_engine = self._create_fusion_engine(config, logger)

        return EngineFactoryResult(
            semantic_engine=semantic_engine,
            tantivy_engine=tantivy_engine,
            fusion_engine=fusion_engine,
            reranker_engine=config.reranker_engine,
        )

    def _create_semantic_engine(
        self,
        config: Config,
        logger: IStructuredLogger | None,
        coordinator: IStorageCoordinator | None = None,
    ) -> USearchEngine:
        """Create and configure USearch semantic engine.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            Configured USearchEngine instance.
        """
        usearch_config = USearchConfig.from_config(ConfigAdapter(config))
        embedder = self._create_embedder(config, logger)
        return USearchEngine(
            usearch_config,
            embedder=embedder,
            logger=logger,
            coordinator=coordinator,
        )

    def _create_embedder(
        self,
        config: Config,
        logger: IStructuredLogger | None,
    ) -> Embeddings:
        """Create embedder with optional caching.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            Embedder instance (possibly wrapped with caching).
        """
        match config.embedder_provider:
            case EmbedderProvider.OPENAI:
                base_embedder: Embeddings = LangchainQwenEmbeddings(
                    config={
                        "model": config.embedding_model,
                        "embedding_dims": config.embedding_dims,
                        "api_key": config.openrouter_api_key.get_secret_value(),
                        "openai_base_url": config.openrouter_base_url,
                        "batch_size": config.embedding_batch_size,
                        "max_concurrent_batches": config.embedding_max_concurrent_batches,
                    }
                )
            case EmbedderProvider.LANGCHAIN:
                base_embedder = LangchainQwenEmbeddings(
                    config={
                        "model": config.embedding_model,
                        "embedding_dims": config.qwen_embedding_dims,
                        "api_key": config.openrouter_api_key.get_secret_value(),
                        "openai_base_url": config.openrouter_base_url,
                        "batch_size": config.embedding_batch_size,
                        "max_concurrent_batches": config.embedding_max_concurrent_batches,
                    }
                )
            case EmbedderProvider.WEMM:
                base_embedder = WeMMEmbeddings(
                    WeMMEmbeddingConfig(
                        model=WeMMModel.from_config(config.embedding_model),
                        dimensions=config.wemm_embedding_dims,
                        device=config.wemm_device,
                        batch_size=config.embedding_batch_size,
                    )
                )
        if config.embedding_cache_enabled:
            match config.embedder_provider:
                case EmbedderProvider.WEMM:
                    return CachedEmbeddings(
                        embedder=base_embedder,
                        cache_size=config.embedding_cache_size,
                        enabled=True,
                        role_separated=True,
                        logger=logger,
                    )
                case EmbedderProvider.OPENAI | EmbedderProvider.LANGCHAIN:
                    return CachedEmbeddings(
                        embedder=base_embedder,
                        cache_size=config.embedding_cache_size,
                        enabled=True,
                        logger=logger,
                    )
        return base_embedder

    def _create_tantivy_engine(
        self,
        config: Config,
        logger: IStructuredLogger | None,
        coordinator: IStorageCoordinator | None = None,
    ) -> TantivyEngine:
        """Create the Tantivy full-text engine.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            TantivyEngine instance.
        """
        tantivy_config = TantivyConfig(
            workspace_id=config.workspace_id,
            index_path=config.tantivy_index_path_template.format(
                workspace_id=config.workspace_id
            ).lower(),
            normalize_scores=config.tantivy_normalize_scores,
            soft_delete_enabled=config.tantivy_soft_delete_enabled,
            compaction_threshold_ratio=config.tantivy_compaction_threshold_ratio,
            compaction_max_tombstones=config.tantivy_compaction_max_tombstones,
            tombstone_ttl_days=config.tantivy_tombstone_ttl_days,
        )
        return TantivyEngine(tantivy_config, logger=logger, coordinator=coordinator)

    def _create_fusion_engine(
        self,
        config: Config,
        logger: IStructuredLogger | None,
    ) -> FusionEngine:
        """Create fusion engine for hybrid ranking.

        Args:
            config: Application configuration.
            logger: Structured logger instance.

        Returns:
            Configured FusionEngine instance.
        """
        fusion_weights = config.fusion_weights
        return create_fusion_engine(
            method=config.fusion_method,
            normalization=config.fusion_normalization,
            rrf_k=config.fusion_rrf_k,
            weights=fusion_weights if isinstance(fusion_weights, list) else None,
            logger=logger,
        )


def create_cross_encoder_reranker(
    config: Config,
    logger: IStructuredLogger | None,
) -> CrossEncoderReranker | None:
    """Create CrossEncoder reranker if configured.

    Args:
        config: Application configuration.
        logger: Structured logger instance.

    Returns:
        CrossEncoderReranker instance or None if not configured.
    """
    if config.reranker_engine != RerankerEngine.CROSS_ENCODER:
        return None

    ce_config = CrossEncoderConfig.from_config(ConfigAdapter(config))
    return CrossEncoderReranker(config=ce_config, logger=logger)


def create_smart_replacer(
    config: Config,
    logger: IStructuredLogger | None,
) -> SmartReplacer | None:
    """Create SmartReplacer if enabled.

    Args:
        config: Application configuration.
        logger: Structured logger instance.

    Returns:
        SmartReplacer instance or None if disabled.
    """
    if not config.enable_smart_replace:
        return None

    replacer_config = SmartReplacerConfig.from_config(ConfigAdapter(config))
    return SmartReplacer(config=replacer_config, logger=logger)
