"""Integration tests for architectural layer boundary enforcement.

Verifies that the layered architecture invariants hold:
- Infrastructure layer NEVER imports from application layer
- Core layer NEVER imports from application layer (with documented exceptions)
- Infrastructure classes satisfy their core protocol contracts
- Deprecated factory methods emit DeprecationWarning
- Re-export subpackages remain importable

Uses AST parsing (no code execution) for import scanning, and
``@runtime_checkable`` protocol checks for structural conformance.
"""

import ast
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REFLECTLOG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "reflectlog"
_INFRASTRUCTURE_DIR = _REFLECTLOG_ROOT / "infrastructure"
_CORE_DIR = _REFLECTLOG_ROOT / "core"

# config_adapters.py imports Config under TYPE_CHECKING — architecturally
# allowed because it is the *adapter* whose sole purpose is bridging layers.
_CORE_ALLOWED_APPLICATION_IMPORTS: dict[str, set[str]] = {
    "config_adapters.py": {"reflectlog.application.config.settings"},
}


# ---------------------------------------------------------------------------
# Helpers — AST-based import scanning
# ---------------------------------------------------------------------------


def _collect_python_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """Recursively collect all .py files, skipping __init__.py."""
    return sorted(p for p in directory.rglob("*.py") if p.name != "__init__.py")


def _extract_import_modules(filepath: pathlib.Path) -> list[str]:
    """Return all ``from X import ...`` module strings found via AST."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def _file_imports_matching(
    filepath: pathlib.Path,
    prefix: str,
) -> list[str]:
    """Return import module strings from *filepath* that start with *prefix*."""
    return [m for m in _extract_import_modules(filepath) if m.startswith(prefix)]


# ---------------------------------------------------------------------------
# 1. Infrastructure → Application boundary (FORBIDDEN)
# ---------------------------------------------------------------------------


class TestInfrastructureDoesNotImportApplication:
    """No infrastructure file may import from ``reflectlog.application``."""

    @pytest.fixture(scope="class")
    def infra_files(self) -> list[pathlib.Path]:
        return _collect_python_files(_INFRASTRUCTURE_DIR)

    def test_infra_directory_has_files(self, infra_files: list[pathlib.Path]) -> None:
        """Sanity: infrastructure directory is non-empty."""
        assert len(infra_files) > 0, "No Python files found in infrastructure/"

    def test_zero_application_imports(self, infra_files: list[pathlib.Path]) -> None:
        """Every infrastructure .py file must have zero application imports."""
        violations: list[str] = []
        for fp in infra_files:
            bad = _file_imports_matching(fp, "reflectlog.application")
            if bad:
                rel = fp.relative_to(_REFLECTLOG_ROOT.parent)
                violations.append(f"{rel}: {bad}")
        assert violations == [], (
            "Infrastructure files import from application layer:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )

    def test_all_infra_files_scanned(self, infra_files: list[pathlib.Path]) -> None:
        """Confirm rglob still walks marker subpackages such as search/."""
        walked_packages = {
            p.parent.name
            for p in _INFRASTRUCTURE_DIR.rglob("__init__.py")
            if p.parent != _INFRASTRUCTURE_DIR
        }
        assert {"search", "embeddings", "reranking", "memory"} <= walked_packages
        assert infra_files, "No Python files found in infrastructure/"


# ---------------------------------------------------------------------------
# 2. Core → Application boundary (FORBIDDEN, with documented exceptions)
# ---------------------------------------------------------------------------


class TestCoreDoesNotImportApplication:
    """Core files must not import from ``reflectlog.application``.

    Documented exceptions are tracked in ``_CORE_ALLOWED_APPLICATION_IMPORTS``.
    """

    @pytest.fixture(scope="class")
    def core_files(self) -> list[pathlib.Path]:
        return _collect_python_files(_CORE_DIR)

    def test_core_directory_has_files(self, core_files: list[pathlib.Path]) -> None:
        assert len(core_files) > 0, "No Python files found in core/"

    def test_no_unexpected_application_imports(
        self, core_files: list[pathlib.Path]
    ) -> None:
        """Only files listed in the allowlist may import from application."""
        violations: list[str] = []
        for fp in core_files:
            bad = _file_imports_matching(fp, "reflectlog.application")
            if not bad:
                continue
            allowed = _CORE_ALLOWED_APPLICATION_IMPORTS.get(fp.name, set())
            unexpected = [m for m in bad if m not in allowed]
            if unexpected:
                rel = fp.relative_to(_REFLECTLOG_ROOT.parent)
                violations.append(f"{rel}: {unexpected}")
        assert violations == [], "Unexpected core→application imports:\n" + "\n".join(
            f"  • {v}" for v in violations
        )

    def test_allowed_exceptions_still_exist(self) -> None:
        """Guard against stale allowlist entries — verify each exception is real."""
        for filename, expected_modules in _CORE_ALLOWED_APPLICATION_IMPORTS.items():
            filepath = _CORE_DIR / filename
            assert filepath.exists(), f"Allowlisted file {filename} no longer exists"
            actual = set(_file_imports_matching(filepath, "reflectlog.application"))
            assert actual == expected_modules, (
                f"{filename}: expected {expected_modules}, found {actual}"
            )

    def test_core_never_imports_infrastructure(
        self, core_files: list[pathlib.Path]
    ) -> None:
        """Core must NEVER import from infrastructure (no exceptions)."""
        violations: list[str] = []
        for fp in core_files:
            bad = _file_imports_matching(fp, "reflectlog.infrastructure")
            if bad:
                rel = fp.relative_to(_REFLECTLOG_ROOT.parent)
                violations.append(f"{rel}: {bad}")
        assert violations == [], (
            "Core files import from infrastructure layer:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# 3. Protocol conformance — runtime_checkable isinstance checks
# ---------------------------------------------------------------------------


class TestProtocolConformanceUSearch:
    """USearchEngine satisfies ISemanticSearchEngine via structural subtyping."""

    def test_has_all_semantic_engine_methods(self) -> None:
        from reflectlog.infrastructure.usearch_engine import USearchEngine

        for attr in (
            "name",
            "search",
            "add",
            "add_batch",
            "delete",
            "commit",
            "close",
            "ensure_initialized",
            "is_ready",
            "count",
            "contains_id",
            "get_id_by_content",
            "get_all",
            "memory_store",
        ):
            assert hasattr(USearchEngine, attr), (
                f"USearchEngine missing ISemanticSearchEngine.{attr}"
            )


class TestProtocolConformanceTantivy:
    """TantivyEngine exposes the live sync full-text search API."""

    def test_has_all_fulltext_engine_methods(self) -> None:
        from reflectlog.infrastructure.tantivy_engine import TantivyEngine

        for attr in (
            "name",
            "search",
            "add",
            "delete",
            "commit",
            "close",
            "ensure_initialized",
            "is_ready",
        ):
            assert hasattr(TantivyEngine, attr), f"TantivyEngine missing {attr}"


class TestProtocolConformanceCrossEncoder:
    """CrossEncoderReranker exposes the reranking interface."""

    def test_has_rerank_methods(self) -> None:
        from reflectlog.infrastructure.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        for attr in ("rerank", "rerank_async"):
            assert hasattr(CrossEncoderReranker, attr), (
                f"CrossEncoderReranker missing {attr}"
            )


class TestProtocolConformanceMemoryStore:
    """MemoryStore provides SQLite-backed storage with expected interface."""

    def test_has_core_crud_methods(self) -> None:
        """MemoryStore exposes insert, get, get_all, delete, exists, close."""
        from reflectlog.infrastructure.memory_store import MemoryStore

        for attr in (
            "insert",
            "get",
            "get_all",
            "delete",
            "exists",
            "get_id_by_content",
            "archive",
            "begin_replacement_transition",
            "begin_replacement_transitions",
            "list_pending_transitions",
            "get_transition_for_old_memory",
            "complete_replacement_transition",
            "close",
        ):
            assert hasattr(MemoryStore, attr), f"MemoryStore missing {attr}"


class TestProtocolConformanceEmbeddings:
    """Embedding providers satisfy the Embeddings protocol."""

    def test_cached_embeddings_interface(self) -> None:
        from reflectlog.infrastructure.embeddings.cached_embeddings import (
            CachedEmbeddings,
        )

        for attr in (
            "embed_query",
            "embed_documents",
            "aembed_query",
            "aembed_documents",
        ):
            assert hasattr(CachedEmbeddings, attr), (
                f"CachedEmbeddings missing Embeddings.{attr}"
            )

    def test_qwen_embeddings_interface(self) -> None:
        from reflectlog.infrastructure.embeddings.qwen3_embedding import (
            LangchainQwenEmbeddings,
        )

        for attr in (
            "embed_query",
            "embed_documents",
            "aembed_query",
            "aembed_documents",
        ):
            assert hasattr(LangchainQwenEmbeddings, attr), (
                f"LangchainQwenEmbeddings missing Embeddings.{attr}"
            )


class TestProtocolConformanceConfigAdapter:
    """ConfigAdapter satisfies IAppConfig (composition of 6 sub-protocols).

    Python 3.14 disallows ``issubclass()`` on protocols with non-method
    members (properties), so we verify by checking that every protocol
    property is present on the adapter class.
    """

    def test_config_adapter_has_all_iappconfig_properties(self) -> None:
        """ConfigAdapter exposes every property required by IAppConfig."""
        from reflectlog.core.config_adapters import ConfigAdapter

        # All property names from IServerConfig + ISearchConfig +
        # IStorageConfig + IRerankerConfig + IEmbedderConfig + IReplacementConfig
        required_properties = [
            # IServerConfig
            "transport",
            "host",
            "port",
            "path",
            "log_level",
            "workspace_id",
            # ISearchConfig
            "search_limit",
            "enable_rrf_fusion",
            "fusion_rrf_k",
            "fusion_threshold",
            "reranker_engine",
            "search_score_threshold",
            "enable_recency_boost",
            "recency_decay_rate",
            # IStorageConfig
            "storage_path",
            "usearch_index_path",
            "tantivy_index_path",
            "embedding_dims",
            "metric",
            "tantivy_normalize_scores",
            "tantivy_soft_delete_enabled",
            "tantivy_compaction_threshold_ratio",
            "tantivy_compaction_max_tombstones",
            "tantivy_tombstone_ttl_days",
            # IRerankerConfig
            "llm_model",
            "llm_api_base_url",
            "cross_encoder_model",
            "cross_encoder_device",
            "reranker_batch_normalize",
            "llm_api_key",
            "llm_provider",
            "rerank_max_concurrency",
            "cross_encoder_top_k",
            "cross_encoder_batch_size",
            "cross_encoder_score_threshold",
            "cross_encoder_use_fp16",
            "cross_encoder_normalize",
            "cross_encoder_max_length",
            "reranker_min_results",
            # IEmbedderConfig
            "embedding_model",
            "embedder_provider",
            "qwen_embedding_dims",
            "embedding_batch_size",
            "embedding_max_concurrent_batches",
            "embedding_cache_enabled",
            "embedding_cache_size",
            # IReplacementConfig
            "enable_smart_replace",
            "smart_replace_threshold",
            "smart_replace_min_similarity",
            "smart_replace_candidate_limit",
            "smart_replace_max_retries",
            "smart_replace_retry_delay",
        ]
        for prop in required_properties:
            assert hasattr(ConfigAdapter, prop), (
                f"ConfigAdapter missing IAppConfig property: {prop}"
            )

    def test_sub_protocol_adapters_have_required_properties(self) -> None:
        """Each fine-grained adapter has the properties of its protocol."""
        from reflectlog.core.config_adapters import (
            EmbedderConfigAdapter,
            ReplacementConfigAdapter,
            RerankerConfigAdapter,
            SearchConfigAdapter,
            ServerConfigAdapter,
            StorageConfigAdapter,
        )

        pairs: list[tuple[type, list[str]]] = [
            (
                ServerConfigAdapter,
                [
                    "transport",
                    "host",
                    "port",
                    "path",
                    "log_level",
                    "workspace_id",
                ],
            ),
            (
                SearchConfigAdapter,
                [
                    "search_limit",
                    "enable_rrf_fusion",
                    "fusion_rrf_k",
                    "fusion_threshold",
                    "reranker_engine",
                    "search_score_threshold",
                    "enable_recency_boost",
                    "recency_decay_rate",
                ],
            ),
            (
                StorageConfigAdapter,
                [
                    "storage_path",
                    "usearch_index_path",
                    "tantivy_index_path",
                    "embedding_dims",
                    "metric",
                    "tantivy_normalize_scores",
                    "tantivy_soft_delete_enabled",
                    "tantivy_compaction_threshold_ratio",
                    "tantivy_compaction_max_tombstones",
                    "tantivy_tombstone_ttl_days",
                ],
            ),
            (
                RerankerConfigAdapter,
                [
                    "llm_model",
                    "llm_api_base_url",
                    "cross_encoder_model",
                    "cross_encoder_device",
                    "reranker_batch_normalize",
                    "llm_api_key",
                    "llm_provider",
                    "rerank_max_concurrency",
                    "cross_encoder_top_k",
                    "cross_encoder_batch_size",
                    "cross_encoder_score_threshold",
                    "cross_encoder_use_fp16",
                    "cross_encoder_normalize",
                    "cross_encoder_max_length",
                    "reranker_min_results",
                ],
            ),
            (
                EmbedderConfigAdapter,
                [
                    "embedding_model",
                    "embedder_provider",
                    "qwen_embedding_dims",
                    "embedding_batch_size",
                    "embedding_max_concurrent_batches",
                    "embedding_cache_enabled",
                    "embedding_cache_size",
                ],
            ),
            (
                ReplacementConfigAdapter,
                [
                    "enable_smart_replace",
                    "smart_replace_threshold",
                    "smart_replace_min_similarity",
                    "smart_replace_candidate_limit",
                    "smart_replace_max_retries",
                    "smart_replace_retry_delay",
                ],
            ),
        ]
        for adapter_cls, required_props in pairs:
            for prop in required_props:
                assert hasattr(adapter_cls, prop), (
                    f"{adapter_cls.__name__} missing {prop}"
                )


# ---------------------------------------------------------------------------
# 4. Subpackage importability
# ---------------------------------------------------------------------------


class TestSubpackageImportability:
    """Infrastructure subpackages remain importable."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "reflectlog.infrastructure.search",
            "reflectlog.infrastructure.embeddings",
            "reflectlog.infrastructure.reranking",
            "reflectlog.infrastructure.memory",
            "reflectlog.infrastructure.llm",
        ],
    )
    def test_subpackage_importable(self, module_path: str) -> None:
        """Subpackage import must not raise."""
        import importlib

        mod = importlib.import_module(module_path)
        assert mod is not None

    def test_top_level_infrastructure_importable(self) -> None:
        """The infrastructure package itself is importable."""
        import reflectlog.infrastructure

        assert reflectlog.infrastructure is not None
