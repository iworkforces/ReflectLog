"""Import smoke tests for infrastructure layer cross-boundary patterns.

Verifies that all infrastructure modules import cleanly, their public APIs
are stable, and cross-boundary imports from the application layer work
correctly. This serves as a regression guard for refactoring these imports.
"""

import ast
import importlib
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFRASTRUCTURE_DIR = (
    pathlib.Path(__file__).resolve().parents[3] / "reflectlog" / "infrastructure"
)

# No infrastructure modules should import from reflectlog.application anymore.
# This list was emptied after Wave 2 dependency inversion.
CROSS_BOUNDARY_MODULES: list[str] = []

# Expected baseline count of infrastructure files importing from application
EXPECTED_CROSS_BOUNDARY_COUNT = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_infrastructure_python_files() -> list[pathlib.Path]:
    """Return all top-level .py files in the infrastructure directory."""
    return sorted(p for p in INFRASTRUCTURE_DIR.glob("*.py") if p.name != "__init__.py")


def _file_imports_from_application(filepath: pathlib.Path) -> bool:
    """Check if a Python file has any ``from reflectlog.application`` imports.

    Uses AST parsing so the check works without executing the file.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("reflectlog.application")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Module import smoke tests (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", CROSS_BOUNDARY_MODULES)
def test_module_imports_cleanly(module_name: str) -> None:
    """Verify each cross-boundary infrastructure module imports without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerAPI:
    """Verify cross_encoder_reranker public API surface."""

    def test_exports_exist(self) -> None:
        """CrossEncoderReranker, CrossEncoderConfig are importable."""
        from reflectlog.infrastructure.cross_encoder_reranker import (
            CrossEncoderConfig,
            CrossEncoderReranker,
        )

        assert CrossEncoderConfig is not None
        assert CrossEncoderReranker is not None

    def test_reranker_has_rerank_and_rerank_async(self) -> None:
        """CrossEncoderReranker must expose rerank and rerank_async methods."""
        from reflectlog.infrastructure.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        assert hasattr(CrossEncoderReranker, "rerank")
        assert hasattr(CrossEncoderReranker, "rerank_async")


class TestUSearchEngineAPI:
    """Verify usearch_engine public API surface."""

    def test_exports_exist(self) -> None:
        """USearchEngine and USearchConfig are importable."""
        from reflectlog.infrastructure.usearch_engine import (
            USearchConfig,
            USearchEngine,
        )

        assert USearchConfig is not None
        assert USearchEngine is not None

    def test_engine_has_search_backend_methods(self) -> None:
        """USearchEngine must expose search, add, delete, commit, close."""
        from reflectlog.infrastructure.usearch_engine import USearchEngine

        for method_name in (
            "search",
            "add",
            "delete",
            "commit",
            "close",
            "ensure_initialized",
            "is_ready",
        ):
            assert hasattr(USearchEngine, method_name), f"Missing {method_name}"

    def test_engine_has_name_property(self) -> None:
        """USearchEngine must have a name property."""
        from reflectlog.infrastructure.usearch_engine import USearchEngine

        assert hasattr(USearchEngine, "name")


class TestSmartReplacerAPI:
    """Verify smart_replacer public API surface."""

    def test_exports_exist(self) -> None:
        """SmartReplacer, SmartReplacerConfig, ReplacementDecision are importable."""
        from reflectlog.infrastructure.smart_replacer import (
            OpenAIReplacementProvider,
            ReplacementDecision,
            SmartReplacer,
            SmartReplacerConfig,
            create_replacement_provider,
        )

        assert callable(create_replacement_provider)
        assert SmartReplacer is not None
        assert SmartReplacerConfig is not None
        assert ReplacementDecision is not None
        assert OpenAIReplacementProvider is not None

    def test_smart_replacer_has_check_replacement(self) -> None:
        """SmartReplacer must expose check_replacement method."""
        from reflectlog.infrastructure.smart_replacer import SmartReplacer

        assert hasattr(SmartReplacer, "check_replacement")


class TestCachedEmbeddingsAPI:
    """Verify cached_embeddings public API surface."""

    def test_exports_exist(self) -> None:
        """CachedEmbeddings is importable."""
        from reflectlog.infrastructure.embeddings.cached_embeddings import (
            CachedEmbeddings,
        )

        assert CachedEmbeddings is not None

    def test_has_embedding_methods(self) -> None:
        """CachedEmbeddings must expose embed_query, embed_documents, and async variants."""
        from reflectlog.infrastructure.embeddings.cached_embeddings import (
            CachedEmbeddings,
        )

        for method_name in (
            "embed_query",
            "embed_documents",
            "aembed_query",
            "aembed_documents",
        ):
            assert hasattr(CachedEmbeddings, method_name), f"Missing {method_name}"

    def test_has_cache_management_methods(self) -> None:
        """CachedEmbeddings must expose get_cache_stats and clear_cache."""
        from reflectlog.infrastructure.embeddings.cached_embeddings import (
            CachedEmbeddings,
        )

        assert hasattr(CachedEmbeddings, "get_cache_stats")
        assert hasattr(CachedEmbeddings, "clear_cache")


class TestLLMProviderBaseAPI:
    """Verify llm_provider_base public API surface."""

    def test_exports_exist(self) -> None:
        """BaseOpenAIProvider and IStructuredOutputSchema are importable."""
        from reflectlog.infrastructure.llm_provider_base import (
            BaseOpenAIProvider,
            IStructuredOutputSchema,
        )

        assert BaseOpenAIProvider is not None
        assert IStructuredOutputSchema is not None

    def test_base_provider_has_structured_output_method(self) -> None:
        """BaseOpenAIProvider must expose _call_llm_with_structured_output."""
        from reflectlog.infrastructure.llm_provider_base import BaseOpenAIProvider

        assert hasattr(BaseOpenAIProvider, "_call_llm_with_structured_output")


class TestQwen3EmbeddingAPI:
    """Verify qwen3_embedding public API surface."""

    def test_exports_exist(self) -> None:
        """LangchainQwenEmbeddings and EmbedderConfig are importable."""
        from reflectlog.infrastructure.embeddings.qwen3_embedding import (
            EmbedderConfig,
            LangchainQwenEmbeddings,
        )

        assert EmbedderConfig is not None
        assert LangchainQwenEmbeddings is not None

    def test_has_embedding_methods(self) -> None:
        """LangchainQwenEmbeddings must expose embed_query, embed_documents, and async variants."""
        from reflectlog.infrastructure.embeddings.qwen3_embedding import (
            LangchainQwenEmbeddings,
        )

        for method_name in (
            "embed_query",
            "embed_documents",
            "aembed_query",
            "aembed_documents",
        ):
            assert hasattr(LangchainQwenEmbeddings, method_name), (
                f"Missing {method_name}"
            )


class TestTantivyEngineAPI:
    """Verify tantivy_engine public API surface."""

    def test_exports_exist(self) -> None:
        """TantivyEngine and TantivyConfig are importable."""
        from reflectlog.infrastructure.tantivy_engine import (
            TantivyConfig,
            TantivyEngine,
        )

        assert TantivyConfig is not None
        assert TantivyEngine is not None

    def test_engine_has_search_backend_methods(self) -> None:
        """TantivyEngine must expose search, add, delete, commit, close."""
        from reflectlog.infrastructure.tantivy_engine import TantivyEngine

        for method_name in (
            "search",
            "add",
            "delete",
            "commit",
            "close",
            "ensure_initialized",
        ):
            assert hasattr(TantivyEngine, method_name), f"Missing {method_name}"

    def test_engine_has_name_property(self) -> None:
        """TantivyEngine must have a name property."""
        from reflectlog.infrastructure.tantivy_engine import TantivyEngine

        assert hasattr(TantivyEngine, "name")

    def test_engine_has_soft_delete(self) -> None:
        """TantivyEngine must expose soft_delete method."""
        from reflectlog.infrastructure.tantivy_engine import TantivyEngine

        assert hasattr(TantivyEngine, "soft_delete")

    def test_engine_has_compaction_methods(self) -> None:
        """TantivyEngine must expose compact, needs_compaction, get_tombstone_stats."""
        from reflectlog.infrastructure.tantivy_engine import TantivyEngine

        for method_name in ("compact", "needs_compaction", "get_tombstone_stats"):
            assert hasattr(TantivyEngine, method_name), f"Missing {method_name}"


# ---------------------------------------------------------------------------
# Cross-boundary import verification
# ---------------------------------------------------------------------------


class TestCrossBoundaryImportBaseline:
    """Track which infrastructure files import from the application layer.

    This test establishes a baseline count so we can verify our refactoring
    doesn't accidentally introduce new cross-boundary imports or miss one.
    """

    def test_cross_boundary_count_matches_baseline(self) -> None:
        """Exactly EXPECTED_CROSS_BOUNDARY_COUNT infrastructure files import from application."""
        infra_files = _get_infrastructure_python_files()
        violating = [f.name for f in infra_files if _file_imports_from_application(f)]
        assert len(violating) == EXPECTED_CROSS_BOUNDARY_COUNT, (
            f"Expected {EXPECTED_CROSS_BOUNDARY_COUNT} cross-boundary files, "
            f"found {len(violating)}: {violating}"
        )

    def test_cross_boundary_files_are_known(self) -> None:
        """All cross-boundary files should be in our known set."""
        infra_files = _get_infrastructure_python_files()
        violating = {f.stem for f in infra_files if _file_imports_from_application(f)}
        expected_stems = {m.rsplit(".", 1)[-1] for m in CROSS_BOUNDARY_MODULES}
        assert violating == expected_stems, (
            f"Unknown cross-boundary files: {violating - expected_stems}"
        )

    @pytest.mark.parametrize("module_name", CROSS_BOUNDARY_MODULES)
    def test_each_cross_boundary_module_actually_imports_from_application(
        self, module_name: str
    ) -> None:
        """Each listed module actually imports from reflectlog.application."""
        stem = module_name.rsplit(".", 1)[-1]
        filepath = INFRASTRUCTURE_DIR / f"{stem}.py"
        assert filepath.exists(), f"{filepath} does not exist"
        assert _file_imports_from_application(filepath), (
            f"{module_name} does not import from reflectlog.application"
        )


# ---------------------------------------------------------------------------
# Protocol conformance checks (structural typing)
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify infrastructure classes satisfy their core protocols."""

    def test_usearch_engine_is_search_backend(self) -> None:
        """USearchEngine must have all ISemanticSearchEngine methods."""
        from reflectlog.infrastructure.usearch_engine import USearchEngine

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
            assert hasattr(USearchEngine, attr), f"USearchEngine missing {attr}"

    def test_tantivy_engine_is_search_backend(self) -> None:
        """TantivyEngine must have the live sync full-text search methods."""
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

    def test_cross_encoder_reranker_is_ireranker(self) -> None:
        """CrossEncoderReranker must have all IReranker protocol methods."""
        from reflectlog.infrastructure.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        for attr in ("rerank", "rerank_async"):
            assert hasattr(CrossEncoderReranker, attr), (
                f"CrossEncoderReranker missing {attr}"
            )

    def test_cached_embeddings_has_embeddings_interface(self) -> None:
        """CachedEmbeddings must have the Embeddings protocol methods."""
        from reflectlog.infrastructure.embeddings.cached_embeddings import (
            CachedEmbeddings,
        )

        for method_name in (
            "embed_query",
            "embed_documents",
            "aembed_query",
            "aembed_documents",
        ):
            assert hasattr(CachedEmbeddings, method_name), (
                f"CachedEmbeddings missing {method_name}"
            )

    def test_qwen3_embeddings_has_embeddings_interface(self) -> None:
        """LangchainQwenEmbeddings must have the Embeddings protocol methods."""
        from reflectlog.infrastructure.embeddings.qwen3_embedding import (
            LangchainQwenEmbeddings,
        )

        for method_name in (
            "embed_query",
            "embed_documents",
            "aembed_query",
            "aembed_documents",
        ):
            assert hasattr(LangchainQwenEmbeddings, method_name), (
                f"LangchainQwenEmbeddings missing {method_name}"
            )
