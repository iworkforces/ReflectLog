"""Unit tests for RanxFusionEngine class."""

import os
from pathlib import Path
import subprocess
import sys
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.memory.fusion import create_fusion_engine
from reflectlog.application.memory.fusion.ranx_fusion import (
    SUPPORTED_METHODS,
    SUPPORTED_NORMALIZATIONS,
    RanxFusionEngine,
)
from reflectlog.utility.scoring import compute_weighted_rrf_scores_batch


@pytest.mark.unit
class TestRanxFusionEngineInitialization:
    """Test RanxFusionEngine initialization."""

    def test_default_values(self) -> None:
        """Test default method and k value."""
        engine = RanxFusionEngine()
        assert engine.method == "rrf"
        assert engine.rrf_k == 60

    def test_custom_method(self) -> None:
        """Test setting custom fusion method."""
        engine = RanxFusionEngine(method="sum")
        assert engine.method == "sum"

    def test_custom_rrf_k(self) -> None:
        """Test setting custom RRF k value."""
        engine = RanxFusionEngine(rrf_k=30)
        assert engine.rrf_k == 30

    def test_custom_normalization(self) -> None:
        """Test setting custom normalization."""
        engine = RanxFusionEngine(normalization="max")
        assert engine.normalization == "max"

    def test_with_logger(self) -> None:
        """Test initialization with logger."""
        mock_logger = MagicMock()
        engine = RanxFusionEngine(logger=mock_logger)
        assert engine.logger is mock_logger

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported fusion method"):
            RanxFusionEngine(method="invalid_method")

    def test_invalid_normalization_raises_error(self) -> None:
        """Test that invalid normalization raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported normalization"):
            RanxFusionEngine(normalization="invalid_norm")


@pytest.mark.unit
class TestRanxFusionEngineSupportedMethods:
    """Test all supported fusion methods."""

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_supported_method_initializes(self, method: str) -> None:
        """Test that all supported methods can be initialized."""
        engine = RanxFusionEngine(method=method)
        assert engine.method == method

    @pytest.mark.parametrize("norm", SUPPORTED_NORMALIZATIONS)
    def test_supported_normalization_initializes(self, norm: str) -> None:
        """Test that all supported normalizations can be initialized."""
        engine = RanxFusionEngine(normalization=norm)
        assert engine.normalization == norm


@pytest.mark.unit
class TestRanxFusionEngineAlgorithm:
    """Test RanxFusionEngine fusion algorithm correctness."""

    @pytest.fixture
    def engine(self) -> RanxFusionEngine:
        """Create RanxFusionEngine instance with RRF."""
        return RanxFusionEngine(method="rrf", rrf_k=60)

    def test_empty_inputs(self, engine: RanxFusionEngine) -> None:
        """Test fusion with empty input lists."""
        result = engine.fuse([], [])
        assert result == []

    def test_single_result_set(self, engine: RanxFusionEngine) -> None:
        """Test fusion with single non-empty result set."""
        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.8)]
        results2: List[Tuple[str, float]] = []

        fused = engine.fuse(results1, results2)

        assert len(fused) == 2
        # Order should be preserved (A was ranked higher)
        assert fused[0][0] == "A"
        assert fused[1][0] == "B"
        assert fused[0][1] == pytest.approx(0.9)
        assert fused[1][1] == pytest.approx(0.8)

    def test_single_list_keeps_near_tied_scores_above_threshold(
        self, engine: RanxFusionEngine
    ) -> None:
        """A single backend list must not min-max near-ties under 0.8."""
        results1: List[Tuple[str, float]] = [("near-a", 0.91), ("near-b", 0.89)]
        results2: List[Tuple[str, float]] = []

        with patch.object(engine, "_fuse_rrf_numba") as numba:
            fused = engine.fuse(results1, results2)

        numba.assert_not_called()
        assert [name for name, _score in fused] == ["near-a", "near-b"]
        assert fused[0][1] == pytest.approx(0.91)
        assert fused[1][1] == pytest.approx(0.89)
        assert all(score >= 0.8 for _name, score in fused)

    def test_disjoint_rank1_ties_keep_raw_rrf(self, engine: RanxFusionEngine) -> None:
        """Disjoint rank-1s keep raw RRF instead of being min-maxed to 1.0."""
        results1: List[Tuple[str, float]] = [("semantic-only", 0.65)]
        results2: List[Tuple[str, float]] = [("lexical-only", 0.40)]

        fused = engine.fuse(results1, results2)

        assert {name for name, _score in fused} == {"semantic-only", "lexical-only"}
        raw = 1.0 / (60 + 1)
        assert all(score == pytest.approx(raw) for _name, score in fused)

    def test_disjoint_rank2_survives_without_minmax(
        self, engine: RanxFusionEngine
    ) -> None:
        """Rank-2 unique hit stays above a raw-RRF 0.0 gate (was dropped at 0.8)."""
        semantic: List[Tuple[str, float]] = [("A", 0.9), ("C", 0.7)]
        lexical: List[Tuple[str, float]] = [("B", 0.8)]

        fused = engine.fuse(semantic, lexical)
        names = [name for name, _score in fused]
        assert set(names) == {"A", "B", "C"}
        scores = {name: score for name, score in fused}
        assert scores["C"] > 0.0
        assert scores["C"] == pytest.approx(1.0 / (60 + 2))

    def test_weighted_rrf_uses_numba_weights(self, engine: RanxFusionEngine) -> None:
        """Weighted RRF must change rank order versus unweighted RRF."""
        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.8)]
        results2: List[Tuple[str, float]] = [("B", 0.9), ("A", 0.1)]
        weighted = RanxFusionEngine(method="rrf", rrf_k=60, weights=[1.0, 2.0])
        with patch(
            "reflectlog.application.memory.fusion.ranx_fusion."
            "compute_weighted_rrf_scores_batch",
            wraps=compute_weighted_rrf_scores_batch,
        ) as weighted_rrf:
            fused = weighted.fuse(results1, results2)

        assert weighted_rrf.called
        assert [name for name, _score in fused][0] == "B"

    def test_document_in_both_lists_ranks_higher(
        self, engine: RanxFusionEngine
    ) -> None:
        """Test that a document appearing in both lists gets higher score."""
        # B appears in both lists
        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.8)]
        results2: List[Tuple[str, float]] = [("B", 0.9), ("C", 0.8)]

        fused = engine.fuse(results1, results2)

        # B should be first because it appears in both lists
        assert fused[0][0] == "B"

    def test_unique_documents_preserved(self, engine: RanxFusionEngine) -> None:
        """Test that unique documents from each list are preserved."""
        results1: List[Tuple[str, float]] = [("A", 0.9)]
        results2: List[Tuple[str, float]] = [("B", 0.9)]

        fused = engine.fuse(results1, results2)

        doc_names = [f[0] for f in fused]
        assert "A" in doc_names
        assert "B" in doc_names

    def test_three_result_sets(self, engine: RanxFusionEngine) -> None:
        """Test fusion with three result sets."""
        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.8)]
        results2: List[Tuple[str, float]] = [("B", 0.9), ("C", 0.8)]
        results3: List[Tuple[str, float]] = [("B", 0.9), ("D", 0.8)]

        fused = engine.fuse(results1, results2, results3)

        # B appears in all three lists, should be first
        assert fused[0][0] == "B"

    def test_scores_are_descending(self, engine: RanxFusionEngine) -> None:
        """Test that results are sorted by score descending."""
        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        results2: List[Tuple[str, float]] = [("D", 0.9), ("E", 0.8), ("F", 0.7)]

        fused = engine.fuse(results1, results2)

        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestRanxFusionEngineLogging:
    """Test RanxFusionEngine logging behavior."""

    def test_logs_debug_info_when_logger_provided(self) -> None:
        """Test that debug info is logged when logger is provided."""
        mock_logger = MagicMock()
        engine = RanxFusionEngine(logger=mock_logger)

        results1: List[Tuple[str, float]] = [("A", 0.9)]
        results2: List[Tuple[str, float]] = [("B", 0.8)]

        engine.fuse(results1, results2)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert "Fusion completed" in call_args[0][0]

    def test_no_error_when_no_logger(self) -> None:
        """Test that fusion works without logger."""
        engine = RanxFusionEngine(logger=None)

        results1: List[Tuple[str, float]] = [("A", 0.9)]
        results2: List[Tuple[str, float]] = [("B", 0.8)]

        # Should not raise
        fused = engine.fuse(results1, results2)
        assert len(fused) == 2


@pytest.mark.unit
class TestRanxFusionEngineEdgeCases:
    """Test RanxFusionEngine edge cases."""

    @pytest.fixture
    def engine(self) -> RanxFusionEngine:
        """Create RanxFusionEngine instance."""
        return RanxFusionEngine(method="rrf", rrf_k=60)

    def test_duplicate_in_same_list(self, engine: RanxFusionEngine) -> None:
        """Test handling of duplicates within the same list."""
        # Same document appearing twice in one list (uses first occurrence)
        results1: List[Tuple[str, float]] = [("A", 0.9), ("A", 0.8)]
        results2: List[Tuple[str, float]] = []

        fused = engine.fuse(results1, results2)

        # A should appear once (deduped by ranx)
        assert len(fused) == 1
        assert fused[0][0] == "A"

    def test_very_long_lists(self, engine: RanxFusionEngine) -> None:
        """Test fusion with long lists."""
        results1 = [(f"doc_{i}", 1.0 - i * 0.01) for i in range(100)]
        results2 = [(f"doc_{i + 50}", 1.0 - i * 0.01) for i in range(100)]

        fused = engine.fuse(results1, results2)

        # Should have 150 unique documents (50 overlap)
        assert len(fused) == 150


@pytest.mark.unit
class TestRanxFusionEngineDifferentMethods:
    """Test different fusion methods."""

    def test_sum_fusion(self) -> None:
        """Test CombSUM fusion method (named 'sum' in ranx)."""
        engine = RanxFusionEngine(method="sum")

        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.8)]
        results2: List[Tuple[str, float]] = [("B", 0.9), ("C", 0.7)]

        fused = engine.fuse(results1, results2)

        # B should be first (highest combined score)
        assert fused[0][0] == "B"

    def test_max_fusion(self) -> None:
        """Test CombMAX fusion method (named 'max' in ranx)."""
        engine = RanxFusionEngine(method="max")

        results1: List[Tuple[str, float]] = [("A", 0.9), ("B", 0.5)]
        results2: List[Tuple[str, float]] = [("B", 0.8), ("C", 0.7)]

        fused = engine.fuse(results1, results2)

        # All scores should be non-negative
        assert all(score >= 0 for _, score in fused)

    def test_weighted_non_rrf_fails_closed_on_typeerror(self) -> None:
        """Non-RRF fusion must not silently drop weights on TypeError."""
        engine = RanxFusionEngine(method="sum", weights=[1.0, 0.5])
        results1: List[Tuple[str, float]] = [("A", 0.9)]
        results2: List[Tuple[str, float]] = [("B", 0.8)]

        with patch(
            "reflectlog.application.memory.fusion.ranx_fusion._load_ranx",
            return_value=(
                MagicMock(),
                MagicMock(side_effect=TypeError("weights not supported")),
            ),
        ):
            with pytest.raises(RuntimeError, match="refusing unweighted fallback"):
                engine.fuse(results1, results2)


@pytest.mark.unit
class TestCreateFusionEngineFactory:
    """Test the create_fusion_engine factory function."""

    def test_creates_ranx_engine(self) -> None:
        """Test that factory creates RanxFusionEngine."""
        engine = create_fusion_engine()
        assert isinstance(engine, RanxFusionEngine)

    def test_passes_method(self) -> None:
        """Test that method is passed correctly."""
        engine = create_fusion_engine(method="sum")
        assert engine.method == "sum"

    def test_passes_normalization(self) -> None:
        """Test that normalization is passed correctly."""
        engine = create_fusion_engine(normalization="max")
        assert engine.normalization == "max"

    def test_passes_rrf_k(self) -> None:
        """Test that rrf_k is passed correctly."""
        engine = create_fusion_engine(rrf_k=30)
        assert isinstance(engine, RanxFusionEngine)
        assert engine.rrf_k == 30

    def test_passes_logger(self) -> None:
        """Test that logger is passed correctly."""
        mock_logger = MagicMock()
        engine = create_fusion_engine(logger=mock_logger)
        assert isinstance(engine, RanxFusionEngine)
        assert engine.logger is mock_logger


@pytest.mark.unit
class TestRanxLazyImport:
    """Default RRF stays local; ranx loads only for ranx-backed methods."""

    def test_fresh_ranx_compile_fuses_and_preserves_unrelated_warnings(
        self, tmp_path: Path
    ) -> None:
        script = """
import warnings
from reflectlog.application.memory.fusion.ranx_fusion import RanxFusionEngine

warnings.simplefilter("error")
first = [("A", 0.9), ("B", 0.5)]
second = [("B", 0.8), ("C", 0.7)]
sum_scores = dict(RanxFusionEngine(method="sum").fuse(first, second))
max_scores = dict(RanxFusionEngine(method="max").fuse(first, second))
assert sum_scores == {"A": 0.9, "B": 1.3, "C": 0.7}
assert max_scores == {"A": 0.9, "B": 0.8, "C": 0.7}
print("fusion succeeded", flush=True)
warnings.warn("unrelated syntax warning", SyntaxWarning)
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-X",
                f"pycache_prefix={tmp_path / 'fresh-pycache'}",
                "-c",
                script,
            ],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
        )
        assert completed.stdout == "fusion succeeded\n", completed.stderr
        assert completed.returncode != 0
        assert "SyntaxWarning: unrelated syntax warning" in completed.stderr

    def test_local_rrf_does_not_import_ranx(self) -> None:
        script = """
import sys
from reflectlog.application.memory.fusion.ranx_fusion import RanxFusionEngine
engine = RanxFusionEngine(method="rrf")
fused = engine.fuse([("A", 0.9)], [("B", 0.8)])
assert fused[0][0] in {"A", "B"}
assert "ranx" not in sys.modules
"""
        completed = _run_fresh_python(script)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_ranx_method_imports_ranx_and_fuses(self) -> None:
        script = """
import sys
from reflectlog.application.memory.fusion.ranx_fusion import RanxFusionEngine
engine = RanxFusionEngine(method="sum")
fused = engine.fuse([("A", 0.9), ("B", 0.8)], [("B", 0.9), ("C", 0.7)])
assert fused[0][0] == "B"
assert "ranx" in sys.modules
"""
        completed = _run_fresh_python(script)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_unrelated_runtime_warning_still_fails(self) -> None:
        script = """
import warnings
warnings.simplefilter("error")
warnings.warn("unrelated fusion warning", RuntimeWarning)
"""
        completed = _run_fresh_python(script)
        assert completed.returncode != 0
        assert "RuntimeWarning" in completed.stderr


def _run_fresh_python(script: str):
    from pathlib import Path
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=None,
    )
