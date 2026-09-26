"""ranx-based fusion engine implementation."""

import logging
import math
from typing import TYPE_CHECKING, Any, final, override
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable

    from ranx import Run

    from reflectlog.core.logging import IStructuredLogger

import numpy as np

from reflectlog.core.enums import FusionMethod, FusionNormalization
from reflectlog.utility.scoring import (
    compute_rrf_scores_batch,
    compute_weighted_rrf_scores_batch,
)

from .base import FusionEngine

# Supported fusion methods (unsupervised, no training data required)
# Note: ranx uses 'sum', 'mnz', 'max' instead of 'combsum', 'combmnz', 'combmax'
SUPPORTED_METHODS: frozenset[FusionMethod] = frozenset(FusionMethod)

# Supported normalization strategies
SUPPORTED_NORMALIZATIONS: frozenset[FusionNormalization] = frozenset(
    FusionNormalization
)

# Default normalization per fusion method (applied to inputs before fusion)
# Note: We use None for RRF since it works on ranks, not scores
DEFAULT_NORMALIZATIONS: dict[FusionMethod, FusionNormalization | None] = {
    FusionMethod.RRF: None,
    FusionMethod.SUM: None,
    FusionMethod.MNZ: None,
    FusionMethod.MAX: None,
    FusionMethod.BORDAFUSE: None,
}

_ranx_run: type[Run] | None = None
_ranx_fuse: Callable[..., Run] | None = None


def _load_ranx() -> tuple[type[Run], Callable[..., Run]]:
    """Import ranx only for non-RRF fusion methods.

    Isolates ranx's known invalid-escape SyntaxWarning at this boundary so
    unrelated warnings still fail under warnings-as-errors.
    """
    global _ranx_run, _ranx_fuse
    loaded_run = _ranx_run
    loaded_fuse = _ranx_fuse
    if loaded_run is not None and loaded_fuse is not None:
        return loaded_run, loaded_fuse
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*is an invalid escape sequence.*",
            category=SyntaxWarning,
        )
        from ranx import Run
        from ranx import fuse as ranx_fuse
    _ranx_run = Run
    _ranx_fuse = ranx_fuse
    return Run, ranx_fuse


def _one_list_results(
    result_set: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Return one backend list without importing ranx.

    Duplicate memories keep the averaged score, matching the previous ranx
    Run conversion used for a single non-empty list.
    """
    score_sums: dict[str, float] = {}
    score_counts: dict[str, int] = {}
    for memory, score in result_set:
        if memory in score_sums:
            score_sums[memory] += score
            score_counts[memory] += 1
        else:
            score_sums[memory] = score
            score_counts[memory] = 1
    results = [
        (memory, score_sums[memory] / score_counts[memory]) for memory in score_sums
    ]
    results.sort(key=lambda item: item[1], reverse=True)
    return results


@final
class RanxFusionEngine(FusionEngine):
    """Fusion engine using the ranx library.

    This engine supports multiple fusion algorithms and normalization
    strategies provided by the ranx library. It converts between the
    internal (memory, score) tuple format and ranx's Run objects.

    Supported methods:
        - rrf: Reciprocal Rank Fusion (default)
        - sum: CombSUM (score addition)
        - mnz: CombMNZ (weighted score addition)
        - max: CombMAX (maximum score)
        - bordafuse: Borda voting fusion

    Supported normalizations:
        - min-max: Min-max normalization
        - max: Max normalization
        - sum: Sum normalization
        - zmuv: Zero-mean unit-variance normalization
        - rank: Rank-based normalization
        - borda: Borda count normalization
    """

    # Fixed query ID for single-query fusion scenario
    _QUERY_ID = "q"

    def __init__(
        self,
        method: FusionMethod | str = FusionMethod.RRF,
        normalization: FusionNormalization | str | None = None,
        rrf_k: int = 60,
        weights: list[float] | None = None,
        logger: IStructuredLogger | None = None,
    ) -> None:
        """Initialize the ranx fusion engine.

        Args:
            method: Fusion algorithm to use. One of: rrf, sum, mnz,
                   max, bordafuse. Defaults to 'rrf'.
            normalization: Score normalization strategy. One of: min-max, max,
                          sum, zmuv, rank, borda. Defaults to auto-select
                          based on method.
            rrf_k: RRF k parameter (only used when method='rrf'). Lower values
                  give more weight to top ranks. Defaults to 60.
            weights: Optional list of weights for weighted RRF fusion. Must have
                  at least 2 elements if provided. Defaults to None (equal weights).
            logger: Optional StructuredLogger instance for debug logging.

        Raises:
            ValueError: If method or normalization is not supported.
        """
        super().__init__()

        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported fusion method: '{method}'. "
                f"Supported methods: {sorted(member.value for member in FusionMethod)}"
            )

        if normalization is not None and normalization not in SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization: '{normalization}'. "
                f"Supported normalizations: "
                f"{sorted(member.value for member in FusionNormalization)}"
            )

        resolved_method = FusionMethod(method)
        self._method = resolved_method
        self._normalization = normalization or DEFAULT_NORMALIZATIONS.get(
            resolved_method
        )
        self._rrf_k = rrf_k
        self._weights = weights
        self.logger = logger

    @property
    def method(self) -> str:
        """Return the fusion method name."""
        return self._method

    @property
    def normalization(self) -> str | None:
        """Return the normalization strategy."""
        return self._normalization

    @property
    def rrf_k(self) -> int:
        """Return the RRF k parameter."""
        return self._rrf_k

    @property
    def weights(self) -> list[float] | None:
        """Return the fusion weights for weighted RRF."""
        return self._weights

    def _convert_to_run(self, result_set: list[tuple[str, float]], name: str) -> Run:
        """Convert (memory, score) tuples to a ranx Run object.

        Handles duplicate documents within the same result set by averaging their
        scores. This provides better fusion quality than keeping only the first
        occurrence, as multiple high-scoring instances from different sources
        indicate stronger relevance.

        Example:
            Input: [("doc1", 0.9), ("doc2", 0.5), ("doc1", 0.7)]
            Output: Run({"q": {"doc1": 0.8, "doc2": 0.5}})
            # doc1 score: (0.9 + 0.7) / 2 = 0.8

        Args:
            result_set: List of (memory, score) tuples. May contain duplicate
                       memories with different scores.
            name: Name for the Run object.

        Returns:
            ranx Run object with memories as document IDs. Duplicate memories
            are represented once with their averaged score.
        """
        run_cls, _fuse = _load_ranx()
        if not result_set:
            return run_cls({self._QUERY_ID: {}}, name=name)

        # Use memory content as doc_id, original score as value
        # Handle duplicates within a list by averaging their scores
        # This provides better fusion quality than keeping first occurrence
        doc_score_sums: dict[str, float] = {}
        doc_score_counts: dict[str, int] = {}
        for msg, score in result_set:
            if msg in doc_score_sums:
                doc_score_sums[msg] += score
                doc_score_counts[msg] += 1
            else:
                doc_score_sums[msg] = score
                doc_score_counts[msg] = 1

        # Compute averaged scores
        doc_scores: dict[str, float] = {
            msg: doc_score_sums[msg] / doc_score_counts[msg] for msg in doc_score_sums
        }

        return run_cls({self._QUERY_ID: doc_scores}, name=name)

    def _convert_from_run(self, run: Run) -> list[tuple[str, float]]:
        """Convert a ranx Run object back to (memory, score) tuples.

        Args:
            run: ranx Run object after fusion.

        Returns:
            List of (memory, fused_score) tuples sorted by score descending.
        """
        if self._QUERY_ID not in run.run:
            return []

        query_results = run.run[self._QUERY_ID]

        # Convert to list of tuples and sort by score descending
        results = [(doc_id, float(score)) for doc_id, score in query_results.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _fuse_rrf_numba(
        self, result_sets: list[list[tuple[str, float]]]
    ) -> list[tuple[str, float]]:
        """RRF via interned doc ids and the compiled batch scorer."""
        doc_index: dict[str, int] = {}
        docs: list[str] = []
        for result_set in result_sets:
            for memory, _score in result_set:
                if memory not in doc_index:
                    doc_index[memory] = len(docs)
                    docs.append(memory)

        ranks = np.zeros((len(docs), len(result_sets)), dtype=np.int64)
        for run_idx, result_set in enumerate(result_sets):
            seen: set[int] = set()
            for rank, (memory, _score) in enumerate(result_set, start=1):
                doc_idx = doc_index[memory]
                if doc_idx in seen:
                    continue
                ranks[doc_idx, run_idx] = rank
                seen.add(doc_idx)

        if self._weights is not None:
            if len(self._weights) != len(result_sets):
                raise ValueError(
                    f"RRF weights length {len(self._weights)} does not match "
                    f"{len(result_sets)} result sets"
                )
            weight_arr = np.asarray(self._weights, dtype=np.float64)
            scores = compute_weighted_rrf_scores_batch(ranks, weight_arr, k=self._rrf_k)
        else:
            scores = compute_rrf_scores_batch(ranks, k=self._rrf_k)
        paired = [(docs[idx], float(scores[idx])) for idx in range(len(docs))]
        paired.sort(key=lambda item: item[1], reverse=True)
        # Raw RRF stays on the 1/(k+rank) scale. Min-max stretch plus a
        # 0-1 gate drops near-ties (rank-2 becomes ~0.49).
        if self.logger:
            self.logger.debug(
                f"Fusion completed: {len(paired)} unique results "
                f"from {len(result_sets)} result sets",
                extra={
                    "method": self._method,
                    "input_sets": len(result_sets),
                    "unique_count": len(paired),
                },
            )
        return paired

    @override
    def fuse(
        self,
        *result_sets: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists using the configured algorithm.

        Args:
            *result_sets: Variable number of (memory, score) tuple lists.
                         Each list should be sorted by score descending.

        Returns:
            List of (memory, fused_score) tuples sorted by score descending.
        """
        # Filter out empty result sets
        non_empty = [rs for rs in result_sets if rs]
        if not non_empty:
            return []

        if len(non_empty) == 1:
            # Keep original backend scores without importing ranx.
            return _one_list_results(non_empty[0])

        if self._method == FusionMethod.RRF:
            return self._fuse_rrf_numba(non_empty)

        # Validate score ranges and log unusual inputs (debug level)
        if self.logger and self.logger.is_enabled_for(logging.DEBUG):
            for i, result_set in enumerate(non_empty):
                scores = [score for _, score in result_set]
                if not scores:
                    continue

                # Check for unusual score ranges
                has_negative = any(score < 0 for score in scores)
                has_nan = any(math.isnan(score) for score in scores)
                has_inf = any(math.isinf(score) for score in scores)
                max_score = max(scores)
                min_score = min(scores)

                # Log if any unusual conditions detected
                if has_negative or has_nan or has_inf:
                    self.logger.debug(
                        f"Unusual score range detected in result set {i}",
                        extra={
                            "result_set_index": i,
                            "score_count": len(scores),
                            "min_score": min_score,
                            "max_score": max_score,
                            "has_negative": has_negative,
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                        },
                    )

        # Convert each result set to a ranx Run
        runs = [
            self._convert_to_run(rs, name=f"run_{i}")
            for i, rs in enumerate(result_sets)
            if rs
        ]

        if not runs:
            return []

        # RRF already returned above. Remaining methods only take weights.
        params: dict[str, Any] | None = None
        if self._weights is not None:
            params = {"weights": self._weights}

        try:
            _run_cls, ranx_fuse = _load_ranx()
            combined = ranx_fuse(
                runs=runs,
                norm=self._normalization,
                method=self._method,
                params=params,
            )
        except TypeError as fuse_error:
            if params is not None and "weights" in params:
                raise RuntimeError(
                    "Fusion algorithm rejected weights; refusing unweighted fallback"
                ) from fuse_error
            raise

        # Convert back to tuple format. Do not min-max: a leftover 0.8
        # fusion gate then keeps only the top hit.
        sorted_results = self._convert_from_run(combined)

        if self.logger:
            self.logger.debug(
                f"Fusion completed: {len(sorted_results)} unique results "
                f"from {len(non_empty)} result sets",
                extra={
                    "method": self._method,
                    "normalization": self._normalization,
                    "input_sets": len(result_sets),
                    "non_empty_sets": len(non_empty),
                    "unique_count": len(sorted_results),
                },
            )

        return sorted_results
