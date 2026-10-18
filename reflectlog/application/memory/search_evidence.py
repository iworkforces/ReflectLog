"""Typed, request-local ranking evidence, independent of memory identity lookup."""

from dataclasses import dataclass, field
from enum import StrEnum

from reflectlog.core.enums import SearchScoreKind


class StageOutcome(StrEnum):
    DISABLED = "disabled"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    EMPTY = "empty"
    SKIPPED_SINGLE_HIT = "skipped_single_hit"
    APPLIED = "applied"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class BackendContribution:
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RankingExplanation:
    semantic: BackendContribution | None
    fulltext: BackendContribution | None
    pre_rerank_score: float
    pre_rerank_score_kind: SearchScoreKind
    final_score_kind: SearchScoreKind


@dataclass(frozen=True, slots=True)
class SearchStageOutcomes:
    semantic: StageOutcome
    fulltext: StageOutcome
    reranker: StageOutcome


@dataclass(slots=True)
class SearchEvidence:
    """Per-execution builder populated at the actual stage decision points."""

    semantic: dict[str, BackendContribution] = field(
        default_factory=dict[str, BackendContribution]
    )
    fulltext: dict[str, BackendContribution] = field(
        default_factory=dict[str, BackendContribution]
    )
    candidates: dict[str, float] = field(default_factory=dict[str, float])
    pre_rerank_score_kind: SearchScoreKind = SearchScoreKind.NONE
    semantic_outcome: StageOutcome = StageOutcome.EMPTY
    fulltext_outcome: StageOutcome = StageOutcome.EMPTY
    reranker_outcome: StageOutcome = StageOutcome.EMPTY

    def outcomes(self) -> SearchStageOutcomes:
        return SearchStageOutcomes(
            self.semantic_outcome, self.fulltext_outcome, self.reranker_outcome
        )

    def explanations(
        self, memories: list[str], final_kind: SearchScoreKind
    ) -> dict[str, RankingExplanation]:
        return {
            content: RankingExplanation(
                semantic=self.semantic.get(content),
                fulltext=self.fulltext.get(content),
                pre_rerank_score=self.candidates[content],
                pre_rerank_score_kind=self.pre_rerank_score_kind,
                final_score_kind=final_kind,
            )
            for content in memories
        }
