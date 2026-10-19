import { describe, expect, it } from "vitest";

import {
  addMemoriesSchema,
  getMemoryByIdSchema,
  getAllMemoriesSchema,
  parseToolResponse,
  deleteMemoriesSchema,
  searchMemoriesSchema,
  cursorMemoriesSchema, successorMemoryByIdSchema, explainedSearchMemoriesSchema,
} from "../../src/reflectlog/parse.js";
import { ReflectLogError } from "../../src/reflectlog/types.js";

const record = { id: 7, memory: "Prefers tea", created_at: "2026-10-10 00:00:00" };
const evidence = { semantic: { rank: 2, score: -0.3 }, fulltext: null,
  pre_rerank_score: 0.016, pre_rerank_score_kind: "rrf", final_score_kind: "reranker" };
const stages = { semantic: "applied", fulltext: "empty", reranker: "degraded" };

function caught(run: () => unknown): ReflectLogError {
  try {
    run();
  } catch (error) {
    if (error instanceof ReflectLogError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected a ReflectLogError");
}

describe("ReflectLog tool results", () => {
  it.each([0, 7])("decodes cursor continuation %i without adding it to offset pages", (cursor) => {
    const payload = { records: [], total: 0, offset: 0, limit: 500, truncated: false, next_after_id: cursor };
    expect(parseToolResponse({ structuredContent: payload }, cursorMemoriesSchema)).toHaveProperty("nextAfterId", cursor);
    expect(parseToolResponse({ structuredContent: payload }, getAllMemoriesSchema)).not.toHaveProperty("nextAfterId");
  });

  it.each([undefined, null, -1, 1.5, "7", Number.MAX_SAFE_INTEGER + 1])("rejects malformed cursor %s", (cursor) => {
    const payload = { records: [], total: 0, offset: 0, limit: 500, truncated: false, next_after_id: cursor };
    expect(caught(() => parseToolResponse({ structuredContent: payload }, cursorMemoriesSchema)).code).toBe("malformed");
  });

  it.each([null, { id: 9 }])("decodes a requested nullable successor %s", (successor) => {
    expect(parseToolResponse({ structuredContent: { record: null, successor } }, successorMemoryByIdSchema))
      .toEqual({ record: null, successor });
  });

  it.each([undefined, {}, { id: 0 }, { id: -1 }, { id: 1.2 }, { id: "9" }, { id: Number.MAX_SAFE_INTEGER + 1 }])("rejects a missing or malformed requested successor %s", (successor) => {
    expect(caught(() => parseToolResponse({ structuredContent: { record: null, successor } }, successorMemoryByIdSchema)).code).toBe("malformed");
  });

  it("accepts an existing original without successor metadata and strips unsolicited default hints", () => {
    expect(parseToolResponse({ structuredContent: { record } }, successorMemoryByIdSchema)).toHaveProperty("record.id", 7);
    expect(parseToolResponse({ structuredContent: { record: null, successor: { id: 9 } } }, getMemoryByIdSchema)).toEqual({ record: null });
  });

  it("preserves raw evidence and explicit absent backend contributions", () => {
    const payload = { results: [{ ...record, score: 0.83, explanation: evidence }], score_kind: "reranker", limit: 5, stage_outcomes: stages };
    expect(parseToolResponse({ structuredContent: payload }, explainedSearchMemoriesSchema)).toMatchObject({
      stageOutcomes: stages, results: [{ score: 0.83, explanation: { semantic: { rank: 2, score: -0.3 }, fulltext: null,
        preRerankScore: 0.016, preRerankScoreKind: "rrf", finalScoreKind: "reranker" } }],
    });
    expect(parseToolResponse({ structuredContent: payload }, searchMemoriesSchema)).not.toHaveProperty("stageOutcomes");
    expect(parseToolResponse({ structuredContent: payload }, searchMemoriesSchema).results[0]).not.toHaveProperty("explanation");
  });

  it.each(["disabled", "failed", "unavailable", "empty", "skipped_single_hit", "applied", "degraded"])("accepts stage outcome %s even with no hits", (stage) => {
    const stageOutcomes = { semantic: stage, fulltext: stage, reranker: stage };
    const payload = { results: [], score_kind: "none", limit: 5, stage_outcomes: stageOutcomes };
    expect(parseToolResponse({ structuredContent: payload }, explainedSearchMemoriesSchema)).toHaveProperty("stageOutcomes", stageOutcomes);
  });

  it.each([
    undefined, { ...evidence, semantic: undefined }, { ...evidence, fulltext: undefined },
    { ...evidence, semantic: { rank: 0, score: 1 } }, { ...evidence, semantic: { rank: 1.5, score: 1 } },
    { ...evidence, fulltext: { rank: 1, score: Infinity } }, { ...evidence, pre_rerank_score: NaN },
    { ...evidence, pre_rerank_score_kind: "bm25" }, { ...evidence, final_score_kind: "confidence" },
  ])("rejects missing or malformed hit evidence %s", (explanation) => {
    const payload = { results: [{ ...record, score: 0.8, explanation }], score_kind: "reranker", limit: 5, stage_outcomes: stages };
    expect(caught(() => parseToolResponse({ structuredContent: payload }, explainedSearchMemoriesSchema)).code).toBe("malformed");
  });

  it.each([undefined, { ...stages, reranker: "skipped" }, { semantic: "applied" }])("requires complete known stages on empty responses %s", (stage_outcomes) => {
    const payload = { results: [], score_kind: "none", limit: 5, stage_outcomes };
    expect(caught(() => parseToolResponse({ structuredContent: payload }, explainedSearchMemoriesSchema)).code).toBe("malformed");
  });

  it("maps search_memories to camelCase scored records", () => {
    const payload = { results: [{ ...record, score: 0.82 }], score_kind: "reranker", limit: 5 };
    expect(parseToolResponse({ structuredContent: payload }, searchMemoriesSchema)).toEqual({
      results: [{ id: 7, memory: "Prefers tea", createdAt: "2026-10-10 00:00:00", score: 0.82 }],
      scoreKind: "reranker",
      limit: 5,
    });
  });

  it("maps page and delete results", () => {
    expect(
      parseToolResponse(
        { structuredContent: { records: [{ ...record, created_at: null }], total: 3, offset: 1, limit: 1, truncated: true } },
        getAllMemoriesSchema,
      ),
    ).toEqual({
      records: [{ id: 7, memory: "Prefers tea", createdAt: null }],
      total: 3,
      offset: 1,
      limit: 1,
      truncated: true,
    });
    expect(
      parseToolResponse({ structuredContent: { removed: [7], missing: [] } }, deleteMemoriesSchema),
    ).toEqual({ removed: [7], missing: [] });
  });

  it("maps add results including replacement details", () => {
    const payload = {
      stored_count: 1,
      skipped_count: 0,
      replaced_count: 1,
      dry_run: false,
      replacements: [{ old_memory: "Likes coffee", new_memory: "Likes tea", confidence: 0.9 }],
    };
    expect(parseToolResponse({ structuredContent: payload }, addMemoriesSchema)).toEqual({
      storedCount: 1,
      skippedCount: 0,
      replacedCount: 1,
      dryRun: false,
      replacements: [{ oldMemory: "Likes coffee", newMemory: "Likes tea", confidence: 0.9, reason: null }],
    });
  });

  it.each(["structuredContent", "text"] as const)("maps a singular record via %s", (envelope) => {
    const payload = { record };
    const response = envelope === "structuredContent" ? { structuredContent: payload }
      : { content: [{ type: "text", text: JSON.stringify(payload) }] };
    expect(parseToolResponse(response, getMemoryByIdSchema)).toEqual({
      record: { id: 7, memory: "Prefers tea", createdAt: "2026-10-10 00:00:00" },
    });
  });

  it.each(["structuredContent", "text"] as const)("maps a missing memory to null via %s", (envelope) => {
    const payload = { record: null };
    const response = envelope === "structuredContent" ? { structuredContent: payload }
      : { content: [{ type: "text", text: JSON.stringify(payload) }] };
    expect(parseToolResponse(response, getMemoryByIdSchema)).toEqual({ record: null });
  });

  it.each(["structuredContent", "text"] as const)("rejects a missing record key via %s", (envelope) => {
    const response = envelope === "structuredContent" ? { structuredContent: {} }
      : { content: [{ type: "text", text: "{}" }] };
    expect(caught(() => parseToolResponse(response, getMemoryByIdSchema)).code).toBe("malformed");
  });

  it("falls back to JSON text when structured content is absent", () => {
    const response = { content: [{ type: "text", text: JSON.stringify({ removed: [], missing: [1] }) }] };
    expect(parseToolResponse(response, deleteMemoriesSchema)).toEqual({ removed: [], missing: [1] });
  });

  it("prefers structured content over text", () => {
    const response = {
      structuredContent: { removed: [2], missing: [] },
      content: [{ type: "text", text: "not json" }],
    };
    expect(parseToolResponse(response, deleteMemoriesSchema)).toEqual({ removed: [2], missing: [] });
  });

  it("classifies tool errors and bearer rejections", () => {
    expect(
      caught(() => parseToolResponse({ isError: true, content: [{ type: "text", text: "StorageError: busy" }] }, deleteMemoriesSchema)).code,
    ).toBe("tool_error");
    expect(
      caught(() => parseToolResponse({ isError: true, content: [{ type: "text", text: "Unauthorized" }] }, deleteMemoriesSchema)).code,
    ).toBe("unauthorized");
  });

  it("reports unexpected shapes and unparsable text as malformed", () => {
    expect(caught(() => parseToolResponse({ structuredContent: { removed: "7" } }, deleteMemoriesSchema)).code).toBe(
      "malformed",
    );
    expect(caught(() => parseToolResponse({ content: [{ type: "text", text: "{" }] }, deleteMemoriesSchema)).code).toBe(
      "malformed",
    );
    expect(caught(() => parseToolResponse({}, deleteMemoriesSchema)).code).toBe("malformed");
    expect(
      caught(() =>
        parseToolResponse(
          { structuredContent: { results: [{ ...record, score: 1 }], score_kind: "bm25", limit: 5 } },
          searchMemoriesSchema,
        ),
      ).code,
    ).toBe("malformed");
  });
});
